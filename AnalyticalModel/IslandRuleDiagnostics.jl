"""
    IslandRuleDiagnostics

A third, independent GCVB-based check, built from a *wall-plus-cut* region
rather than a closed `q = f/H` contour (`GCVBDiagnostics.jl`) or a closed
isobath (`IsobathCirculationDiagnostics.jl`). This is the direct analogue of
Godfrey's (1989) island rule, applied to this single connected (island-free)
basin by splitting it along an open zonal "cut" (here, the Fram Strait
transect at `y = cut_y`) instead of around a real island.

Integrate the GCVB equation,

    J(f/H, Ψ) = J(1/H, χ) - Ek·curl((τ + τ_b)/H),

over the region `R = {y > cut_y}` (the basin north of the cut). Green's
theorem converts the left-hand side to a line integral around `∂R`,

    ∬_R J(f/H, Ψ) dA = ∮_{∂R} (f/H) dΨ,

and `∂R` splits into the solid northern coastline arc (where `Ψ = 0`
identically, so `dΨ = 0` there) and the open cut itself. If the wall
contributes nothing, only the cut survives:

    ∮_{cut, west→east} (f/H) V dx  =  ∬_R [jebar - bottom_stress_curl - surface_stress_curl] dA,

i.e. the net `q`-weighted meridional transport through the cut is *predicted*
by the area integral of the forcing terms, none of which require solving for
the full flow (JEBAR and the wind-stress curl are known analytically from
the problem's forcing and topography; only the bottom-stress curl needs the
already-solved bottom pressure).

Rather than trying to *verify* the `Ψ = 0` wall assumption directly (which
would require evaluating the streamfunction at the true coastline -- exactly
where finite-element fields are least reliable, and where this codebase's
diagnostic grids are deliberately inset to avoid such artifacts), this
module takes the pragmatic approach suggested by the physics: treat the
right-hand side forcing integral as a closure-free *prediction* for the cut
transport, and report it next to the *independently observed* cut transport
(computed directly from the solved flow at the cut, which is comfortably
interior and needs no wall assumption at all). The gap between prediction
and observation is reported honestly as `wall_leakage`, rather than asserted
to be small: at this problem's standard inset diagnostic grid, direct
testing showed that `Ψ` decays roughly *linearly* approaching the coast
rather than across a thin boundary layer, so the area integral computed only
out to the inset edge of the sampling grid can omit an `O(1)` fraction of
the true region's contribution. `wall_leakage` is therefore itself
diagnostic of how much near-wall structure the sampling grid misses, not
merely of discretization error.

Separately, and with no wall assumption whatsoever, this module also
reports the pointwise GCVB identity's own area-integrated residual
(advection area integral vs. forcing area integral over the same region),
which is the ordinary closure-free consistency check already used by
`GCVBDiagnostics.jl` and should be small whenever the solve/sampling
resolution is adequate.

This module is self-contained (helpers are duplicated rather than imported,
as in `GCVBDiagnostics.jl` and `IsobathCirculationDiagnostics.jl`) so it can
be switched in or out independently.
"""
module IslandRuleDiagnostics

import Plots

"""
    trapezoidal_line_integral(xs, values)

Trapezoidal integral of `values` with respect to `xs`, skipping any gap
where either endpoint is not finite. `xs` must be sorted.
"""
function trapezoidal_line_integral(xs, values)
    total = 0.0
    for index in 1:(length(xs) - 1)
        a, b = values[index], values[index + 1]
        (isfinite(a) && isfinite(b)) || continue
        total += 0.5 * (a + b) * (xs[index + 1] - xs[index])
    end
    return total
end

"""
    area_integral(field, dxdy, mask)

Sum `field[i, j] * dxdy` over grid points where `mask[i, j]` is true and
`field[i, j]` is finite.
"""
function area_integral(field, dxdy, mask)
    total = 0.0
    for index in eachindex(field, mask)
        (mask[index] && isfinite(field[index])) || continue
        total += field[index] * dxdy
    end
    return total
end

"""
    half_basin_island_rule(solution, diagnostics, gcvb, wet, dxdy, cut_index,
                            region_mask, orientation)

Shared computation for one side (`north` or `south`) of the cut:
- `advection_integral`, `forcing_integral`: the two sides of the pointwise
  GCVB identity, area-integrated over `region_mask` -- an ordinary
  closure-free consistency check requiring no assumption about the wall.
- `cut_transport_integral`: the q-weighted transport `∮ (f/H) V dx` observed
  directly at the cut row `cut_index`, oriented `orientation * (west→east)`
  (see module docstring: `+1` for the region north of the cut, `-1` south).
  This also needs no wall assumption -- it is evaluated well inside the
  domain, at the cut.
- `wall_leakage`: `forcing_integral - cut_transport_integral`, the gap
  between the closure-free area-integral *prediction* (assuming the wall
  contributes zero) and the independently *observed* cut transport. This
  quantity is reported, not asserted small; see module docstring.
"""
function half_basin_island_rule(
    solution, diagnostics, gcvb, wet, dxdy, cut_index, region_mask, orientation,
)
    problem = solution.problem
    x, y = gcvb.x, gcvb.y
    cut_y = y[cut_index]

    forcing_field = gcvb.jebar .- gcvb.bottom_stress_curl .- gcvb.surface_stress_curl
    advection_integral = area_integral(
        gcvb.geostrophic_contour_advection, dxdy, region_mask,
    )
    forcing_integral = area_integral(forcing_field, dxdy, region_mask)

    cut_wet = wet[:, cut_index]
    cut_integrand = fill(NaN, length(x))
    for i in eachindex(x)
        cut_wet[i] || continue
        H = problem.depth(x[i], cut_y)
        f = problem.coriolis(x[i], cut_y)
        # Use the directly evaluated ("analytical") transport, not the
        # smoothed/projected `diagnostics.transport_V`, since
        # `gcvb.geostrophic_contour_advection` is itself built from the
        # directly evaluated current (`depth_integrated_current`) -- mixing
        # the two would spuriously reintroduce the streamfunction
        # projection error already tracked separately as
        # `diagnostics.timings.transport_projection_relative_l2`.
        cut_integrand[i] = (f / H) * diagnostics.analytical_transport_V[i, cut_index]
    end
    cut_transport_integral = orientation *
        trapezoidal_line_integral(x, cut_integrand)

    gross_scale = area_integral(
        abs.(gcvb.jebar) .+ abs.(gcvb.bottom_stress_curl) .+
            abs.(gcvb.surface_stress_curl),
        dxdy, region_mask,
    )

    return (;
        advection_integral, forcing_integral, cut_transport_integral,
        wall_leakage=forcing_integral - cut_transport_integral,
        gross_scale,
    )
end

"""
    compute_wall_cut_island_rule(solution, diagnostics, gcvb; cut_y=0.0)

Evaluate the wall-plus-cut GCVB construction described in the module
docstring at the zonal transect `y = cut_y` (snapped to the nearest
diagnostic grid row), for both the northern and southern sub-basins.
Returns, for each side:
- the ordinary closure-free pointwise-balance check (`advection_integral`
  vs. `forcing_integral`, `local_balance_relative_residual`), which needs no
  wall assumption and should be small at adequate resolution;
- the forcing-integral *prediction* for the cut transport vs. the
  *observed* `cut_transport_integral`, and their difference
  `wall_leakage` -- reported, not asserted small (see module docstring for
  why this gap can be `O(1)` at this codebase's standard inset diagnostic
  grids).
"""
function compute_wall_cut_island_rule(solution, diagnostics, gcvb; cut_y=0.0)
    x, y = gcvb.x, gcvb.y
    size(diagnostics.analytical_transport_V) == size(gcvb.wet) ||
        throw(ArgumentError("GCVB diagnostics must use the PG diagnostic grid"))

    wet = gcvb.wet .& diagnostics.wet
    any(wet) ||
        throw(ArgumentError("no wet points are common to the GCVB and PG diagnostic grids"))

    cut_index = argmin(abs.(y .- cut_y))
    any(wet[:, cut_index]) ||
        throw(ArgumentError("the cut row has no wet points"))
    actual_cut_y = y[cut_index]

    dxdy = (x[2] - x[1]) * (y[2] - y[1])
    # The cut row sits exactly on a grid line; assign it wholly to the
    # northern region so `north_mask`/`south_mask` partition every wet grid
    # point exactly once (otherwise the full-domain zero-sum property of the
    # pointwise balance would be off by that row's contribution).
    north_mask = wet .& (y' .>= actual_cut_y)
    south_mask = wet .& (y' .< actual_cut_y)
    any(north_mask) || throw(ArgumentError("no wet points north of the cut"))
    any(south_mask) || throw(ArgumentError("no wet points south of the cut"))

    north = half_basin_island_rule(
        solution, diagnostics, gcvb, wet, dxdy, cut_index, north_mask, 1.0,
    )
    south = half_basin_island_rule(
        solution, diagnostics, gcvb, wet, dxdy, cut_index, south_mask, -1.0,
    )

    local_balance_relative_residual(side) =
        abs(side.advection_integral - side.forcing_integral) /
        max(side.gross_scale, eps())

    wall_leakage_relative(side) =
        abs(side.wall_leakage) / max(side.gross_scale, eps())

    metrics = (
        cut_y=actual_cut_y,
        north_advection_integral=north.advection_integral,
        north_forcing_integral=north.forcing_integral,
        north_cut_transport_integral=north.cut_transport_integral,
        north_local_balance_relative_residual=local_balance_relative_residual(north),
        north_wall_leakage=north.wall_leakage,
        north_wall_leakage_relative=wall_leakage_relative(north),
        south_advection_integral=south.advection_integral,
        south_forcing_integral=south.forcing_integral,
        south_cut_transport_integral=south.cut_transport_integral,
        south_local_balance_relative_residual=local_balance_relative_residual(south),
        south_wall_leakage=south.wall_leakage,
        south_wall_leakage_relative=wall_leakage_relative(south),
    )

    return (; x, y, cut_index, north, south, metrics)
end

function plot_wall_cut_island_rule(result; output=nothing)
    labels = ["forcing (prediction)\n∬(jebar-curls)dA", "cut (observed)\n∮(f/H)VdX"]
    north_values = [
        result.north.forcing_integral,
        result.north.cut_transport_integral,
    ]
    south_values = [
        result.south.forcing_integral,
        result.south.cut_transport_integral,
    ]
    panel = Plots.bar(
        labels, hcat(north_values, south_values);
        label=["north (y > cut)" "south (y < cut)"],
        ylabel="area/line integral",
        title="Wall-plus-cut island rule at y = $(round(result.metrics.cut_y; digits=3))",
        color=[:steelblue :firebrick],
    )
    Plots.hline!(panel, [0.0]; color=:black, linestyle=:dot, label=false)
    output === nothing || Plots.savefig(panel, output)
    return panel
end

export compute_wall_cut_island_rule, plot_wall_cut_island_rule

end # module IslandRuleDiagnostics
