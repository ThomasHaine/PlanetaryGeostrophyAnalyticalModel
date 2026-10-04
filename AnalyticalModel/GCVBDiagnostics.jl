"""
    GCVBDiagnostics

Two independent, closure-free-where-possible tests built from the
geostrophic-contour vorticity balance (GCVB, equation (43)),

    J(f/H, Ψ) = J(1/H, χ) - Ek·curl((τ + τ_b)/H),

as a counterpart to the bottom-pressure `G(q) = dp_b/dq` theory in
`NostIsachsenDiagnostics.jl`. This module is self-contained (it duplicates
the handful of small contour-geometry helpers it needs rather than importing
them) so it can be included or omitted independently of that file.

1. `compute_gcvb_area_consistency`: GCVB's advection term `J(f/H, Ψ)` is the
   Jacobian of `q = f/H` with the *actual* transport streamfunction `Ψ`.
   Green's theorem applied to the area enclosed by any closed `q`-contour
   gives `∬ J(q, Ψ) dA = ∮_{q=q0} Ψ dq = 0` *exactly*, with no drag-closure
   assumption at all -- it only requires `Ψ` to be a well-defined,
   single-valued scalar (guaranteed by mass conservation). GCVB therefore
   implies the closure-free constraint

       ∬_{A(q0)} [jebar - bottom_stress_curl - surface_stress_curl] dA ≈ 0

   for every closed `q`-contour. This is a strictly stronger (assumption-free)
   statement than anything in `NostIsachsenDiagnostics.jl`, so it is a clean
   model/solver consistency check: a nonzero residual here reflects grid or
   solver error, not a theoretical approximation.

2. `compute_gcvb_streamfunction_theory`: tests the *separate*, approximate
   question of whether the transport streamfunction itself is organized
   along `q`-contours, i.e. `Ψ ≈ Ψ(q)`. Unlike `G(q)` for the bottom
   pressure, there is no "exact" (closure-free) version of this prediction,
   because the bottom stress `τ_b` in GCVB depends on `∇p_b`, not on `Ψ` --
   the area-integral trick above carries no information about `Ψ`'s
   structure. Instead this reproduces the original Nøst & Isachsen (2003)
   approach directly: fit global linear/quadratic bottom-drag coefficients
   against the *total* depth-integrated transport speed `|U, V|` (the
   physically relevant velocity here, as opposed to the bottom-geostrophic
   proxy used for `G(q)`), then solve for `dΨ/dq` from the same
   Green's-theorem argument used for `G(q)`.
"""
module GCVBDiagnostics

using Statistics
import Contour
import Plots

# ---------------------------------------------------------------------------
# Small, self-contained contour-geometry helpers. Deliberately duplicated
# (not imported) from `NostIsachsenDiagnostics.jl` so this module stays
# switchable in/out independently -- see the module docstring.
# ---------------------------------------------------------------------------

signed_polygon_area(xs, ys) = 0.5sum(
    xs[index] * ys[index + 1] - xs[index + 1] * ys[index]
    for index in 1:(length(xs) - 1)
)

polygon_area(xs, ys) = abs(signed_polygon_area(xs, ys))

function contour_contains(x, y, xs, ys)
    inside = false
    previous = length(xs) - 1
    for current in 1:(length(xs) - 1)
        crosses = (ys[current] > y) != (ys[previous] > y)
        if crosses && x < (xs[previous] - xs[current]) *
                (y - ys[current]) / (ys[previous] - ys[current]) + xs[current]
            inside = !inside
        end
        previous = current
    end
    return inside
end

function interpolate_contour_function(q_value, contour_q, values)
    contour_q[1] <= q_value <= contour_q[end] || return NaN
    upper = searchsortedfirst(contour_q, q_value)
    upper == 1 && return values[1]
    lower = upper - 1
    weight = (q_value - contour_q[lower]) /
        (contour_q[upper] - contour_q[lower])
    return (1 - weight) * values[lower] + weight * values[upper]
end

"""
    cumulative_trapezoid(xs, ys)

Cumulative trapezoidal integral of `ys` with respect to `xs`, zero at
`xs[1]`. `xs` must be sorted.
"""
function cumulative_trapezoid(xs, ys)
    result = zeros(length(xs))
    for index in 2:length(xs)
        result[index] = result[index - 1] +
            0.5 * (ys[index] + ys[index - 1]) * (xs[index] - xs[index - 1])
    end
    return result
end

"""
    traced_q_contours(x, y, q, wet; contour_levels)

Trace closed `q = f/H` contours on the rectilinear grid `x × y`. At each
level, keep only the largest-area closed loop (multiply-connected `q`-levels,
e.g. separate per-basin loops, are a known shared limitation -- see
`NostIsachsenDiagnostics.jl`) and orient it counter-clockwise so downstream
Green's/Stokes'-theorem line integrals have a consistent sign. Returns a
vector of `(level, x, y)` named tuples sorted by level.

`q` must be sized `(length(x), length(y))` with `q[i, j] == f(x[i], y[j])`
-- i.e. *not* transposed -- matching `Contour.contours`' expected layout.
"""
function traced_q_contours(x, y, q, wet; contour_levels=20)
    q_for_contours = copy(q)
    q_for_contours[.!isfinite.(q_for_contours)] .= maximum(q[wet])
    contour_set = Contour.contours(x, y, q_for_contours, contour_levels)
    traced = NamedTuple{(:level, :x, :y),
        Tuple{Float64,Vector{Float64},Vector{Float64}}}[]
    for contour_level in Contour.levels(contour_set)
        closed_lines = Tuple{Vector{Float64},Vector{Float64}}[]
        for line in Contour.lines(contour_level)
            line_x, line_y = Contour.coordinates(line)
            hypot(line_x[end] - line_x[1], line_y[end] - line_y[1]) <
                1e-10 || continue
            push!(closed_lines, (collect(line_x), collect(line_y)))
        end
        isempty(closed_lines) && continue
        line_x, line_y = closed_lines[
            argmax(polygon_area(xs, ys) for (xs, ys) in closed_lines)
        ]
        if signed_polygon_area(line_x, line_y) < 0
            reverse!(line_x)
            reverse!(line_y)
        end
        push!(
            traced,
            (level=Contour.level(contour_level), x=line_x, y=line_y),
        )
    end
    return sort!(traced; by=t -> t.level)
end

"""
    enclosed_area_integral(field, x, y, wet, line_x, line_y)

Sum `field[i, j] * dx * dy` over wet grid points enclosed by the closed
polygon `(line_x, line_y)`.
"""
function enclosed_area_integral(field, x, y, wet, line_x, line_y)
    dxdy = (x[2] - x[1]) * (y[2] - y[1])
    total = 0.0
    for j in eachindex(y), i in eachindex(x)
        (wet[i, j] && isfinite(field[i, j])) || continue
        contour_contains(x[i], y[j], line_x, line_y) || continue
        total += field[i, j] * dxdy
    end
    return total
end

"""
    h_normalized_contour_denominators(problem, xs, ys)

Geometric denominators for the `H`-normalized (GCVB) linear/quadratic
drag-closure line integrals, paralleling
`NostIsachsenDiagnostics.contour_denominators` (which is `f`-normalized for
the bottom-pressure theory). Here the closure velocity is the *total*
transport `(U, V) = G_Ψ(q)·(k̂ × ∇q)` -- with no extra `1/f` factor, unlike
the bottom-geostrophic velocity -- so only a single power of `H` appears:

    linear    = ∮ (|∇q| / H) ds
    quadratic = ∮ (|∇q|^2 / H) ds
"""
function h_normalized_contour_denominators(problem, xs, ys)
    linear = 0.0
    quadratic = 0.0
    for index in 1:(length(xs) - 1)
        x = (xs[index] + xs[index + 1]) / 2
        y = (ys[index] + ys[index + 1]) / 2
        segment_length = hypot(
            xs[index + 1] - xs[index], ys[index + 1] - ys[index],
        )
        H = problem.depth(x, y)
        f = problem.coriolis(x, y)
        q_x = (problem.coriolis_x(x, y) * H - f * problem.depth_x(x, y)) / H^2
        q_y = (problem.coriolis_y(x, y) * H - f * problem.depth_y(x, y)) / H^2
        q_gradient_magnitude = hypot(q_x, q_y)
        linear += q_gradient_magnitude / H * segment_length
        quadratic += q_gradient_magnitude^2 / H * segment_length
    end
    return (; linear, quadratic)
end

# ---------------------------------------------------------------------------
# 1. GCVB area-integral consistency check (closure-free).
# ---------------------------------------------------------------------------

"""
    compute_gcvb_area_consistency(solution, gcvb; contour_levels=20)

Validate the closure-free GCVB area-integral identity described in the
module docstring: for every closed `q = f/H` contour, both the area integral
of the advection term `J(q, Ψ)` and the area integral of
`jebar - bottom_stress_curl - surface_stress_curl` should vanish (the first
exactly, by Green's theorem and single-valuedness of `Ψ`; the second
following from it plus the pointwise GCVB residual, which is independently
already verified to be small). Returns per-contour integrals together with a
gross per-contour forcing scale (area integral of `|jebar| + |bottom_stress_curl|
+ |surface_stress_curl|`) so residuals can be reported relative to the size
of the terms actually being balanced.
"""
function compute_gcvb_area_consistency(solution, gcvb; contour_levels=20)
    problem = solution.problem
    x, y = gcvb.x, gcvb.y
    q = fill(NaN, size(gcvb.wet))
    for j in eachindex(y), i in eachindex(x)
        gcvb.wet[i, j] || continue
        q[i, j] = problem.coriolis(x[i], y[j]) / problem.depth(x[i], y[j])
    end

    contours = traced_q_contours(x, y, q, gcvb.wet; contour_levels)
    length(contours) >= 1 ||
        throw(ArgumentError("no closed q contours were resolved"))

    forcing_field = gcvb.jebar .- gcvb.bottom_stress_curl .- gcvb.surface_stress_curl
    gross_field = abs.(gcvb.jebar) .+ abs.(gcvb.bottom_stress_curl) .+
        abs.(gcvb.surface_stress_curl)

    contour_q = Float64[]
    advection_integral = Float64[]
    forcing_integral = Float64[]
    gross_scale = Float64[]
    for contour in contours
        push!(contour_q, contour.level)
        push!(
            advection_integral,
            enclosed_area_integral(
                gcvb.geostrophic_contour_advection, x, y, gcvb.wet,
                contour.x, contour.y,
            ),
        )
        push!(
            forcing_integral,
            enclosed_area_integral(
                forcing_field, x, y, gcvb.wet, contour.x, contour.y,
            ),
        )
        push!(
            gross_scale,
            enclosed_area_integral(
                gross_field, x, y, gcvb.wet, contour.x, contour.y,
            ),
        )
    end

    relative_advection_residual = abs.(advection_integral) ./
        max.(gross_scale, eps())
    relative_forcing_residual = abs.(forcing_integral) ./
        max.(gross_scale, eps())

    metrics = (
        contour_levels=length(contour_q),
        max_absolute_advection_integral=maximum(abs, advection_integral),
        max_absolute_forcing_integral=maximum(abs, forcing_integral),
        max_relative_advection_residual=maximum(relative_advection_residual),
        max_relative_forcing_residual=maximum(relative_forcing_residual),
    )

    return (;
        x, y, q,
        contour_q,
        advection_integral,
        forcing_integral,
        gross_scale,
        relative_advection_residual,
        relative_forcing_residual,
        metrics,
    )
end

function plot_gcvb_area_consistency(result; output=nothing)
    panel = Plots.plot(
        result.contour_q, result.advection_integral;
        label="∬ J(q, Ψ) dA  (exactly 0 by Green's theorem)",
        marker=:circle, markersize=3, color=:steelblue, linewidth=1.5,
        xlabel="q = f/H", ylabel="area integral",
        title="GCVB area-integral consistency",
        legend=:best,
    )
    Plots.plot!(
        panel, result.contour_q, result.forcing_integral;
        label="∬ (jebar − bottom_stress_curl − surface_stress_curl) dA",
        marker=:diamond, markersize=3, color=:firebrick, linewidth=1.5,
        linestyle=:dash,
    )
    Plots.hline!(panel, [0.0]; color=:black, linestyle=:dot, label=false)
    output === nothing || Plots.savefig(panel, output)
    return panel
end

# ---------------------------------------------------------------------------
# 2. Ψ(q) scatter and Nøst-Isachsen/GCVB linear & quadratic predictions.
# ---------------------------------------------------------------------------

"""
    compute_gcvb_streamfunction_theory(solution, diagnostics, gcvb;
                                       contour_levels=20)

Test whether the transport streamfunction is organized along closed
`q = f/H` contours (`Ψ ≈ Ψ(q)`), and build the Nøst & Isachsen (2003)
linear and quadratic drag-closure predictions for `dΨ/dq` (see the module
docstring for why no "exact", closure-free prediction of `Ψ(q)` exists,
unlike `G(q)` for the bottom pressure). The drag coefficients are fit
against the *total* depth-integrated transport speed `|U, V|`, not the
bottom-geostrophic proxy used elsewhere, since `Ψ` generates the total
transport directly.
"""
function compute_gcvb_streamfunction_theory(
    solution, diagnostics, gcvb; contour_levels=20,
)
    problem = solution.problem
    x, y = gcvb.x, gcvb.y
    size(diagnostics.streamfunction) == size(gcvb.wet) ||
        throw(ArgumentError("GCVB diagnostics must use the PG diagnostic grid"))

    wet = gcvb.wet .& diagnostics.wet
    any(wet) ||
        throw(ArgumentError("no wet points are common to the GCVB and PG diagnostic grids"))

    q = fill(NaN, size(wet))
    for j in eachindex(y), i in eachindex(x)
        wet[i, j] || continue
        q[i, j] = problem.coriolis(x[i], y[j]) / problem.depth(x[i], y[j])
    end

    total_speed = hypot.(diagnostics.transport_U, diagnostics.transport_V)
    effective_bottom_stress_magnitude = problem.parameters.Ek .* hypot.(
        diagnostics.bottom_stress_x, diagnostics.bottom_stress_y,
    )

    # Global linear/quadratic drag coefficients fit against the *total*
    # transport speed, in direct analogy with (but distinct from) the
    # bottom-geostrophic-speed fits in `NostIsachsenDiagnostics.jl`.
    linear_drag_transport = sum(
        (effective_bottom_stress_magnitude .* total_speed)[wet],
    ) / sum((total_speed .^ 2)[wet])
    quadratic_drag_transport = sum(
        (effective_bottom_stress_magnitude .* total_speed .^ 2)[wet],
    ) / sum((total_speed .^ 4)[wet])

    gcvb_forcing_field = gcvb.jebar .- gcvb.surface_stress_curl

    contours = traced_q_contours(x, y, q, wet; contour_levels)
    length(contours) >= 2 ||
        throw(ArgumentError("fewer than two closed q contours were resolved"))

    contour_q = Float64[]
    linear_G_psi = Float64[]
    quadratic_G_psi = Float64[]
    for contour in contours
        gcvb_forcing = enclosed_area_integral(
            gcvb_forcing_field, x, y, wet, contour.x, contour.y,
        )
        denominators = h_normalized_contour_denominators(
            problem, contour.x, contour.y,
        )
        push!(contour_q, contour.level)
        push!(
            linear_G_psi,
            -gcvb_forcing / (linear_drag_transport * denominators.linear),
        )
        push!(
            quadratic_G_psi,
            -sign(gcvb_forcing) * sqrt(
                abs(gcvb_forcing) /
                (quadratic_drag_transport * denominators.quadratic),
            ),
        )
    end
    ordering = sortperm(contour_q)
    contour_q = contour_q[ordering]
    linear_G_psi = linear_G_psi[ordering]
    quadratic_G_psi = quadratic_G_psi[ordering]

    # Predicted Ψ(q), up to an additive constant (Ψ's gauge is arbitrary)
    # fixed by least-squares matching the mean of the actual scatter.
    linear_psi_pred = cumulative_trapezoid(contour_q, linear_G_psi)
    quadratic_psi_pred = cumulative_trapezoid(contour_q, quadratic_G_psi)

    scatter_q = q[wet]
    scatter_psi = diagnostics.streamfunction[wet]
    linear_interp_at_scatter = [
        interpolate_contour_function(qq, contour_q, linear_psi_pred)
        for qq in scatter_q
    ]
    quadratic_interp_at_scatter = [
        interpolate_contour_function(qq, contour_q, quadratic_psi_pred)
        for qq in scatter_q
    ]
    linear_valid = isfinite.(linear_interp_at_scatter)
    quadratic_valid = isfinite.(quadratic_interp_at_scatter)
    linear_psi_pred .+= mean(
        scatter_psi[linear_valid] .- linear_interp_at_scatter[linear_valid],
    )
    quadratic_psi_pred .+= mean(
        scatter_psi[quadratic_valid] .-
            quadratic_interp_at_scatter[quadratic_valid],
    )

    linear_psi_at_points = fill(NaN, size(wet))
    quadratic_psi_at_points = fill(NaN, size(wet))
    for index in findall(wet)
        linear_psi_at_points[index] =
            interpolate_contour_function(q[index], contour_q, linear_psi_pred)
        quadratic_psi_at_points[index] = interpolate_contour_function(
            q[index], contour_q, quadratic_psi_pred,
        )
    end
    theory_wet = wet .& isfinite.(linear_psi_at_points) .&
        isfinite.(quadratic_psi_at_points)
    any(theory_wet) ||
        throw(ArgumentError("no points have both a resolved q contour prediction and a sampled Ψ"))

    l2(values; mask=theory_wet) = sqrt(sum(abs2, values[mask]))
    psi_scale = l2(diagnostics.streamfunction; mask=theory_wet)
    metrics = (
        comparison_points=count(theory_wet),
        linear_drag_transport=linear_drag_transport,
        quadratic_drag_transport=quadratic_drag_transport,
        linear_relative_l2_streamfunction_error=l2(
            diagnostics.streamfunction .- linear_psi_at_points,
        ) / psi_scale,
        quadratic_relative_l2_streamfunction_error=l2(
            diagnostics.streamfunction .- quadratic_psi_at_points,
        ) / psi_scale,
        linear_streamfunction_correlation=cor(
            diagnostics.streamfunction[theory_wet],
            linear_psi_at_points[theory_wet],
        ),
        quadratic_streamfunction_correlation=cor(
            diagnostics.streamfunction[theory_wet],
            quadratic_psi_at_points[theory_wet],
        ),
    )

    return (;
        x, y, q, wet, theory_wet,
        scatter_q, scatter_psi,
        contour_q, linear_G_psi, quadratic_G_psi,
        linear_psi_pred, quadratic_psi_pred,
        linear_psi_at_points, quadratic_psi_at_points,
        metrics,
    )
end

function plot_gcvb_streamfunction_theory(result; output=nothing)
    panel = Plots.scatter(
        result.scatter_q, result.scatter_psi;
        xlabel="q = f/H", ylabel="Ψ (transport streamfunction)",
        label="PG Ψ (scatter)", markersize=1.5, markerstrokewidth=0,
        markeralpha=0.35, color=:gray40, legend=:best,
        title="Is Ψ organized along q contours?",
    )
    Plots.plot!(
        panel, result.contour_q, result.linear_psi_pred;
        label="Nøst-Isachsen / GCVB linear closure",
        color=:steelblue, linewidth=2,
    )
    Plots.plot!(
        panel, result.contour_q, result.quadratic_psi_pred;
        label="Nøst-Isachsen / GCVB quadratic closure",
        color=:firebrick, linewidth=2, linestyle=:dash,
    )
    output === nothing || Plots.savefig(panel, output)
    return panel
end

export compute_gcvb_area_consistency,
       plot_gcvb_area_consistency,
       compute_gcvb_streamfunction_theory,
       plot_gcvb_streamfunction_theory

end # module GCVBDiagnostics
