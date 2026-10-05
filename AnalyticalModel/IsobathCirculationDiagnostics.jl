"""
    IsobathCirculationDiagnostics

A third, independent, closure-free consistency test built from the
barotropic vorticity balance (BV, equation (36)),

    J(f, Ψ) = J(H, p_b) - Ek·curl(τ + τ_b),

as a counterpart to `GCVBDiagnostics.compute_gcvb_area_consistency`. That
check integrates GCVB over the area enclosed by a `q = f/H` contour (killing
the `Ψ` advection term exactly). This module instead integrates **BV** over
the area enclosed by an ordinary **isobath** `H = h0` (killing the bottom-
pressure term exactly instead), which is the "Strategy B" construction
discussed in the chat: pick the divisor field `m` to eliminate one unwanted
term, then pick which of the two remaining level sets (`f/m` or `H/m`) to
integrate around depending on which term you want to keep.

Using `⨜_R J(a, b) dA = ∮_{∂R} a db` (Green's theorem) with `a = H`: if the
whole boundary of `R` is a single isobath, `H` is literally constant along
it, so `∮_{∂R} H dp_b = h0·∮_{∂R} dp_b = 0` **exactly** (`p_b` is single-
valued -- no drag-closure assumption at all, exactly parallel to how the
`q`-contour check eliminates `Ψ` because `Ψ` is single-valued). What
survives is a line-integral identity directly relating the circulation of
the transport streamfunction `Ψ` to the net wind-plus-bottom-stress
circulation, around the *same* isobath:

    ∮_{H=h0} f dΨ = -Ek ∮_{H=h0} (τ + τ_b)·t̂ ds.

On a true `f`-plane (`f = f0` constant, as in this problem) the left-hand
side also vanishes identically (`Ψ` is single-valued, so `∮ dΨ = 0` around
any closed loop), independently of the right-hand side. The non-trivial,
checkable content is then the prediction that the net wind-plus-bottom-
stress circulation integrates to (approximately) zero around *every* closed
isobath -- a strong, assumption-free statement about the solved bottom-
pressure field (`τ_b` depends on `∇p_b`), checkable at `O(contour length)`
cost without ever differentiating or solving for `Ψ`. On a `β`-plane the
left-hand side would instead be the informative, non-trivial prediction.

`compute_isobath_circulation_consistency` reports both sides of this
identity contour-by-contour so the fit can be judged directly, and a
relative residual normalized by the gross stress-circulation scale on each
contour.
"""
module IsobathCirculationDiagnostics

import Contour
import Plots

# ---------------------------------------------------------------------------
# Small, self-contained contour-geometry and interpolation helpers.
# Deliberately duplicated (not imported) from `GCVBDiagnostics.jl` /
# `NostIsachsenDiagnostics.jl` so this module stays switchable in/out
# independently -- see the module docstring.
# ---------------------------------------------------------------------------

signed_polygon_area(xs, ys) = 0.5sum(
    xs[index] * ys[index + 1] - xs[index + 1] * ys[index]
    for index in 1:(length(xs) - 1)
)

polygon_area(xs, ys) = abs(signed_polygon_area(xs, ys))

"""
    traced_isobath_contours(x, y, H, wet; contour_levels)

Trace closed `H = const` isobaths on the rectilinear grid `x × y`. At each
level, keep only the largest-area closed loop and orient it counter-
clockwise, exactly as `GCVBDiagnostics.traced_q_contours` does for `q = f/H`
contours (see there for why: consistent line-integral sign, and a documented
shared limitation for multiply-connected levels). Returns a vector of
`(level, x, y)` named tuples sorted by level.

`H` must be sized `(length(x), length(y))` with `H[i, j] == depth(x[i],
y[j])` -- i.e. *not* transposed -- matching `Contour.contours`' expected
layout.
"""
function traced_isobath_contours(x, y, H, wet; contour_levels=20)
    H_for_contours = copy(H)
    H_for_contours[.!isfinite.(H_for_contours)] .= minimum(H[wet])
    contour_set = Contour.contours(x, y, H_for_contours, contour_levels)
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
    bilinear_interpolate(x, y, field, xq, yq)

Bilinearly interpolate the rectilinear grid `field` (sized `(length(x),
length(y))`, `field[i, j]` at `(x[i], y[j])`) at the point `(xq, yq)`.
Returns `NaN` if the query point falls outside `[x[1], x[end]] × [y[1],
y[end]]`, or if any of the four enclosing grid corners is non-finite (e.g.
masked out near the domain wall or a dry cell) -- which simply excludes that
contour segment from the line integral below, rather than silently
extrapolating across the mask.
"""
function bilinear_interpolate(x, y, field, xq, yq)
    (x[1] <= xq <= x[end] && y[1] <= yq <= y[end]) || return NaN
    i = clamp(searchsortedlast(x, xq), 1, length(x) - 1)
    j = clamp(searchsortedlast(y, yq), 1, length(y) - 1)
    tx = (xq - x[i]) / (x[i + 1] - x[i])
    ty = (yq - y[j]) / (y[j + 1] - y[j])
    c00, c10 = field[i, j], field[i + 1, j]
    c01, c11 = field[i, j + 1], field[i + 1, j + 1]
    all(isfinite, (c00, c10, c01, c11)) || return NaN
    return (1 - tx) * (1 - ty) * c00 + tx * (1 - ty) * c10 +
        (1 - tx) * ty * c01 + tx * ty * c11
end

# ---------------------------------------------------------------------------
# Isobath circulation consistency check (closure-free).
# ---------------------------------------------------------------------------

"""
    compute_isobath_circulation_consistency(solution, diagnostics;
                                            contour_levels=20)

Validate the closure-free isobath-circulation identity described in the
module docstring: for every closed isobath `H = h0`, the circulation
`∮ f dΨ` (bilinearly interpolated from `diagnostics.streamfunction`, so this
reuses the already-solved streamfunction rather than resolving it along the
contour) should match `-Ek ∮ (τ + τ_b)·t̂ ds` (wind stress evaluated
analytically from `solution.problem.wind_stress`; bottom stress bilinearly
interpolated from the already-projected `diagnostics.bottom_stress_x/y`).
Both line integrals, and hence the whole check, cost `O(contour length)` --
no new PDE solve, no area integral over the diagnostic grid.
"""
function compute_isobath_circulation_consistency(
    solution, diagnostics; contour_levels=20,
)
    problem = solution.problem
    x, y = diagnostics.x, diagnostics.y
    H = fill(NaN, size(diagnostics.wet))
    for j in eachindex(y), i in eachindex(x)
        diagnostics.wet[i, j] || continue
        H[i, j] = problem.depth(x[i], y[j])
    end

    contours = traced_isobath_contours(x, y, H, diagnostics.wet; contour_levels)
    length(contours) >= 1 ||
        throw(ArgumentError("no closed isobaths were resolved"))

    Ek = problem.parameters.Ek
    contour_H = Float64[]
    circulation_integral = Float64[]
    forcing_integral = Float64[]
    gross_scale = Float64[]
    perimeter = Float64[]

    for contour in contours
        line_x, line_y = contour.x, contour.y
        circulation = 0.0
        forcing = 0.0
        gross = 0.0
        length_total = 0.0
        complete = true
        for index in 1:(length(line_x) - 1)
            x1, y1 = line_x[index], line_y[index]
            x2, y2 = line_x[index + 1], line_y[index + 1]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1

            Ψ1 = bilinear_interpolate(x, y, diagnostics.streamfunction, x1, y1)
            Ψ2 = bilinear_interpolate(x, y, diagnostics.streamfunction, x2, y2)
            bottom_stress_x =
                bilinear_interpolate(x, y, diagnostics.bottom_stress_x, xm, ym)
            bottom_stress_y =
                bilinear_interpolate(x, y, diagnostics.bottom_stress_y, xm, ym)
            if !(
                isfinite(Ψ1) && isfinite(Ψ2) &&
                isfinite(bottom_stress_x) && isfinite(bottom_stress_y)
            )
                complete = false
                continue
            end

            f_mid = problem.coriolis(xm, ym)
            wind = problem.wind_stress(xm, ym)
            τx = real(wind) + bottom_stress_x
            τy = imag(wind) + bottom_stress_y

            circulation += f_mid * (Ψ2 - Ψ1)
            forcing += -Ek * (τx * dx + τy * dy)
            gross += Ek * hypot(τx, τy) * hypot(dx, dy)
            length_total += hypot(dx, dy)
        end
        complete || continue
        push!(contour_H, contour.level)
        push!(circulation_integral, circulation)
        push!(forcing_integral, forcing)
        push!(gross_scale, gross)
        push!(perimeter, length_total)
    end

    length(contour_H) >= 1 || throw(ArgumentError(
        "every traced isobath had at least one masked (dry or out-of-grid) vertex",
    ))

    ordering = sortperm(contour_H)
    contour_H = contour_H[ordering]
    circulation_integral = circulation_integral[ordering]
    forcing_integral = forcing_integral[ordering]
    gross_scale = gross_scale[ordering]
    perimeter = perimeter[ordering]

    residual = circulation_integral .- forcing_integral
    relative_residual = abs.(residual) ./ max.(gross_scale, eps())

    metrics = (
        contour_levels=length(contour_H),
        max_absolute_circulation_integral=maximum(abs, circulation_integral),
        max_absolute_forcing_integral=maximum(abs, forcing_integral),
        max_relative_residual=maximum(relative_residual),
        coriolis_is_spatially_uniform=isapprox(
            problem.coriolis(x[1], y[1]), problem.coriolis(x[end], y[end]);
            rtol=1e-12, atol=1e-12,
        ),
    )

    return (;
        x, y, H,
        contour_H,
        circulation_integral,
        forcing_integral,
        gross_scale,
        perimeter,
        relative_residual,
        metrics,
    )
end

function plot_isobath_circulation_consistency(result; output=nothing)
    panel = Plots.plot(
        result.contour_H, result.circulation_integral;
        label="∮ f dΨ",
        marker=:circle, markersize=3, color=:steelblue, linewidth=1.5,
        xlabel="H (isobath depth)", ylabel="line integral",
        title="Isobath circulation consistency (BV, Strategy B)",
        legend=:best,
    )
    Plots.plot!(
        panel, result.contour_H, result.forcing_integral;
        label="-Ek ∮ (τ + τ_b)·t̂ ds",
        marker=:diamond, markersize=3, color=:firebrick, linewidth=1.5,
        linestyle=:dash,
    )
    Plots.hline!(panel, [0.0]; color=:black, linestyle=:dot, label=false)
    output === nothing || Plots.savefig(panel, output)
    return panel
end

export compute_isobath_circulation_consistency,
       plot_isobath_circulation_consistency

end # module IsobathCirculationDiagnostics
