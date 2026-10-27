module NostIsachsenDiagnostics

using Statistics
import Contour
import Plots
import PlanetaryGeostrophy

export compute_nost_isachsen_diagnostics,
       plot_nost_isachsen_diagnostics

function centered_gradient(values, wet, xs, ys)
    value_x = fill(NaN, size(values))
    value_y = fill(NaN, size(values))
    stencil_wet = falses(size(values))
    for j in 2:(length(ys) - 1), i in 2:(length(xs) - 1)
        all(wet[ii, jj] for (ii, jj) in (
            (i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
        )) || continue
        value_x[i, j] =
            (values[i + 1, j] - values[i - 1, j]) / (xs[i + 1] - xs[i - 1])
        value_y[i, j] =
            (values[i, j + 1] - values[i, j - 1]) / (ys[j + 1] - ys[j - 1])
        stencil_wet[i, j] = true
    end
    return (; x=value_x, y=value_y, wet=stencil_wet)
end

include("ContourGeometry.jl")

function contour_denominators(problem, xs, ys)
    linear = 0.0
    quadratic = 0.0
    for index in 1:(length(xs) - 1)
        x = (xs[index] + xs[index + 1]) / 2
        y = (ys[index] + ys[index + 1]) / 2
        segment_length = hypot(
            xs[index + 1] - xs[index],
            ys[index + 1] - ys[index],
        )
        H = problem.depth(x, y)
        f = problem.coriolis(x, y)
        q_x = (problem.coriolis_x(x, y) * H - f * problem.depth_x(x, y)) / H^2
        q_y = (problem.coriolis_y(x, y) * H - f * problem.depth_y(x, y)) / H^2
        q_gradient_magnitude = hypot(q_x, q_y)
        linear += q_gradient_magnitude / f^2 * segment_length
        quadratic += q_gradient_magnitude^2 / f^3 * segment_length
    end
    return (; linear, quadratic)
end



"""
    exact_drag_contour_integrals(problem, p, xs, ys, grid_x, grid_y, T_star_over_f)

Exact (closure-free) contour-integral theory for `G(q) = dp_b/dq`, derived
from the model's actual Ekman bottom-stress closure `τ_b = S·∇p_b + T*`
(with `T* = S·∇χ + T` from equations (52)--(54)) rather than a fitted linear
or quadratic drag law. Stokes' theorem applied to the TDVB identity (37) on a
closed `q = f/H` contour gives
`Ek·[G(q)·D(q) + F_T(q)] = -enclosed_forcing(q)`, where

    D(q) = Re[ ∮ conj((S/f)·∇q) dz ]
    F_T(q) = Re[ ∮ conj(T*/f) dz ]

This returns `(D, FT)` for one closed contour `(xs, ys)`; `S` and `∇q` are
evaluated analytically at each segment midpoint, while `T*/f` is bilinearly
interpolated from the precomputed grid field `T_star_over_f`.
"""
function exact_drag_contour_integrals(
    problem, p, xs, ys, grid_x, grid_y, T_star_over_f,
)
    D = 0.0
    FT = 0.0
    for index in 1:(length(xs) - 1)
        x = (xs[index] + xs[index + 1]) / 2
        y = (ys[index] + ys[index + 1]) / 2
        dz = Complex(xs[index + 1] - xs[index], ys[index + 1] - ys[index])
        H = problem.depth(x, y)
        f = problem.coriolis(x, y)
        q_x = (problem.coriolis_x(x, y) * H - f * problem.depth_x(x, y)) / H^2
        q_y = (problem.coriolis_y(x, y) * H - f * problem.depth_y(x, y)) / H^2
        S = PlanetaryGeostrophy.bottom_stress_coefficient(p, H, f)
        D += real(conj(S / f * Complex(q_x, q_y)) * dz)
        T_star_f = bilinear_interpolate(grid_x, grid_y, T_star_over_f, x, y)
        isfinite(T_star_f) && (FT += real(conj(T_star_f) * dz))
    end
    return (; D, FT)
end

"""
    compute_nost_isachsen_diagnostics(solution, diagnostics, dim, tdvb;
                                      contour_levels=20,
                                      q_gradient_floor=1e-10)

Compare the PG bottom-geostrophic flow with the closed-`q` contour
predictions of Nøst & Isachsen (2003) using globally fitted linear and
quadratic drag magnitudes, together with an exact, closure-free `G(q)`
theory derived from the model's actual Ekman bottom-stress closure
`τ_b = S·∇p_b + T*` (see `exact_drag_contour_integrals`). The returned
`metrics.stress_velocity_alignment` reports the signed alignment that is
intentionally excluded from the fitted positive drag laws.
"""
function compute_nost_isachsen_diagnostics(
    solution,
    diagnostics,
    dim,
    tdvb;
    contour_levels=20,
    q_gradient_floor=1e-10,
)
    x, y = diagnostics.x, diagnostics.y
    size(dim.bottom_pressure) == (length(x), length(y)) ||
        throw(ArgumentError("DIM diagnostics must use the PG diagnostic grid"))
    size(tdvb.residual) == (length(x), length(y)) ||
        throw(ArgumentError("TDVB diagnostics must use the PG diagnostic grid"))

    problem = solution.problem
    q = fill(NaN, size(dim.bottom_pressure))
    f = fill(NaN, size(dim.bottom_pressure))
    for j in eachindex(y), i in eachindex(x)
        diagnostics.wet[i, j] || continue
        xx, yy = x[i], y[j]
        f[i, j] = problem.coriolis(xx, yy)
        q[i, j] = f[i, j] / problem.depth(xx, yy)
    end

    Ek = problem.parameters.Ek
    p_internal = PlanetaryGeostrophy.internal_parameters(problem.parameters)

    pb_gradient = centered_gradient(dim.bottom_pressure, diagnostics.wet, x, y)
    q_gradient = centered_gradient(q, diagnostics.wet, x, y)
    q_gradient_squared = q_gradient.x .^ 2 .+ q_gradient.y .^ 2
    resolved_q_gradient = q_gradient_squared .>
        q_gradient_floor * maximum(q_gradient_squared[q_gradient.wet])
    comparison_wet =
        tdvb.wet .& pb_gradient.wet .& q_gradient.wet .& resolved_q_gradient
    any(comparison_wet) ||
        throw(ArgumentError("no wet points have a resolved q gradient"))

    pg_bottom_u = -pb_gradient.y ./ f
    pg_bottom_v = pb_gradient.x ./ f
    local_dpb_dq = (
        pb_gradient.x .* q_gradient.x .+
        pb_gradient.y .* q_gradient.y
    ) ./ q_gradient_squared
    pg_speed = hypot.(pg_bottom_u, pg_bottom_v)

    effective_bottom_stress_x =
        -problem.parameters.Ek .* diagnostics.bottom_stress_x
    effective_bottom_stress_y =
        -problem.parameters.Ek .* diagnostics.bottom_stress_y
    effective_bottom_stress_magnitude = hypot.(
        effective_bottom_stress_x,
        effective_bottom_stress_y,
    )
    stress_velocity_dot = (
        effective_bottom_stress_x .* pg_bottom_u .+
        effective_bottom_stress_y .* pg_bottom_v
    )
    linear_drag = sum(
        (effective_bottom_stress_magnitude .* pg_speed)[comparison_wet],
    ) / sum(pg_speed[comparison_wet] .^ 2)
    quadratic_drag = sum(
        (effective_bottom_stress_magnitude .* pg_speed .^ 2)[comparison_wet],
    ) / sum(pg_speed[comparison_wet] .^ 4)
    drag_alignment = sum(stress_velocity_dot[comparison_wet]) / sum(
        (effective_bottom_stress_magnitude .* pg_speed)[comparison_wet],
    )

    # Exact (closure-free) bottom-stress decomposition τ_b = S·∇p_b + T*,
    # with T* = τ_b - S·∇p_b (equations (52)--(54)). `S` is evaluated
    # analytically; `∇p_b` reuses the grid-based `pb_gradient` already
    # computed above, so no additional expensive quadratures are needed.
    S_field = fill(NaN + 0.0im, size(q))
    for j in eachindex(y), i in eachindex(x)
        diagnostics.wet[i, j] || continue
        H = problem.depth(x[i], y[j])
        S_field[i, j] =
            PlanetaryGeostrophy.bottom_stress_coefficient(p_internal, H, f[i, j])
    end
    tau_b = complex.(diagnostics.bottom_stress_x, diagnostics.bottom_stress_y)
    T_star = tau_b .- S_field .* complex.(pb_gradient.x, pb_gradient.y)
    T_star_over_f = T_star ./ f

    ni_forcing = tdvb.surface_stress_curl .+ tdvb.baroclinic_torque
    dxdy = (x[2] - x[1]) * (y[2] - y[1])
    contour_q = Float64[]
    linear_G = Float64[]
    quadratic_G = Float64[]
    exact_G = Float64[]
    # Closed q contours, oriented counter-clockwise so that the line
    # integrals D(q) and F_T(q) below match the orientation of the area
    # integral `enclosed_forcing`.
    # The filled field is also returned for plotting the contours.
    q_for_contours = copy(q)
    q_for_contours[.!isfinite.(q_for_contours)] .= maximum(q[diagnostics.wet])
    for contour in traced_closed_contours(
        x, y, q_for_contours, diagnostics.wet; contour_levels,
        fill_value=maximum(q[diagnostics.wet]),
    )
        line_x, line_y = contour.x, contour.y
        enclosed_forcing = 0.0
        for j in eachindex(y), i in eachindex(x)
            tdvb.wet[i, j] || continue
            contour_contains(x[i], y[j], line_x, line_y) || continue
            enclosed_forcing += ni_forcing[i, j] * dxdy
        end
        denominators = contour_denominators(problem, line_x, line_y)
        exact = exact_drag_contour_integrals(
            problem, p_internal, line_x, line_y, x, y, T_star_over_f,
        )
        push!(contour_q, contour.level)
        push!(linear_G, enclosed_forcing / (linear_drag * denominators.linear))
        push!(quadratic_G, sign(enclosed_forcing) * sqrt(
            abs(enclosed_forcing) / (quadratic_drag * denominators.quadratic),
        ))
        push!(exact_G, (-enclosed_forcing / Ek - exact.FT) / exact.D)
    end
    length(contour_q) >= 2 ||
        throw(ArgumentError("fewer than two closed q contours were resolved"))
    ordering = sortperm(contour_q)
    contour_q = contour_q[ordering]
    linear_G = linear_G[ordering]
    quadratic_G = quadratic_G[ordering]
    exact_G = exact_G[ordering]

    linear_bottom_u = fill(NaN, size(q))
    linear_bottom_v = fill(NaN, size(q))
    quadratic_bottom_u = fill(NaN, size(q))
    quadratic_bottom_v = fill(NaN, size(q))
    exact_bottom_u = fill(NaN, size(q))
    exact_bottom_v = fill(NaN, size(q))
    theory_wet = falses(size(q))
    for index in findall(comparison_wet)
        linear_gradient =
            interpolate_contour_function(q[index], contour_q, linear_G)
        quadratic_gradient =
            interpolate_contour_function(q[index], contour_q, quadratic_G)
        exact_gradient =
            interpolate_contour_function(q[index], contour_q, exact_G)
        isfinite(linear_gradient) && isfinite(quadratic_gradient) &&
            isfinite(exact_gradient) || continue
        linear_bottom_u[index] =
            -linear_gradient * q_gradient.y[index] / f[index]
        linear_bottom_v[index] =
            linear_gradient * q_gradient.x[index] / f[index]
        quadratic_bottom_u[index] =
            -quadratic_gradient * q_gradient.y[index] / f[index]
        quadratic_bottom_v[index] =
            quadratic_gradient * q_gradient.x[index] / f[index]
        exact_bottom_u[index] =
            -exact_gradient * q_gradient.y[index] / f[index]
        exact_bottom_v[index] =
            exact_gradient * q_gradient.x[index] / f[index]
        theory_wet[index] = true
    end

    linear_speed = hypot.(linear_bottom_u, linear_bottom_v)
    quadratic_speed = hypot.(quadratic_bottom_u, quadratic_bottom_v)
    exact_speed = hypot.(exact_bottom_u, exact_bottom_v)
    linear_velocity_difference = hypot.(
        linear_bottom_u .- pg_bottom_u,
        linear_bottom_v .- pg_bottom_v,
    )
    quadratic_velocity_difference = hypot.(
        quadratic_bottom_u .- pg_bottom_u,
        quadratic_bottom_v .- pg_bottom_v,
    )
    exact_velocity_difference = hypot.(
        exact_bottom_u .- pg_bottom_u,
        exact_bottom_v .- pg_bottom_v,
    )
    normal_pg_velocity = (
        pg_bottom_u .* q_gradient.x .+
        pg_bottom_v .* q_gradient.y
    ) ./ sqrt.(q_gradient_squared)
    l2(values; wet=comparison_wet) = sqrt(sum(abs2, values[wet]))
    tdvb_nonpressure_sum = (
        tdvb.baroclinic_torque .+
        tdvb.bottom_stress_curl .+
        tdvb.surface_stress_curl
    )
    tdvb_scale = maximum(l2(field) for field in (
        tdvb.bottom_pressure_torque,
        tdvb.baroclinic_torque,
        tdvb.bottom_stress_curl,
        tdvb.surface_stress_curl,
    ))
    metrics = (
        comparison_points=count(theory_wet),
        calibrated_linear_drag=linear_drag,
        calibrated_quadratic_drag=quadratic_drag,
        stress_velocity_alignment=drag_alignment,
        linear_relative_l2_velocity_error=
            l2(linear_velocity_difference; wet=theory_wet) /
            l2(pg_speed; wet=theory_wet),
        quadratic_relative_l2_velocity_error=
            l2(quadratic_velocity_difference; wet=theory_wet) /
            l2(pg_speed; wet=theory_wet),
        exact_relative_l2_velocity_error=
            l2(exact_velocity_difference; wet=theory_wet) /
            l2(pg_speed; wet=theory_wet),
        linear_speed_correlation=
            cor(pg_speed[theory_wet], linear_speed[theory_wet]),
        quadratic_speed_correlation=
            cor(pg_speed[theory_wet], quadratic_speed[theory_wet]),
        exact_speed_correlation=
            cor(pg_speed[theory_wet], exact_speed[theory_wet]),
        relative_l2_cross_q_velocity=l2(normal_pg_velocity) / l2(pg_speed),
        pg_tdvb_relative_residual=l2(tdvb.residual) / tdvb_scale,
        ni_zero_pressure_torque_relative_residual=
            l2(tdvb_nonpressure_sum) / tdvb_scale,
    )

    return (;
        x,
        y,
        q,
        q_for_contours,
        q_gradient,
        comparison_wet,
        theory_wet,
        pg_bottom_u,
        pg_bottom_v,
        pg_speed,
        local_dpb_dq,
        contour_q,
        linear_G,
        quadratic_G,
        exact_G,
        linear_bottom_u,
        linear_bottom_v,
        quadratic_bottom_u,
        quadratic_bottom_v,
        exact_bottom_u,
        exact_bottom_v,
        linear_speed,
        quadratic_speed,
        exact_speed,
        linear_velocity_difference,
        quadratic_velocity_difference,
        exact_velocity_difference,
        tdvb_nonpressure_sum,
        tdvb,
        metrics,
    )
end

function velocity_panel(result, wet, u, v, title, velocity_scale)
    panel = PlanetaryGeostrophy.vector_diagnostic_plot(
        (; x=result.x, y=result.y, wet),
        u,
        v,
        title;
        stride=3,
        arrow_scale=0.70,
        arrowhead_size=0.08,
        contours=result.q_for_contours,
        magnitude_scale=velocity_scale,
        color=:viridis,
    )
    Plots.plot!(
        panel;
        xlabel="", clims=(0.0, velocity_scale), colorbar=true,
    )
    return panel
end

function error_panel(result, error, title, velocity_error_scale)
    panel = Plots.heatmap(
        result.x,
        result.y,
        error';
        xlabel="x",
        ylabel="y",
        title,
        color=:magma,
        clims=(0.0, velocity_error_scale),
        colorbar_title="vector error",
        aspect_ratio=:equal,
    )
    Plots.contour!(
        panel,
        result.x,
        result.y,
        result.q_for_contours';
        levels=10,
        color=:white,
        linewidth=0.6,
        alpha=0.65,
        label="",
    )
    return panel
end

"""
    plot_nost_isachsen_diagnostics(result; output_prefix=nothing)

Plot bottom-geostrophic velocity predictions and the TDVB consequence of
`p_b = p_b(f/H)`. If supplied, `output_prefix` is used for both PNG paths.
"""
function plot_nost_isachsen_diagnostics(result; output_prefix=nothing)
    velocity_scale = maximum((
        maximum(result.pg_speed[result.theory_wet]),
        maximum(result.linear_speed[result.theory_wet]),
        maximum(result.quadratic_speed[result.theory_wet]),
        maximum(result.exact_speed[result.theory_wet]),
    ))
    velocity_error_scale = max(
        maximum(result.linear_velocity_difference[result.theory_wet]),
        maximum(result.quadratic_velocity_difference[result.theory_wet]),
        maximum(result.exact_velocity_difference[result.theory_wet]),
    )

    pressure_gradient_panel = Plots.scatter(
        result.q[result.comparison_wet],
        result.local_dpb_dq[result.comparison_wet];
        xlabel="q = f/H",
        ylabel="G(q) = dp_b/dq",
        title="closed-contour drag predictions",
        markersize=2,
        markerstrokewidth=0,
        alpha=0.20,
        label="local PG",
    )
    Plots.plot!(
        pressure_gradient_panel,
        result.contour_q,
        result.linear_G;
        linewidth=3,
        marker=:circle,
        label="linear drag",
    )
    Plots.plot!(
        pressure_gradient_panel,
        result.contour_q,
        result.quadratic_G;
        linewidth=3,
        marker=:diamond,
        label="quadratic drag",
    )
    Plots.plot!(
        pressure_gradient_panel,
        result.contour_q,
        result.exact_G;
        linewidth=3,
        marker=:star5,
        label="exact theory",
    )

    blank_panel = Plots.plot(; framestyle=:none, legend=false)

    velocity_figure = Plots.plot(
        velocity_panel(
            result,
            result.comparison_wet,
            result.pg_bottom_u,
            result.pg_bottom_v,
            "PG bottom-geostrophic speed",
            velocity_scale,
        ),
        velocity_panel(
            result,
            result.theory_wet,
            result.linear_bottom_u,
            result.linear_bottom_v,
            "Nøst--Isachsen linear drag",
            velocity_scale,
        ),
        velocity_panel(
            result,
            result.theory_wet,
            result.quadratic_bottom_u,
            result.quadratic_bottom_v,
            "Nøst--Isachsen quadratic drag",
            velocity_scale,
        ),
        velocity_panel(
            result,
            result.theory_wet,
            result.exact_bottom_u,
            result.exact_bottom_v,
            "Exact S, T* theory",
            velocity_scale,
        ),
        error_panel(
            result,
            result.linear_velocity_difference,
            "linear drag: vector error",
            velocity_error_scale,
        ),
        error_panel(
            result,
            result.quadratic_velocity_difference,
            "quadratic drag: vector error",
            velocity_error_scale,
        ),
        error_panel(
            result,
            result.exact_velocity_difference,
            "exact theory: vector error",
            velocity_error_scale,
        ),
        pressure_gradient_panel,
        blank_panel;
        layout=(3, 3),
        size=(1650, 1350),
        plot_title="PG versus Nøst & Isachsen (2003)",
    )

    tdvb_plot_scale = maximum(
        abs,
        result.tdvb.bottom_pressure_torque[result.comparison_wet],
    )
    tdvb_map(field, title) = Plots.heatmap(
        result.x,
        result.y,
        field';
        xlabel="x",
        ylabel="y",
        title,
        color=:balance,
        clims=(-tdvb_plot_scale, tdvb_plot_scale),
        aspect_ratio=:equal,
    )
    ni_pressure_torque = fill(NaN, size(result.q))
    ni_pressure_torque[result.comparison_wet] .= 0.0
    tdvb_figure = Plots.plot(
        tdvb_map(result.tdvb.bottom_pressure_torque, "PG J(p_b, H/f)"),
        tdvb_map(ni_pressure_torque, "Nøst--Isachsen prediction: 0"),
        tdvb_map(
            result.tdvb_nonpressure_sum,
            "TDVB remainder if torque = 0",
        );
        layout=(1, 3),
        size=(1350, 390),
        plot_title="TDVB consequence of p_b = p_b(f/H)",
    )

    if !isnothing(output_prefix)
        Plots.savefig(velocity_figure, output_prefix * "_velocity_comparison.png")
        Plots.savefig(tdvb_figure, output_prefix * "_tdvb_comparison.png")
    end
    return (; velocity=velocity_figure, tdvb=tdvb_figure)
end

end
