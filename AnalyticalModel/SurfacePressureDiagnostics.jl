module SurfacePressureDiagnostics

import PlanetaryGeostrophy

export compute_pressure_equation_diagnostics,
       compute_adjoint_sensitivity_diagnostics,
       plot_pressure_equation_diagnostics,
       plot_adjoint_sensitivity_diagnostics

finite_extrema(values) = extrema(filter(isfinite, vec(values)))

function sampled_target_cost(x, y, sampled_transport, target, target_radius, cost)
    size(sampled_transport.U) == (length(x), length(y)) ||
        throw(ArgumentError("transport U must match the sensitivity grid"))
    size(sampled_transport.V) == (length(x), length(y)) ||
        throw(ArgumentError("transport V must match the sensitivity grid"))
    target_x, target_y = target
    weighted_cost = 0.0
    weight_sum = 0.0
    for j in eachindex(y), i in eachindex(x)
        U, V = sampled_transport.U[i, j], sampled_transport.V[i, j]
        isfinite(U) && isfinite(V) || continue
        weight = exp(-(
            (x[i] - target_x)^2 + (y[j] - target_y)^2
        ) / (2target_radius^2))
        weighted_cost += weight * cost(U, V)
        weight_sum += weight
    end
    weight_sum > 0 ||
        throw(ArgumentError("target averaging kernel has no sampled transport"))
    return weighted_cost / weight_sum
end

"""
    compute_pressure_equation_diagnostics(solution, diagnostics;
                                          rtol=1e-10, atol=1e-12)

Compute the surface-pressure advection-diffusion diagnostics and verify that
the reported inviscid and frictional advection velocities sum to the total.
"""
function compute_pressure_equation_diagnostics(
    solution,
    diagnostics;
    rtol=1e-10,
    atol=1e-12,
)
    result = PlanetaryGeostrophy.compute_surface_pressure_equation_diagnostics(
        solution,
        diagnostics,
    )
    wet = result.wet
    any(wet) || throw(ArgumentError("surface-pressure diagnostic grid has no wet points"))

    velocity_sum_x =
        result.inviscid_advection_velocity_x .+
        result.frictional_advection_velocity_x
    velocity_sum_y =
        result.inviscid_advection_velocity_y .+
        result.frictional_advection_velocity_y
    velocity_sum_error = maximum(hypot.(
        velocity_sum_x[wet] .- result.advection_velocity_x[wet],
        velocity_sum_y[wet] .- result.advection_velocity_y[wet],
    ))
    isapprox(
        velocity_sum_x[wet],
        result.advection_velocity_x[wet];
        rtol,
        atol,
    ) || error("surface-pressure x-advection decomposition does not close")
    isapprox(
        velocity_sum_y[wet],
        result.advection_velocity_y[wet];
        rtol,
        atol,
    ) || error("surface-pressure y-advection decomposition does not close")

    metrics = (;
        advection_velocity_sum_error=velocity_sum_error,
        relative_l2_residual=result.timings.relative_l2_residual,
    )
    return (; result, metrics)
end

"""
    compute_adjoint_sensitivity_diagnostics(
        solution, x, y, z;
        target, cost, cost_gradient, sampled_transport=nothing,
        cost_value=nothing, target_weight=nothing, target_metadata=nothing,
        cost_label="J", target_radius,
    )

Compute spatial forcing-sensitivity kernels and summarize the finite range of
each kernel for compact notebook reporting. For the default Gaussian target,
`sampled_transport` contains sampled base-state `U` and `V` arrays used to
report the target cost without another mesh-wide analytical transport
integration. Alternatively, `cost_value` can supply a separately evaluated
section or regional transport.
"""
function compute_adjoint_sensitivity_diagnostics(
    solution,
    x,
    y,
    z;
    target,
    cost,
    cost_gradient,
    sampled_transport=nothing,
    cost_value=nothing,
    target_weight=nothing,
    target_metadata=nothing,
    cost_label="J",
    target_radius,
)
    evaluated_cost_value = if isnothing(cost_value)
        isnothing(sampled_transport) &&
            throw(ArgumentError("sampled_transport or cost_value is required"))
        sampled_target_cost(
            x,
            y,
            sampled_transport,
            target,
            target_radius,
            cost,
        )
    else
        cost_value
    end
    result = PlanetaryGeostrophy.compute_transport_adjoint_sensitivity(
        solution,
        x,
        y,
        z;
        target,
        cost,
        cost_gradient,
        cost_value=evaluated_cost_value,
        target_weight,
        target_metadata,
        cost_label,
        target_radius,
    )
    ranges = (
        wind_stress_x=finite_extrema(result.wind_stress_x),
        wind_stress_y=finite_extrema(result.wind_stress_y),
        surface_buoyancy=finite_extrema(result.surface_buoyancy),
        restoring_buoyancy=[
            finite_extrema(result.restoring_buoyancy[:, :, k])
            for k in eachindex(result.z)
        ],
        bottom_restoring_buoyancy=finite_extrema(
            result.bottom_restoring_buoyancy,
        ),
    )
    return (; result, ranges)
end

function plot_pressure_equation_diagnostics(
    diagnostics;
    arrow_scale=0.70,
    output=nothing,
)
    return PlanetaryGeostrophy.plot_surface_pressure_equation_diagnostics(
        diagnostics.result;
        arrow_scale,
        output,
    )
end

function plot_adjoint_sensitivity_diagnostics(
    diagnostics;
    buoyancy_levels=eachindex(diagnostics.result.z),
    output=nothing,
)
    return PlanetaryGeostrophy.plot_transport_adjoint_sensitivity(
        diagnostics.result;
        buoyancy_levels,
        output,
    )
end

end
