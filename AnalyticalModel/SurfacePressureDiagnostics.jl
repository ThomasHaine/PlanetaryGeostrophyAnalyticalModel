module SurfacePressureDiagnostics

import PlanetaryGeostrophy

export compute_pressure_equation_diagnostics,
       compute_adjoint_sensitivity_diagnostics,
       plot_pressure_equation_diagnostics,
       plot_adjoint_sensitivity_diagnostics

finite_extrema(values) = extrema(filter(isfinite, vec(values)))

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
        solution, x, y, z; target, cost, cost_label="J", target_radius,
    )

Compute spatial forcing-sensitivity kernels and summarize the finite range of
each kernel for compact notebook reporting.
"""
function compute_adjoint_sensitivity_diagnostics(
    solution,
    x,
    y,
    z;
    target,
    cost,
    cost_label="J",
    target_radius,
)
    result = PlanetaryGeostrophy.compute_transport_adjoint_sensitivity(
        solution,
        x,
        y,
        z;
        target,
        cost,
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
