module PlanetaryGeostrophy

using ForwardDiff
using Gridap
import Interpolations
using OrdinaryDiffEqTsit5
using Plots
using Printf
using WriteVTK

stage_metrics(measurement) = (
    seconds=measurement.time,
    allocated_mib=measurement.bytes / 2.0^20,
    gc_seconds=measurement.gctime,
)

function timed_stage(f, label)
    @info "Starting $label"
    measurement = @timed f()
    metrics = stage_metrics(measurement)
    @info "Completed $label" seconds=round(metrics.seconds; digits=3) allocated_mib=round(metrics.allocated_mib; digits=2) gc_seconds=round(metrics.gc_seconds; digits=3)
    return measurement.value, metrics
end

include(joinpath(@__DIR__, "..", "..", "TheoryNotes", "VelocityFieldPrototype.jl"))
using .VelocityFieldPrototype
include(joinpath(@__DIR__, "FastColumns.jl"))
include(joinpath(@__DIR__, "..", "..", "AnalyticalModel", "SurfacePressureTools.jl"))

export CellBudgetQuadrature,
       ColumnQuadrature,
       PGParameters,
       PGProblem,
       PGSolution,
       PGFields,
       PGLagrangianVelocityField,
       PGLagrangianTrajectory,
       PGLagrangianTrajectories,
       PGStreamfunction,
       PGDiagnostics,
       PGDIMDiagnostics,
       PGBarotropicVorticityDiagnostics,
       PGVorticityDiagnostics,
       PGGeostrophicContourVorticityDiagnostics,
       PGTransportDivergenceVorticityDiagnostics,
       PGSurfacePressureEquationDiagnostics,
       PGForcingSensitivities,
       PGTransportAdjointSensitivity,
       rectangle_boundary,
       ellipse_boundary,
       solve_pg,
       surface_pressure,
       horizontal_velocity,
       buoyancy,
       bottom_stress,
       bottom_pressure,
       baroclinic_potential_energy,
       depth_integrated_current,
       solve_forcing_sensitivities,
       horizontal_velocity_sensitivity,
       depth_integrated_current_sensitivity,
       compute_transport_adjoint_sensitivity,
       transport_adjoint_identity,
       depth_integrated_momentum_terms,
       barotropic_vorticity_terms,
       depth_integrated_vorticity_terms,
       geostrophic_contour_vorticity_terms,
       transport_divergence_vorticity_terms,
       surface_pressure_equation_coefficients,
       solve_streamfunction,
       streamfunction,
       streamfunction_current,
       sample_solution,
       sample_lagrangian_velocity,
       lagrangian_velocity,
       compute_trajectories,
       compute_diagnostics,
       compute_dim_diagnostics,
       compute_barotropic_vorticity_diagnostics,
       compute_vorticity_diagnostics,
       compute_geostrophic_contour_vorticity_diagnostics,
       compute_transport_divergence_vorticity_diagnostics,
       compute_surface_pressure_equation_diagnostics,
       write_solution,
       plot_solution,
       plot_trajectories,
       animate_trajectories,
       plot_surface_pressure,
       plot_bottom_stress,
       plot_depth_integrated_current,
       plot_streamfunction,
       plot_diagnostics,
       plot_dim_balance,
       plot_barotropic_vorticity_balance,
       plot_vorticity_balance,
       plot_geostrophic_contour_vorticity_balance,
       plot_transport_divergence_vorticity_balance,
       plot_surface_pressure_equation_diagnostics,
       plot_baroclinic_potential_energy,
       plot_bottom_pressure,
       plot_velocity_sensitivity,
       plot_transport_sensitivity,
       plot_transport_adjoint_sensitivity,
       performance_summary

"""
    PGParameters(Ek, Γ; viscosity=1, diffusivity=1)

Non-dimensional parameters for the steady planetary-geostrophic problem.
`viscosity` and `diffusivity` multiply `Ek` in the momentum and buoyancy
equations, respectively.
"""
struct PGParameters{T<:Real}
    Ek::T
    Γ::T
    viscosity::T
    diffusivity::T
end

function PGParameters(Ek::Real, Γ::Real; viscosity::Real=1, diffusivity::Real=1)
    values = promote(float(Ek), float(Γ), float(viscosity), float(diffusivity))
    all(>(0), values) ||
        throw(ArgumentError("Ek, Γ, viscosity, and diffusivity must be positive"))
    return PGParameters(values...)
end

function point_in_polygon(x, y, boundary)
    inside = false
    j = length(boundary)
    for i in eachindex(boundary)
        xi, yi = boundary[i]
        xj, yj = boundary[j]
        crosses = (yi > y) != (yj > y)
        if crosses && x < (xj - xi) * (y - yi) / (yj - yi) + xi
            inside = !inside
        end
        j = i
    end
    return inside
end

automatic_x(field) = (x, y) -> ForwardDiff.derivative(xq -> field(xq, y), x)
automatic_y(field) = (x, y) -> ForwardDiff.derivative(yq -> field(x, yq), y)

"""
    PGProblem(; parameters, depth, coriolis, wind_stress,
                restoring_buoyancy, surface_buoyancy, boundary, ...)

Specify a non-dimensional PG problem. Horizontal vector fields use complex
notation: `wind_stress(x, y) = τx + im * τy`. The boundary is a counter- or
clockwise polygon of `(x, y)` tuples. Derivatives may be supplied explicitly;
otherwise automatic differentiation is used. Supply
`wind_stress_curl(x, y)` when an exact expression is available or the stress
callable does not accept automatic-differentiation number types.
"""
struct PGProblem{P,H,F,FX,FY,T,TC,B,BS,HX,HY,BD,I}
    parameters::P
    depth::H
    coriolis::F
    coriolis_x::FX
    coriolis_y::FY
    wind_stress::T
    wind_stress_curl::TC
    restoring_buoyancy::B
    surface_buoyancy::BS
    depth_x::HX
    depth_y::HY
    boundary::BD
    inside::I
    derivative_step::Float64
end

function PGProblem(;
    parameters::PGParameters,
    depth,
    coriolis,
    wind_stress,
    restoring_buoyancy,
    surface_buoyancy,
    boundary,
    coriolis_x=nothing,
    coriolis_y=nothing,
    wind_stress_curl=nothing,
    depth_x=nothing,
    depth_y=nothing,
    inside=nothing,
    derivative_step=1e-5,
)
    length(boundary) >= 3 || throw(ArgumentError("boundary needs at least three points"))
    derivative_step > 0 || throw(ArgumentError("derivative_step must be positive"))
    polygon = collect(boundary)
    inside_fn = isnothing(inside) ?
        ((x, y) -> point_in_polygon(x, y, polygon)) : inside
    fx = isnothing(coriolis_x) ? automatic_x(coriolis) : coriolis_x
    fy = isnothing(coriolis_y) ? automatic_y(coriolis) : coriolis_y
    hx = isnothing(depth_x) ? automatic_x(depth) : depth_x
    hy = isnothing(depth_y) ? automatic_y(depth) : depth_y
    τx = (x, y) -> real(wind_stress(x, y))
    τy = (x, y) -> imag(wind_stress(x, y))
    τy_x = automatic_x(τy)
    τx_y = automatic_y(τx)
    stress_curl = isnothing(wind_stress_curl) ?
        ((x, y) -> τy_x(x, y) - τx_y(x, y)) :
        wind_stress_curl

    cx = sum(first, polygon) / length(polygon)
    cy = sum(last, polygon) / length(polygon)
    depth(cx, cy) > 0 || throw(ArgumentError("depth must be positive in the basin"))
    coriolis(cx, cy) != 0 || throw(ArgumentError("coriolis must be nonzero in the basin"))

    return PGProblem(
        parameters, depth, coriolis, fx, fy, wind_stress, stress_curl,
        restoring_buoyancy, surface_buoyancy, hx, hy, polygon, inside_fn,
        float(derivative_step),
    )
end

struct PGSolution{P,K,S,B,V,T}
    problem::P
    kernels::K
    pressure::S
    buoyancy_function::B
    horizontal_velocity_function::V
    mesh_file::String
    timings::T
end

internal_parameters(p::PGParameters) =
    VelocityParameters(p.Ek, p.Γ, p.viscosity, p.diffusivity, 1.0, 0.0)

"""
    solve_pg(problem; mesh_file=tempname() * ".msh", mesh_resolution=0.1,
             pressure_order=2, overwrite_mesh=false, method=:fast,
             quadrature=ColumnQuadrature(), forcing_assembly=:quadrature,
             coastal_condition=:no_normal_flow)

Generate or reuse a surface mesh, solve the Gridap surface-pressure equation,
and construct callable Green-function solutions for buoyancy and horizontal
velocity.

`method=:fast` (the default) evaluates the transport forcing (69),
bottom-stress forcing (65), velocity (B5), and buoyancy (51) as single
vertical integrals with the fixed rule `quadrature`. Node and column work is
threaded, so start Julia with `-t auto`. `forcing_assembly=:quadrature`
evaluates `𝔅` at every finite-element quadrature point; `:nodal` interpolates
it from the nodes (see `solve_surface_pressure`). `method=:b23` keeps the
original B23 symbolic-kernel quadratures for comparison.

`coastal_condition=:no_normal_flow` (the default) enforces zero normal
depth-integrated transport through the coast, which matters where the coast
is a vertical wall (`H > 0`), and sets the mean of `p_s` to zero.
`:dirichlet` restores the earlier `p_s = 0` coastal condition.
"""
function solve_pg(
    problem::PGProblem;
    mesh_file=tempname() * ".msh",
    mesh_resolution=0.1,
    pressure_order=2,
    overwrite_mesh=false,
    method=:fast,
    quadrature=ColumnQuadrature(),
    forcing_assembly=:quadrature,
    coastal_condition=:no_normal_flow,
)
    mesh_resolution > 0 || throw(ArgumentError("mesh_resolution must be positive"))
    pressure_order > 0 || throw(ArgumentError("pressure_order must be positive"))
    solve_started = time()
    reusing_mesh = !overwrite_mesh &&
        pressure_mesh_is_current(mesh_file, problem.boundary; resolution=mesh_resolution)
    @info "PG solve started" Ek=problem.parameters.Ek Γ=problem.parameters.Γ viscosity=problem.parameters.viscosity diffusivity=problem.parameters.diffusivity pressure_order mesh_resolution mesh_file=string(mesh_file)
    mesh, mesh_timing = timed_stage(
        reusing_mesh ? "reusing surface mesh" : "generating surface mesh",
    ) do
        generate_pressure_mesh(
            mesh_file, problem.boundary;
            resolution=mesh_resolution, overwrite=overwrite_mesh,
        )
    end
    kernels, kernel_timing = timed_stage("loading analytical Green-function kernels") do
        load_symbolic_kernels()
    end
    p = internal_parameters(problem.parameters)
    pressure = solve_surface_pressure(
        mesh_file, kernels, p, problem.depth, problem.coriolis,
        problem.coriolis_x, problem.coriolis_y, problem.wind_stress,
        problem.restoring_buoyancy, problem.surface_buoyancy,
        problem.depth_x, problem.depth_y;
        order=pressure_order,
        method,
        quadrature,
        gradient_step=problem.derivative_step,
        forcing_assembly,
        coastal_condition,
    )
    b = if method === :b23
        (x, y, z) -> buoyancy_from_forcing(
            p, problem.depth, problem.restoring_buoyancy,
            problem.surface_buoyancy, x, y, z,
        )
    else
        (x, y, z) -> begin
            H = problem.depth(x, y)
            VFP.checked_depth(H, z)
            fast_buoyancy(
                p, H, problem.restoring_buoyancy,
                problem.surface_buoyancy(x, y), x, y, z, quadrature,
                panel_count(quadrature, p, problem.coriolis(x, y), H),
            )
        end
    end
    uv = if method === :b23
        (x, y, z) -> velocity_b23_from_gradps(
            kernels, p, problem.depth, problem.coriolis,
            problem.coriolis_x, problem.coriolis_y, pressure.gradps,
            problem.wind_stress, problem.restoring_buoyancy,
            problem.surface_buoyancy, problem.depth_x, problem.depth_y,
            x, y, z; gradient_step=problem.derivative_step,
        )
    else
        (x, y, z) -> fast_velocity_column(
            kernels, p, problem.depth, problem.coriolis,
            pressure.gradps(x, y), problem.wind_stress(x, y),
            problem.restoring_buoyancy, problem.surface_buoyancy,
            x, y, quadrature; step=problem.derivative_step,
        )(z)
    end
    total_seconds = time() - solve_started
    timings = (
        mesh=mesh_timing,
        kernels=kernel_timing,
        surface_pressure=pressure.timings,
        total_seconds,
    )
    @info "PG solve completed" seconds=round(total_seconds; digits=3) mesh_reused=reusing_mesh
    return PGSolution(problem, kernels, pressure, b, uv, string(mesh), timings)
end

function horizontal_velocity_column(solution::PGSolution, x, y)
    problem = solution.problem
    require_water(solution, x, y)
    if solution.pressure.method === :fast
        return horizontal_velocity_column(
            solution, x, y, solution.pressure.gradps(x, y),
        )
    end
    return VelocityFieldPrototype.velocity_b23_column_from_gradps(
        solution.kernels,
        internal_parameters(problem.parameters),
        problem.depth,
        problem.coriolis,
        problem.coriolis_x,
        problem.coriolis_y,
        solution.pressure.gradps,
        problem.wind_stress,
        problem.restoring_buoyancy,
        problem.surface_buoyancy,
        problem.depth_x,
        problem.depth_y,
        x,
        y;
        gradient_step=problem.derivative_step,
    )
end

"""
    horizontal_velocity_column(solution, x, y, gradps_value)

Column velocity evaluator given a precomputed `(∂x + i∂y)p_s` at `(x, y)`.
Callers that sample many columns in parallel use this to separate the
finite-element pressure-gradient lookup from the column work.
"""
function horizontal_velocity_column(solution::PGSolution, x, y, gradps_value)
    problem = solution.problem
    p = internal_parameters(problem.parameters)
    solution.pressure.method === :fast || return VelocityFieldPrototype.velocity_b23_column_from_gradps(
        solution.kernels, p, problem.depth, problem.coriolis,
        problem.coriolis_x, problem.coriolis_y, (_, _) -> gradps_value,
        problem.wind_stress, problem.restoring_buoyancy,
        problem.surface_buoyancy, problem.depth_x, problem.depth_y, x, y;
        gradient_step=problem.derivative_step,
    )
    return fast_velocity_column(
        solution.kernels, p, problem.depth, problem.coriolis,
        gradps_value, problem.wind_stress(x, y),
        problem.restoring_buoyancy, problem.surface_buoyancy, x, y,
        solution.pressure.quadrature; step=problem.derivative_step,
    )
end

function require_water(solution::PGSolution, x, y, z=nothing)
    problem = solution.problem
    problem.inside(x, y) || throw(DomainError((x, y), "point lies outside the basin"))
    H = problem.depth(x, y)
    H > 0 || throw(DomainError((x, y), "depth must be positive inside the basin"))
    if !isnothing(z)
        -H <= z <= 0 || throw(DomainError(z, "point lies outside the water column"))
    end
    return H
end

function surface_pressure(solution::PGSolution, x, y)
    require_water(solution, x, y)
    return solution.pressure.psurf_at(Point(x, y))
end

function horizontal_velocity(solution::PGSolution, x, y, z)
    require_water(solution, x, y, z)
    uv = solution.horizontal_velocity_function(x, y, z)
    return (u=real(uv), v=imag(uv))
end

function buoyancy(solution::PGSolution, x, y, z)
    require_water(solution, x, y, z)
    return solution.buoyancy_function(x, y, z)
end

"""
    bottom_stress(solution, x, y)

Evaluate the non-dimensional bottom stress, including surface-pressure,
buoyancy, and wind-stress contributions, using equations (63)--(65).
"""
function bottom_stress(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    problem = solution.problem
    p = internal_parameters(problem.parameters)
    pressure_term = bottom_stress_coefficient(
        p, H, problem.coriolis(x, y),
    ) * solution.pressure.gradps(x, y)
    stress = pressure_term + solution.pressure.T_fn(Point(x, y))
    return (x=real(stress), y=imag(stress))
end

function project_bottom_stress(solution::PGSolution; order=2)
    order > 0 || throw(ArgumentError("order must be positive"))
    started = time()
    model = solution.pressure.model
    V, setup_timing = timed_stage(
        "constructing bottom-stress interpolation space",
    ) do
        reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
        TestFESpace(model, reffe; conformity=:H1)
    end
    stress_field(x) = begin
        xx, yy = x[1], x[2]
        H = solution.problem.depth(xx, yy)
        H > 0 || return VectorValue(0.0, 0.0)
        problem = solution.problem
        p = internal_parameters(problem.parameters)
        pressure_term = bottom_stress_coefficient(
            p, H, problem.coriolis(xx, yy),
        ) * solution.pressure.gradps(xx, yy)
        stress = pressure_term + solution.pressure.T_fn(Point(xx, yy))
        return VectorValue(real(stress), imag(stress))
    end
    finite_element, interpolation_timing = timed_stage(
        "interpolating bottom stress",
    ) do
        stress_at, _ = nodal_lookup(stress_field, model, order)
        interpolate_everywhere(stress_at, V)
    end
    evaluator = PooledPointEvaluator(finite_element, solution.pressure.probe)
    value = (x, y) -> evaluator(Point(x, y))
    timings = (
        setup=setup_timing,
        interpolation=interpolation_timing,
        total_seconds=time() - started,
    )
    return (; finite_element, value, timings)
end

function analytical_surface_stress_curl(problem::PGProblem, x, y)
    return problem.wind_stress_curl(x, y)
end

function analytical_surface_stress_over_depth_curl(problem::PGProblem, x, y)
    H = problem.depth(x, y)
    stress = problem.wind_stress(x, y)
    return (
        analytical_surface_stress_curl(problem, x, y) / H +
        (
            real(stress) * problem.depth_y(x, y) -
            imag(stress) * problem.depth_x(x, y)
        ) / H^2
    )
end

function analytical_surface_stress_over_coriolis_curl(
    problem::PGProblem,
    x,
    y,
)
    f = problem.coriolis(x, y)
    stress = problem.wind_stress(x, y)
    return (
        analytical_surface_stress_curl(problem, x, y) / f +
        (
            real(stress) * problem.coriolis_y(x, y) -
            imag(stress) * problem.coriolis_x(x, y)
        ) / f^2
    )
end

"""
    bottom_pressure(solution, x, y)

Evaluate the bottom pressure
`p_b = p_s - ∫_{-H}^0 b dz` for conventional oceanographic buoyancy.
"""
function bottom_pressure(solution::PGSolution, x, y)
    require_water(solution, x, y)
    problem = solution.problem
    p = internal_parameters(problem.parameters)
    column_buoyancy = if solution.pressure.method === :fast
        quadrature = solution.pressure.quadrature
        fast_column_integral(
            solution.kernels, p, problem.depth, problem.restoring_buoyancy,
            problem.surface_buoyancy, x, y, quadrature,
            panel_count(
                quadrature, p, problem.coriolis(x, y), problem.depth(x, y),
            ),
        )
    else
        VelocityFieldPrototype.buoyancy_column_integral(
            solution.kernels, p, problem.depth, problem.restoring_buoyancy,
            problem.surface_buoyancy, x, y,
        )
    end
    return surface_pressure(solution, x, y) - column_buoyancy
end

"""
    baroclinic_potential_energy(solution, x, y)

Evaluate the baroclinic potential energy
`χ = -∫_{-H}^0 z b dz` from equation (34).
"""
function baroclinic_potential_energy(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    solution.pressure.method === :fast || return -VelocityFieldPrototype.split_quadgk(
        z -> z * solution.buoyancy_function(x, y, z),
        -H, 0.0,
    )
    quadrature = solution.pressure.quadrature
    problem = solution.problem
    p = internal_parameters(problem.parameters)
    panels = panel_count(quadrature, p, problem.coriolis(x, y), H)
    bs = problem.surface_buoyancy(x, y)
    return -panel_integral(
        z -> z * fast_buoyancy(
            p, H, problem.restoring_buoyancy, bs, x, y, z, quadrature, panels,
        ),
        -H, 0.0, quadrature, panels,
    )
end

"""
    depth_integrated_current(solution, x, y)

Evaluate the non-dimensional depth-integrated horizontal current `(U, V)`.
"""
function depth_integrated_current(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    p = internal_parameters(solution.problem.parameters)
    A = pressure_coefficient(p, H, solution.problem.coriolis(x, y))
    current = A * solution.pressure.gradps(x, y) +
              solution.pressure.B_fn(Point(x, y))
    return (U=real(current), V=imag(current))
end

struct PGForcingSensitivities{B,BS,TX,TY,T}
    restoring_buoyancy::B
    surface_buoyancy::BS
    wind_stress_x::TX
    wind_stress_y::TY
    timings::T
end

function forcing_component_problem(
    problem::PGProblem;
    restoring_buoyancy=(x, y, z) -> 0.0,
    surface_buoyancy=(x, y) -> 0.0,
    wind_stress=(x, y) -> 0.0 + 0.0im,
)
    return PGProblem(
        parameters=problem.parameters,
        depth=problem.depth,
        depth_x=problem.depth_x,
        depth_y=problem.depth_y,
        coriolis=problem.coriolis,
        coriolis_x=problem.coriolis_x,
        coriolis_y=problem.coriolis_y,
        wind_stress=wind_stress,
        restoring_buoyancy=restoring_buoyancy,
        surface_buoyancy=surface_buoyancy,
        boundary=problem.boundary,
        inside=problem.inside,
        derivative_step=problem.derivative_step,
    )
end

"""
    solve_forcing_sensitivities(problem; restoring_buoyancy, surface_buoyancy,
                                wind_stress_x, wind_stress_y, solve keywords...)

Solve the exact linear response to four independently scaled forcing patterns.
Each returned solution is the derivative with respect to the scalar multiplier
of its pattern, with all other forcing terms set to zero. By default the
patterns are the corresponding components of `problem`; pass a nonzero
`wind_stress_y` pattern when the base problem has no meridional stress.
Geometry and dynamical parameters are held fixed.
"""
function solve_forcing_sensitivities(
    problem::PGProblem;
    restoring_buoyancy=problem.restoring_buoyancy,
    surface_buoyancy=problem.surface_buoyancy,
    wind_stress_x=(x, y) -> real(problem.wind_stress(x, y)),
    wind_stress_y=(x, y) -> imag(problem.wind_stress(x, y)),
    mesh_file=tempname() * ".msh",
    mesh_resolution=0.1,
    pressure_order=2,
    overwrite_mesh=false,
)
    started = time()
    specifications = (
        restoring_buoyancy=forcing_component_problem(
            problem; restoring_buoyancy,
        ),
        surface_buoyancy=forcing_component_problem(
            problem; surface_buoyancy,
        ),
        wind_stress_x=forcing_component_problem(
            problem;
            wind_stress=(x, y) -> complex(wind_stress_x(x, y), 0.0),
        ),
        wind_stress_y=forcing_component_problem(
            problem;
            wind_stress=(x, y) -> complex(0.0, wind_stress_y(x, y)),
        ),
    )
    function solve_component(name, specification; overwrite=false)
        @info "Solving forcing sensitivity" forcing=name
        return solve_pg(
            specification;
            mesh_file,
            mesh_resolution,
            pressure_order,
            overwrite_mesh=overwrite,
        )
    end
    named_solutions = (
        restoring_buoyancy=solve_component(
            :restoring_buoyancy,
            specifications.restoring_buoyancy;
            overwrite=overwrite_mesh,
        ),
        surface_buoyancy=solve_component(
            :surface_buoyancy,
            specifications.surface_buoyancy,
        ),
        wind_stress_x=solve_component(
            :wind_stress_x,
            specifications.wind_stress_x,
        ),
        wind_stress_y=solve_component(
            :wind_stress_y,
            specifications.wind_stress_y,
        ),
    )
    timings = (
        restoring_buoyancy=named_solutions.restoring_buoyancy.timings,
        surface_buoyancy=named_solutions.surface_buoyancy.timings,
        wind_stress_x=named_solutions.wind_stress_x.timings,
        wind_stress_y=named_solutions.wind_stress_y.timings,
        total_seconds=time() - started,
    )
    return PGForcingSensitivities(
        named_solutions.restoring_buoyancy,
        named_solutions.surface_buoyancy,
        named_solutions.wind_stress_x,
        named_solutions.wind_stress_y,
        timings,
    )
end

forcing_sensitivity_pairs(sensitivity::PGForcingSensitivities) = (
    restoring_buoyancy=sensitivity.restoring_buoyancy,
    surface_buoyancy=sensitivity.surface_buoyancy,
    wind_stress_x=sensitivity.wind_stress_x,
    wind_stress_y=sensitivity.wind_stress_y,
)

"""
    horizontal_velocity_sensitivity(sensitivity, x, y, z)

Return the exact `(u, v)` derivatives at `(x, y, z)` with respect to the four
forcing-pattern multipliers.
"""
function horizontal_velocity_sensitivity(
    sensitivity::PGForcingSensitivities,
    x,
    y,
    z,
)
    return map(
        solution -> horizontal_velocity(solution, x, y, z),
        forcing_sensitivity_pairs(sensitivity),
    )
end

"""
    depth_integrated_current_sensitivity(sensitivity, x, y)

Return the exact `(U, V)` depth-integrated-current derivatives at `(x, y)`
with respect to the four forcing-pattern multipliers.
"""
function depth_integrated_current_sensitivity(
    sensitivity::PGForcingSensitivities,
    x,
    y,
)
    return map(
        solution -> depth_integrated_current(solution, x, y),
        forcing_sensitivity_pairs(sensitivity),
    )
end

struct PGTransportAdjointSensitivity{
    TX<:AbstractVector,TY<:AbstractVector,TZ<:AbstractVector,TA,T
}
    x::TX
    y::TY
    z::TZ
    wind_stress_x::Matrix{Float64}
    wind_stress_y::Matrix{Float64}
    surface_buoyancy::Matrix{Float64}
    restoring_buoyancy::Array{Float64,3}
    bottom_restoring_buoyancy::Matrix{Float64}
    surface_buoyancy_local::Matrix{Float64}
    surface_buoyancy_gradient_x::Matrix{Float64}
    surface_buoyancy_gradient_y::Matrix{Float64}
    restoring_buoyancy_local::Array{Float64,3}
    restoring_buoyancy_gradient_x::Array{Float64,3}
    restoring_buoyancy_gradient_y::Array{Float64,3}
    transport_forcing_x::Matrix{Float64}
    transport_forcing_y::Matrix{Float64}
    wet::BitMatrix
    wet_3d::BitArray{3}
    target::NamedTuple
    cost_label::String
    cost_value::Float64
    target_radius::Float64
    adjoint::TA
    timings::T
end

adjoint_pairing(weight, coefficient) =
    real(conj(weight) * coefficient)

"""
    transport_adjoint_identity(sensitivity, response)

Compare the exact finite-element directional derivative of the transport cost
with its adjoint pairing for a linear forward `response`. This bypasses the
sampled plotting grids, so `relative_error` tests the pressure adjoint itself
rather than horizontal or vertical post-processing quadrature.
"""
function transport_adjoint_identity(
    sensitivity::PGTransportAdjointSensitivity,
    response::PGSolution,
)
    adjoint = sensitivity.adjoint
    pressure = response.pressure
    Ω = Triangulation(pressure.model)
    dΩ = Measure(Ω, 2 * pressure.order)
    components = sum(∫(
        point -> begin
            x, y = point[1], point[2]
            ρ = adjoint.weight(x, y)
            derivatives = adjoint.cost_derivatives(x, y)
            forcing = pressure.B_fn(point)
            pressure_transport =
                pressure.A_fn(point) * pressure.gradps(x, y)
            λ = adjoint.gradient(x, y)
            direct = ρ * (
                derivatives.cost_U * real(forcing) +
                derivatives.cost_V * imag(forcing)
            )
            forward_pressure = ρ * (
                derivatives.cost_U * real(pressure_transport) +
                derivatives.cost_V * imag(pressure_transport)
            )
            adjoint_pressure =
                -λ[1] * real(forcing) - λ[2] * imag(forcing)
            VectorValue(direct, forward_pressure, adjoint_pressure)
        end
    )dΩ)
    direct, forward_pressure, adjoint_pressure = Tuple(components)
    forward = direct + forward_pressure
    adjoint_value = direct + adjoint_pressure
    relative_error = abs(forward - adjoint_value) /
        max(abs(forward), abs(adjoint_value), eps(Float64))
    return (;
        forward,
        adjoint=adjoint_value,
        direct,
        forward_pressure,
        adjoint_pressure,
        relative_error,
    )
end

function forcing_kernel_from_local_operator(
    local_values::AbstractMatrix,
    gradient_x::AbstractMatrix,
    gradient_y::AbstractMatrix,
    wet,
    x,
    y
)
    kernel = fill(NaN, size(local_values))
    for j in eachindex(y), i in eachindex(x)
        centered_stencil_is_wet(wet, i, j) || continue
        kernel[i, j] = (
            local_values[i, j] -
            centered_horizontal_derivative(
                gradient_x, wet, x, i, j, 1,
            ) -
            centered_horizontal_derivative(
                gradient_y, wet, y, i, j, 2,
            )
        )
    end
    return kernel
end

function forcing_kernel_from_local_operator(
    local_values::AbstractArray{<:Real,3},
    gradient_x::AbstractArray{<:Real,3},
    gradient_y::AbstractArray{<:Real,3},
    wet,
    x,
    y
)
    kernel = fill(NaN, size(local_values))
    for k in axes(local_values, 3), j in eachindex(y), i in eachindex(x)
        wet2 = @view wet[:, :, k]
        centered_stencil_is_wet(wet2, i, j) || continue
        local2 = @view local_values[:, :, k]
        gradient_x2 = @view gradient_x[:, :, k]
        gradient_y2 = @view gradient_y[:, :, k]
        kernel[i, j, k] = (
            local2[i, j] -
            centered_horizontal_derivative(
                gradient_x2, wet2, x, i, j, 1,
            ) -
            centered_horizontal_derivative(
                gradient_y2, wet2, y, i, j, 2,
            )
        )
    end
    return kernel
end

"""
    compute_transport_adjoint_sensitivity(
        solution, x, y, z;
        target, cost, cost_gradient=nothing, cost_value=nothing,
        transport=nothing, target_weight=nothing, target_metadata=nothing,
        cost_label="J", target_radius,
    )

Compute spatial functional-derivative kernels for a normalized Gaussian
average of the real scalar `cost(U, V)` centered at `target=(x, y)`.
Nonlinear costs are linearized about the solved transport. The
surface-pressure adjoint reverses the pressure operator's advecting velocity.
Supply an analytical `cost_gradient(U, V)` when available; omitting it uses
`ForwardDiff`. A constant two-entry tuple is the efficient path for linear
costs. `cost_value` and `transport` can reuse sampled or projected base-state
data instead of reevaluating analytical transport during adjoint assembly.
`target_weight(x, y)` replaces the default normalized Gaussian when supplied;
it is used without further normalization. `target_metadata` controls how that
region is described in sensitivity plots.
The returned 2D fields are derivatives with respect to local `τx`, `τy`, and
`bs`; the 3D field is the derivative with respect to local restoring
buoyancy `B`. The corresponding `*_local` and `*_gradient_*` fields retain
the coefficients before horizontal integration by parts, allowing direct
directional-derivative checks without differentiating the sampled kernels.
`cost_label` is used in plot annotations.
"""
function compute_transport_adjoint_sensitivity(
    solution::PGSolution,
    x_values,
    y_values,
    z_values;
    target,
    cost,
    cost_gradient=nothing,
    cost_value=nothing,
    transport=nothing,
    target_weight=nothing,
    target_metadata=nothing,
    cost_label="J",
    target_radius,
)
    started = time()
    x = collect(float.(x_values))
    y = collect(float.(y_values))
    z = collect(float.(z_values))
    length(x) >= 3 && length(y) >= 3 ||
        throw(ArgumentError("x and y must each contain at least three points"))
    !isempty(z) || throw(ArgumentError("z must contain at least one level"))
    issorted(x) && all(diff(x) .> 0) ||
        throw(ArgumentError("x must be strictly increasing"))
    issorted(y) && all(diff(y) .> 0) ||
        throw(ArgumentError("y must be strictly increasing"))
    issorted(z) && all(diff(z) .> 0) ||
        throw(ArgumentError("z must be strictly increasing"))
    target_x, target_y = target
    require_water(solution, target_x, target_y)
    adjoint, adjoint_timing = timed_stage(
        "solving transport adjoint",
    ) do
        solve_surface_pressure_adjoint(
            solution.pressure,
            target_x,
            target_y;
            cost,
            cost_gradient,
            cost_value,
            transport,
            target_weight,
            target_radius,
        )
    end

    shape = (length(x), length(y))
    shape3 = (length(x), length(y), length(z))
    transport_x = fill(NaN, shape)
    transport_y = fill(NaN, shape)
    stress_x = fill(NaN, shape)
    stress_y = fill(NaN, shape)
    surface_local = fill(NaN, shape)
    surface_gradient_x = fill(NaN, shape)
    surface_gradient_y = fill(NaN, shape)
    bottom_buoyancy = fill(NaN, shape)
    interior_local = fill(NaN, shape3)
    interior_gradient_x = fill(NaN, shape3)
    interior_gradient_y = fill(NaN, shape3)
    wet = falses(shape)
    wet_3d = falses(shape3)
    problem = solution.problem
    p = internal_parameters(problem.parameters)

    _, sampling_timing = timed_stage(
        "sampling adjoint forcing kernels",
    ) do
        for j in eachindex(y), i in eachindex(x)
            xx, yy = x[i], y[j]
            problem.inside(xx, yy) || continue
            H = problem.depth(xx, yy)
            H > 0 || continue
            adjoint_gradient = adjoint.gradient(xx, yy)
            λx, λy = adjoint_gradient[1], adjoint_gradient[2]
            ρ = adjoint.weight(xx, yy)
            derivatives = adjoint.cost_derivatives(xx, yy)
            transport_kernel = complex(
                ρ * derivatives.cost_U - λx,
                ρ * derivatives.cost_V - λy,
            )
            transport_x[i, j] = real(transport_kernel)
            transport_y[i, j] = imag(transport_kernel)

            wind_coefficient = wind_transport_coefficient(
                p, H, problem.coriolis(xx, yy), 1.0 + 0.0im,
            )
            stress_x[i, j] =
                adjoint_pairing(transport_kernel, wind_coefficient)
            stress_y[i, j] = adjoint_pairing(
                transport_kernel, im * wind_coefficient,
            )

            coefficients =
                buoyancy_transport_sensitivity_coefficients(
                    solution.kernels,
                    p,
                    problem.depth,
                    problem.coriolis,
                    problem.coriolis_x,
                    problem.coriolis_y,
                    problem.depth_x,
                    problem.depth_y,
                    xx,
                    yy,
                    -H / 2,
                    gradient_step=problem.derivative_step,
                )
            surface_local[i, j] = adjoint_pairing(
                transport_kernel, coefficients.surface.value,
            )
            surface_gradient_x[i, j] = adjoint_pairing(
                transport_kernel, coefficients.surface.gradient,
            )
            surface_gradient_y[i, j] = adjoint_pairing(
                transport_kernel, im * coefficients.surface.gradient,
            )
            wet[i, j] = true
            bottom_coefficients =
                buoyancy_transport_sensitivity_coefficients(
                    solution.kernels,
                    p,
                    problem.depth,
                    problem.coriolis,
                    problem.coriolis_x,
                    problem.coriolis_y,
                    problem.depth_x,
                    problem.depth_y,
                    xx,
                    yy,
                    -H,
                    gradient_step=problem.derivative_step,
                ).interior
            bottom_transport_coefficient =
                bottom_coefficients.gradient *
                complex(
                    problem.depth_x(xx, yy),
                    problem.depth_y(xx, yy),
                )
            bottom_buoyancy[i, j] = adjoint_pairing(
                transport_kernel, bottom_transport_coefficient,
            )

            for k in eachindex(z)
                -H <= z[k] <= 0 || continue
                interior =
                    buoyancy_transport_sensitivity_coefficients(
                        solution.kernels,
                        p,
                        problem.depth,
                        problem.coriolis,
                        problem.coriolis_x,
                        problem.coriolis_y,
                        problem.depth_x,
                        problem.depth_y,
                        xx,
                        yy,
                        z[k],
                        gradient_step=problem.derivative_step,
                    ).interior
                interior_local[i, j, k] =
                    adjoint_pairing(transport_kernel, interior.value)
                interior_gradient_x[i, j, k] =
                    adjoint_pairing(transport_kernel, interior.gradient)
                interior_gradient_y[i, j, k] =
                    adjoint_pairing(
                        transport_kernel, im * interior.gradient,
                    )
                wet_3d[i, j, k] = true
            end
        end
    end

    surface_kernel = forcing_kernel_from_local_operator(
        surface_local,
        surface_gradient_x,
        surface_gradient_y,
        wet,
        x,
        y,
    )
    interior_kernel = forcing_kernel_from_local_operator(
        interior_local,
        interior_gradient_x,
        interior_gradient_y,
        wet_3d,
        x,
        y,
    )
    kernel_wet_3d = isfinite.(interior_kernel)
    timings = (
        adjoint=adjoint_timing,
        sampling=sampling_timing,
        total_seconds=time() - started,
    )
    diagnostic_target = isnothing(target_metadata) ?
        (kind=:point, x=float(target_x), y=float(target_y)) :
        target_metadata
    return PGTransportAdjointSensitivity(
        x,
        y,
        z,
        stress_x,
        stress_y,
        surface_kernel,
        interior_kernel,
        bottom_buoyancy,
        surface_local,
        surface_gradient_x,
        surface_gradient_y,
        interior_local,
        interior_gradient_x,
        interior_gradient_y,
        transport_x,
        transport_y,
        BitMatrix(wet),
        BitArray(kernel_wet_3d),
        diagnostic_target,
        string(cost_label),
        adjoint.cost_value,
        float(target_radius),
        adjoint,
        timings,
    )
end

"""
    depth_integrated_momentum_terms(solution, x, y)

Evaluate every signed term in the two depth-integrated momentum equations
corresponding to (32)--(33). Residuals are left-hand side minus the sum of
right-hand-side terms.
"""
function depth_integrated_momentum_terms(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    problem = solution.problem
    step = problem.derivative_step
    current = depth_integrated_current(solution, x, y)
    stress = bottom_stress(solution, x, y)
    bottom_pressure_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> bottom_pressure(solution, xx, yy),
        x, y; step,
    )
    χ_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> baroclinic_potential_energy(solution, xx, yy),
        x, y; step,
    )
    wind = problem.wind_stress(x, y)
    f = problem.coriolis(x, y)

    Ek = problem.parameters.Ek
    x_terms = (
        coriolis=-f * current.V,
        bottom_pressure=-H * real(bottom_pressure_gradient),
        baroclinic=-real(χ_gradient),
        bottom_stress=Ek * stress.x,
        surface_stress=Ek * real(wind),
    )
    y_terms = (
        coriolis=f * current.U,
        bottom_pressure=-H * imag(bottom_pressure_gradient),
        baroclinic=-imag(χ_gradient),
        bottom_stress=Ek * stress.y,
        surface_stress=Ek * imag(wind),
    )
    x_residual = x_terms.coriolis - (
        x_terms.bottom_pressure + x_terms.baroclinic +
        x_terms.bottom_stress + x_terms.surface_stress
    )
    y_residual = y_terms.coriolis - (
        y_terms.bottom_pressure + y_terms.baroclinic +
        y_terms.bottom_stress + y_terms.surface_stress
    )
    return (; x=x_terms, y=y_terms, x_residual, y_residual)
end

function centered_field_derivatives(field, x, y, step)
    x_derivative = (field(x + step, y) - field(x - step, y)) / (2step)
    y_derivative = (field(x, y + step) - field(x, y - step)) / (2step)
    return (; x=x_derivative, y=y_derivative)
end

"""
    barotropic_vorticity_terms(solution, x, y)

Evaluate every signed term in the barotropic vorticity equation (36),

`f div(U) + U f_x + V f_y = J(p_b, H) + Ek curl(τ_b) + Ek curl(τ)`,

where `J(p_b, H) = p_b,x H_y - p_b,y H_x` and
`curl(τ) = τ_y,x - τ_x,y`. On a beta plane the planetary-vorticity term
is `β V`. The fixed-lid, no-normal-flow mass balance gives
`div(U) = 0` exactly. The residual is the left-hand side minus the sum of the
right-hand-side terms.
"""
function barotropic_vorticity_terms(solution::PGSolution, x, y)
    require_water(solution, x, y)
    problem = solution.problem
    step = problem.derivative_step

    current = depth_integrated_current(solution, x, y)
    transport_divergence = zero(current.U)
    planetary_vorticity_advection = (
        current.U * problem.coriolis_x(x, y) +
        current.V * problem.coriolis_y(x, y)
    )

    bottom_pressure_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> bottom_pressure(solution, xx, yy),
        x, y; step,
    )
    bottom_pressure_torque = (
        real(bottom_pressure_gradient) * problem.depth_y(x, y) -
        imag(bottom_pressure_gradient) * problem.depth_x(x, y)
    )

    bottom_stress_complex(xx, yy) = begin
        stress = bottom_stress(solution, xx, yy)
        stress.x + im * stress.y
    end
    bottom_stress_derivatives = centered_field_derivatives(
        bottom_stress_complex, x, y, step,
    )
    Ek = problem.parameters.Ek
    bottom_stress_curl = Ek * (
        imag(bottom_stress_derivatives.x) -
        real(bottom_stress_derivatives.y)
    )
    surface_stress_curl = Ek * analytical_surface_stress_curl(problem, x, y)
    residual = transport_divergence + planetary_vorticity_advection -
        bottom_pressure_torque - bottom_stress_curl - surface_stress_curl
    return (;
        transport_divergence,
        planetary_vorticity_advection,
        bottom_pressure_torque,
        bottom_stress_curl,
        surface_stress_curl,
        residual,
    )
end

"""
    depth_integrated_vorticity_terms(solution, x, y)

Evaluate every signed term in the depth-integrated vorticity equation (41),

`U f_x + V f_y + f w_b = Ek curl(τ_b) + Ek curl(τ)`,

where `w_b = -(u_b H_x + v_b H_y)` is the bottom vertical velocity and
`curl(τ) = τ_y,x - τ_x,y`. At a sloping bottom, the bottom torque in (41)
uses horizontal derivatives at fixed `z`, not derivatives following
`z=-H(x,y)`. The chain-rule correction from the latter to the former is
`f w_b + J(p_b,H)`. On a beta plane the first term is `β V`. The residual is
the left-hand side minus the sum of the right-hand-side terms.
"""
function depth_integrated_vorticity_terms(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    problem = solution.problem
    step = problem.derivative_step
    current = depth_integrated_current(solution, x, y)
    bottom_velocity = horizontal_velocity(solution, x, y, -H)
    bottom_vertical_velocity = -(
        bottom_velocity.u * problem.depth_x(x, y) +
        bottom_velocity.v * problem.depth_y(x, y)
    )

    bottom_stress_complex(xx, yy) = begin
        stress = bottom_stress(solution, xx, yy)
        stress.x + im * stress.y
    end

    bottom_stress_x = (
        bottom_stress_complex(x + step, y) -
        bottom_stress_complex(x - step, y)
    ) / (2step)
    bottom_stress_y = (
        bottom_stress_complex(x, y + step) -
        bottom_stress_complex(x, y - step)
    ) / (2step)
    bottom_pressure_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> bottom_pressure(solution, xx, yy),
        x, y; step,
    )
    bottom_pressure_torque = (
        real(bottom_pressure_gradient) * problem.depth_y(x, y) -
        imag(bottom_pressure_gradient) * problem.depth_x(x, y)
    )

    planetary_vorticity_advection = (
        current.U * problem.coriolis_x(x, y) +
        current.V * problem.coriolis_y(x, y)
    )
    bottom_vortex_stretching =
        problem.coriolis(x, y) * bottom_vertical_velocity
    Ek = problem.parameters.Ek
    terrain_following_bottom_stress_curl =
        Ek * (imag(bottom_stress_x) - real(bottom_stress_y))
    bottom_stress_curl = terrain_following_bottom_stress_curl +
        bottom_vortex_stretching + bottom_pressure_torque
    surface_stress_curl = Ek * analytical_surface_stress_curl(problem, x, y)
    residual = planetary_vorticity_advection + bottom_vortex_stretching -
        (bottom_stress_curl + surface_stress_curl)
    return (;
        planetary_vorticity_advection,
        bottom_vortex_stretching,
        bottom_stress_curl,
        surface_stress_curl,
        residual,
    )
end

"""
    geostrophic_contour_vorticity_terms(solution, x, y)

Evaluate every signed term in the geostrophic-contour vorticity balance (43),

`J(f/H, Ψ) = J(1/H, χ) - Ek curl((τ + τ_b)/H)`.

The returned stress curls include `Ek` but not the minus sign shown on the
right-hand side. Thus the residual is
`geostrophic_contour_advection - jebar + bottom_stress_curl +
surface_stress_curl`.
"""
function geostrophic_contour_vorticity_terms(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    problem = solution.problem
    step = problem.derivative_step
    f = problem.coriolis(x, y)
    H_x, H_y = problem.depth_x(x, y), problem.depth_y(x, y)
    current = depth_integrated_current(solution, x, y)

    f_over_H_x = (problem.coriolis_x(x, y) * H - f * H_x) / H^2
    f_over_H_y = (problem.coriolis_y(x, y) * H - f * H_y) / H^2
    geostrophic_contour_advection =
        -f_over_H_x * current.U - f_over_H_y * current.V

    χ_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> baroclinic_potential_energy(solution, xx, yy),
        x, y; step,
    )
    jebar = (
        H_y * real(χ_gradient) - H_x * imag(χ_gradient)
    ) / H^2

    bottom_stress_over_depth(xx, yy) = begin
        stress = bottom_stress(solution, xx, yy)
        local_depth = problem.depth(xx, yy)
        (stress.x + im * stress.y) / local_depth
    end

    bottom_stress_derivatives = centered_field_derivatives(
        bottom_stress_over_depth, x, y, step,
    )
    Ek = problem.parameters.Ek
    bottom_stress_curl = Ek * (
        imag(bottom_stress_derivatives.x) -
        real(bottom_stress_derivatives.y)
    )
    surface_stress_curl =
        Ek * analytical_surface_stress_over_depth_curl(problem, x, y)
    residual = geostrophic_contour_advection - jebar +
        bottom_stress_curl + surface_stress_curl
    return (;
        geostrophic_contour_advection,
        jebar,
        bottom_stress_curl,
        surface_stress_curl,
        residual,
    )
end

"""
    transport_divergence_vorticity_terms(solution, x, y)

Evaluate every signed term in the transport-divergence vorticity balance (47),

`0 = J(p_b, H/f) + J(χ, 1/f) + Ek curl((τ + τ_b)/f)`.

The residual follows the package convention of left-hand side minus
right-hand side and is therefore the negative sum of the four returned
physical terms.
"""
function transport_divergence_vorticity_terms(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    problem = solution.problem
    step = problem.derivative_step
    f = problem.coriolis(x, y)
    f_x, f_y = problem.coriolis_x(x, y), problem.coriolis_y(x, y)
    H_x, H_y = problem.depth_x(x, y), problem.depth_y(x, y)

    bottom_pressure_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> bottom_pressure(solution, xx, yy),
        x, y; step,
    )
    H_over_f_x = (H_x * f - H * f_x) / f^2
    H_over_f_y = (H_y * f - H * f_y) / f^2
    bottom_pressure_torque = (
        real(bottom_pressure_gradient) * H_over_f_y -
        imag(bottom_pressure_gradient) * H_over_f_x
    )

    χ_gradient = VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> baroclinic_potential_energy(solution, xx, yy),
        x, y; step,
    )
    baroclinic_torque = (
        -f_y * real(χ_gradient) + f_x * imag(χ_gradient)
    ) / f^2

    bottom_stress_over_coriolis(xx, yy) = begin
        stress = bottom_stress(solution, xx, yy)
        local_f = problem.coriolis(xx, yy)
        (stress.x + im * stress.y) / local_f
    end
    bottom_stress_derivatives = centered_field_derivatives(
        bottom_stress_over_coriolis, x, y, step,
    )
    Ek = problem.parameters.Ek
    bottom_stress_curl = Ek * (
        imag(bottom_stress_derivatives.x) -
        real(bottom_stress_derivatives.y)
    )
    surface_stress_curl =
        Ek * analytical_surface_stress_over_coriolis_curl(problem, x, y)
    residual = -(
        bottom_pressure_torque + baroclinic_torque +
        bottom_stress_curl + surface_stress_curl
    )
    return (;
        bottom_pressure_torque,
        baroclinic_torque,
        bottom_stress_curl,
        surface_stress_curl,
        residual,
    )
end

"""
    surface_pressure_equation_coefficients(solution, x, y)

Evaluate the coefficients in the surface-pressure advection-diffusion equation
(75),

`div(A_r grad(p_s)) + velocity ⋅ grad(p_s) = source`,

where `velocity = velocity_inviscid + velocity_frictional`,
`velocity_inviscid = (∂y(H/f), -∂x(H/f))`, `diffusivity = A_r`, and
`source = -∂x B_r - ∂y B_i`. The total velocity retains the derivatives of
the exact `A_i` used by the pressure solve. The source is also returned as
separate baroclinic and surface-stress contributions from
`B = B_baroclinic + B_stress`.
"""
function surface_pressure_equation_coefficients(
    solution::PGSolution,
    x,
    y,
)
    step = solution.problem.derivative_step
    for (xx, yy) in (
        (x, y),
        (x - step, y),
        (x + step, y),
        (x, y - step),
        (x, y + step),
    )
        require_water(solution, xx, yy)
    end
    A(xx, yy) = solution.pressure.A_fn((xx, yy))
    A_i_components(xx, yy) =
        surface_pressure_imaginary_coefficient_components(
            solution.problem, xx, yy,
        )
    A_i_inviscid(xx, yy) = A_i_components(xx, yy).inviscid
    A_i_frictional(xx, yy) = A_i_components(xx, yy).frictional
    B_baroclinic(xx, yy) =
        solution.pressure.baroclinic_B_fn((xx, yy))
    B_stress(xx, yy) = solution.pressure.stress_B_fn((xx, yy))
    A_derivatives = centered_field_derivatives(A, x, y, step)
    A_i_inviscid_derivatives =
        centered_field_derivatives(A_i_inviscid, x, y, step)
    A_i_frictional_derivatives =
        centered_field_derivatives(A_i_frictional, x, y, step)
    B_baroclinic_derivatives = centered_field_derivatives(
        B_baroclinic, x, y, step,
    )
    B_stress_derivatives = centered_field_derivatives(
        B_stress, x, y, step,
    )
    baroclinic_source = -real(B_baroclinic_derivatives.x) -
        imag(B_baroclinic_derivatives.y)
    stress_source =
        -real(B_stress_derivatives.x) - imag(B_stress_derivatives.y)
    return (
        advection_velocity_x=imag(A_derivatives.y),
        advection_velocity_y=-imag(A_derivatives.x),
        inviscid_advection_velocity_x=A_i_inviscid_derivatives.y,
        inviscid_advection_velocity_y=-A_i_inviscid_derivatives.x,
        frictional_advection_velocity_x=A_i_frictional_derivatives.y,
        frictional_advection_velocity_y=-A_i_frictional_derivatives.x,
        diffusivity=real(A(x, y)),
        baroclinic_source,
        stress_source,
        source=baroclinic_source + stress_source,
    )
end

function surface_pressure_imaginary_coefficient_components(problem, x, y)
    H = problem.depth(x, y)
    f = problem.coriolis(x, y)
    f != 0 || throw(ArgumentError("f(x, y) must be nonzero; got $f"))
    parameters = problem.parameters
    α = sqrt(abs(f) / (2 * parameters.viscosity * parameters.Ek))
    s = α * H
    ratio = if s < 0.05
        2s * (1 - 8s^4 / 15 + 992s^8 / 2835)
    elseif s < 20
        (sinh(2s) + sin(2s)) / (cosh(2s) + cos(2s))
    else
        scaled_exponential = exp(-2s)
        (
            1 - scaled_exponential^2 +
            2scaled_exponential * sin(2s)
        ) / (
            1 + scaled_exponential^2 +
            2scaled_exponential * cos(2s)
        )
    end
    inviscid = H / f
    frictional = -ratio / (2α * f)
    return (; inviscid, frictional)
end

struct PGStreamfunction{F,V,G,T}
    finite_element::F
    value::V
    gradient::G
    timings::T
end

"""
    solve_streamfunction(solution; order=2)

Solve for the streamfunction of the depth-integrated horizontal current.
"""
function solve_streamfunction(solution::PGSolution; order=2)
    order > 0 || throw(ArgumentError("order must be positive"))
    result = solve_transport_streamfunction(solution.pressure; order)
    probe = solution.pressure.probe
    value_at = PooledPointEvaluator(result.finite_element, probe)
    gradient_at = PooledPointEvaluator(∇(result.finite_element), probe)
    value = (x, y) -> value_at(Point(x, y))
    gradient = (x, y) -> gradient_at(Point(x, y))
    return PGStreamfunction(
        result.finite_element, value, gradient, result.timings,
    )
end

function streamfunction(solution::PGStreamfunction, x, y)
    return solution.value(x, y)
end

"""
    streamfunction_current(solution, x, y)

Evaluate the discretely nondivergent current represented by the finite-element
streamfunction, using `(U, V) = (-∂yΨ, ∂xΨ)`.
"""
function streamfunction_current(solution::PGStreamfunction, x, y)
    gradient = solution.gradient(x, y)
    return (U=-gradient[2], V=gradient[1])
end

struct PGFields{TX<:AbstractVector,TY<:AbstractVector,TZ<:AbstractVector,T}
    x::TX
    y::TY
    z::TZ
    u::Array{Float64,3}
    v::Array{Float64,3}
    w::Array{Float64,3}
    b::Array{Float64,3}
    surface_pressure::Matrix{Float64}
    depth::Matrix{Float64}
    wet::BitArray{3}
    timings::T
end

PGFields(x, y, z, u, v, w, b, ps, depth, wet) =
    PGFields(x, y, z, u, v, w, b, ps, depth, wet, (;))

struct PGDiagnostics{TX<:AbstractVector,TY<:AbstractVector,T}
    x::TX
    y::TY
    bottom_stress_x::Matrix{Float64}
    bottom_stress_y::Matrix{Float64}
    transport_U::Matrix{Float64}
    transport_V::Matrix{Float64}
    analytical_transport_U::Matrix{Float64}
    analytical_transport_V::Matrix{Float64}
    streamfunction::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

PGDiagnostics(x, y, τx, τy, U, V, Ψ, wet, timings) =
    PGDiagnostics(x, y, τx, τy, U, V, copy(U), copy(V), Ψ, wet, timings)

struct PGDIMDiagnostics{TX<:AbstractVector,TY<:AbstractVector,T}
    x::TX
    y::TY
    chi::Matrix{Float64}
    bottom_pressure::Matrix{Float64}
    x_coriolis::Matrix{Float64}
    x_bottom_pressure::Matrix{Float64}
    x_baroclinic::Matrix{Float64}
    x_bottom_stress::Matrix{Float64}
    x_surface_stress::Matrix{Float64}
    x_residual::Matrix{Float64}
    y_coriolis::Matrix{Float64}
    y_bottom_pressure::Matrix{Float64}
    y_baroclinic::Matrix{Float64}
    y_bottom_stress::Matrix{Float64}
    y_surface_stress::Matrix{Float64}
    y_residual::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

struct PGVorticityDiagnostics{TX<:AbstractVector,TY<:AbstractVector,T}
    x::TX
    y::TY
    planetary_vorticity_advection::Matrix{Float64}
    bottom_vortex_stretching::Matrix{Float64}
    bottom_stress_curl::Matrix{Float64}
    surface_stress_curl::Matrix{Float64}
    residual::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

struct PGBarotropicVorticityDiagnostics{TX<:AbstractVector,TY<:AbstractVector,T}
    x::TX
    y::TY
    transport_divergence::Matrix{Float64}
    planetary_vorticity_advection::Matrix{Float64}
    bottom_pressure_torque::Matrix{Float64}
    bottom_stress_curl::Matrix{Float64}
    surface_stress_curl::Matrix{Float64}
    residual::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

struct PGGeostrophicContourVorticityDiagnostics{
    TX<:AbstractVector,TY<:AbstractVector,T
}
    x::TX
    y::TY
    geostrophic_contour_advection::Matrix{Float64}
    jebar::Matrix{Float64}
    bottom_stress_curl::Matrix{Float64}
    surface_stress_curl::Matrix{Float64}
    residual::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

struct PGTransportDivergenceVorticityDiagnostics{
    TX<:AbstractVector,TY<:AbstractVector,T
}
    x::TX
    y::TY
    bottom_pressure_torque::Matrix{Float64}
    baroclinic_torque::Matrix{Float64}
    bottom_stress_curl::Matrix{Float64}
    surface_stress_curl::Matrix{Float64}
    residual::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

struct PGSurfacePressureEquationDiagnostics{
    TX<:AbstractVector,TY<:AbstractVector,T
}
    x::TX
    y::TY
    advection_velocity_x::Matrix{Float64}
    advection_velocity_y::Matrix{Float64}
    inviscid_advection_velocity_x::Matrix{Float64}
    inviscid_advection_velocity_y::Matrix{Float64}
    frictional_advection_velocity_x::Matrix{Float64}
    frictional_advection_velocity_y::Matrix{Float64}
    diffusivity::Matrix{Float64}
    baroclinic_source::Matrix{Float64}
    stress_source::Matrix{Float64}
    source::Matrix{Float64}
    residual::Matrix{Float64}
    wet::BitMatrix
    timings::T
end

function horizontal_derivative(values, wet, coordinates, i, j, k, axis)
    n = axis == 1 ? size(values, 1) : size(values, 2)
    q = axis == 1 ? i : j
    left = q > 1 && (axis == 1 ? wet[i - 1, j, k] : wet[i, j - 1, k])
    right = q < n && (axis == 1 ? wet[i + 1, j, k] : wet[i, j + 1, k])
    center = values[i, j, k]
    if left && right
        lo = axis == 1 ? values[i - 1, j, k] : values[i, j - 1, k]
        hi = axis == 1 ? values[i + 1, j, k] : values[i, j + 1, k]
        return (hi - lo) / (coordinates[q + 1] - coordinates[q - 1])
    elseif right
        hi = axis == 1 ? values[i + 1, j, k] : values[i, j + 1, k]
        return (hi - center) / (coordinates[q + 1] - coordinates[q])
    elseif left
        lo = axis == 1 ? values[i - 1, j, k] : values[i, j - 1, k]
        return (center - lo) / (coordinates[q] - coordinates[q - 1])
    end
    return 0.0
end

function horizontal_derivative(values::AbstractMatrix, wet, coordinates, i, j, axis)
    n = axis == 1 ? size(values, 1) : size(values, 2)
    q = axis == 1 ? i : j
    left = q > 1 && (axis == 1 ? wet[i - 1, j] : wet[i, j - 1])
    right = q < n && (axis == 1 ? wet[i + 1, j] : wet[i, j + 1])
    center = values[i, j]
    if left && right
        lo = axis == 1 ? values[i - 1, j] : values[i, j - 1]
        hi = axis == 1 ? values[i + 1, j] : values[i, j + 1]
        return (hi - lo) / (coordinates[q + 1] - coordinates[q - 1])
    elseif right
        hi = axis == 1 ? values[i + 1, j] : values[i, j + 1]
        return (hi - center) / (coordinates[q + 1] - coordinates[q])
    elseif left
        lo = axis == 1 ? values[i - 1, j] : values[i, j - 1]
        return (center - lo) / (coordinates[q] - coordinates[q - 1])
    end
    return 0.0
end

function centered_horizontal_derivative(
    values::AbstractMatrix,
    wet,
    coordinates,
    i,
    j,
    axis,
)
    n = axis == 1 ? size(values, 1) : size(values, 2)
    q = axis == 1 ? i : j
    1 < q < n || return NaN
    left = axis == 1 ? wet[i - 1, j] : wet[i, j - 1]
    right = axis == 1 ? wet[i + 1, j] : wet[i, j + 1]
    left && right || return NaN
    lo = axis == 1 ? values[i - 1, j] : values[i, j - 1]
    hi = axis == 1 ? values[i + 1, j] : values[i, j + 1]
    return (hi - lo) / (coordinates[q + 1] - coordinates[q - 1])
end

function centered_stencil_is_wet(wet, i, j)
    return (
        1 < i < size(wet, 1) &&
        1 < j < size(wet, 2) &&
        wet[i, j] &&
        wet[i - 1, j] &&
        wet[i + 1, j] &&
        wet[i, j - 1] &&
        wet[i, j + 1]
    )
end

function reconstruct_vertical_velocity(u, v, wet, depth, x, y, z)
    divergence = fill(NaN, size(u))
    for k in eachindex(z), j in eachindex(y), i in eachindex(x)
        wet[i, j, k] || continue
        divergence[i, j, k] =
            horizontal_derivative(u, wet, x, i, j, k, 1) +
            horizontal_derivative(v, wet, y, i, j, k, 2)
    end

    w = fill(NaN, size(u))
    for j in eachindex(y), i in eachindex(x)
        levels = findall(@view wet[i, j, :])
        isempty(levels) && continue
        bottom, top = first(levels), last(levels)
        w[i, j, bottom] = -0.5 * (z[bottom] + depth[i, j]) * divergence[i, j, bottom]
        for k in (bottom + 1):top
            w[i, j, k] = w[i, j, k - 1] -
                0.5 * (z[k] - z[k - 1]) *
                (divergence[i, j, k - 1] + divergence[i, j, k])
        end
        column_height = z[top] + depth[i, j]
        if column_height > 0
            residual = w[i, j, top]
            for k in bottom:top
                w[i, j, k] -= residual * (z[k] + depth[i, j]) / column_height
            end
        end
    end
    return w
end

"""
    sample_solution(solution, x, y, z)

Sample `u`, `v`, and `b` on an increasing rectilinear grid and reconstruct
`w` from incompressibility. Values outside the water column are `NaN`.
The vertical grid must end at the rigid lid, `z = 0`.
"""
function sample_solution(solution::PGSolution, x_values, y_values, z_values)
    sample_started = time()
    x, y, z = collect(float.(x_values)), collect(float.(y_values)), collect(float.(z_values))
    length(x) >= 2 && length(y) >= 2 && length(z) >= 2 ||
        throw(ArgumentError("x, y, and z must each contain at least two points"))
    issorted(x) && all(diff(x) .> 0) || throw(ArgumentError("x must be strictly increasing"))
    issorted(y) && all(diff(y) .> 0) || throw(ArgumentError("y must be strictly increasing"))
    issorted(z) && all(diff(z) .> 0) || throw(ArgumentError("z must be strictly increasing"))
    iszero(last(z)) || throw(ArgumentError("the vertical grid must end at z = 0"))

    shape = (length(x), length(y), length(z))
    u, v, b = fill(NaN, shape), fill(NaN, shape), fill(NaN, shape)
    ps = fill(NaN, length(x), length(y))
    depth = fill(NaN, length(x), length(y))
    gradps = fill(complex(NaN, NaN), length(x), length(y))
    # BitArray writes are not thread-safe, so fill a Bool array.
    wet_values = fill(false, shape)
    step = solution.problem.derivative_step

    @info "Solution sampling started" Nx=length(x) Ny=length(y) Nz=length(z) grid_points=prod(shape) threads=Threads.nthreads()
    _, pressure_timing = timed_stage("sampling surface pressure and bathymetry") do
        Threads.@threads for j in eachindex(y)
            for i in eachindex(x)
                xx, yy = x[i], y[j]
                solution.problem.inside(xx, yy) || continue
                H = solution.problem.depth(xx, yy)
                H > 0 || continue
                depth[i, j] = H
                ps[i, j] = float(surface_pressure(solution, xx, yy))
                gradps[i, j] = solution.pressure.gradps(xx, yy)
            end
        end
    end

    progress_stride = max(1, cld(length(y), 10))
    rows_completed = Threads.Atomic{Int}(0)
    _, fields_timing = timed_stage("sampling 3D horizontal velocity and buoyancy") do
        Threads.@threads for j in eachindex(y)
            for i in eachindex(x)
                xx, yy = x[i], y[j]
                H = depth[i, j]
                isfinite(H) || continue
                stencil_is_inside = all(solution.problem.inside(xq, yq) for (xq, yq) in (
                    (xx - step, yy), (xx + step, yy), (xx, yy - step), (xx, yy + step),
                ))
                stencil_is_inside || continue
                velocity_column = horizontal_velocity_column(
                    solution, xx, yy, gradps[i, j],
                )
                for k in eachindex(z)
                    -H <= z[k] <= 0 || continue
                    uv = velocity_column(z[k])
                    u[i, j, k], v[i, j, k] = float(real(uv)), float(imag(uv))
                    b[i, j, k] = float(solution.buoyancy_function(xx, yy, z[k]))
                    wet_values[i, j, k] = true
                end
            end
            completed = Threads.atomic_add!(rows_completed, 1) + 1
            if completed == 1 || completed == length(y) || completed % progress_stride == 0
                @info "3D field sampling progress" rows_completed=completed rows_total=length(y) percent=round(100completed / length(y); digits=1)
            end
        end
    end
    wet = BitArray(wet_values)

    w, vertical_velocity_timing =
        timed_stage("reconstructing vertical velocity from incompressibility") do
            reconstruct_vertical_velocity(u, v, wet, depth, x, y, z)
        end
    total_seconds = time() - sample_started
    wet_points = count(wet)
    timings = (
        surface_fields=pressure_timing,
        horizontal_velocity_and_buoyancy=fields_timing,
        vertical_velocity=vertical_velocity_timing,
        total_seconds,
        wet_points,
        wet_points_per_second=wet_points / max(fields_timing.seconds, eps()),
    )
    @info "Solution sampling completed" seconds=round(total_seconds; digits=3) wet_points wet_fraction=round(wet_points / prod(shape); digits=3) field_points_per_second=round(timings.wet_points_per_second; digits=1)
    return PGFields(x, y, z, u, v, w, b, ps, depth, wet, timings)
end

struct PGLagrangianVelocityField{
    TX<:AbstractVector,TY<:AbstractVector,TS<:AbstractVector,
    IU,IV,IW,IS,ID,T,
}
    x::TX
    y::TY
    sigma::TS
    u::Array{Float64,3}
    v::Array{Float64,3}
    w::Array{Float64,3}
    sigma_velocity::Array{Float64,3}
    depth::Matrix{Float64}
    u_interpolant::IU
    v_interpolant::IV
    w_interpolant::IW
    sigma_velocity_interpolant::IS
    depth_interpolant::ID
    timings::T
end

struct PGLagrangianTrajectory
    t::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    sigma::Vector{Float64}
    termination::Symbol
end

struct PGLagrangianTrajectories{F,T}
    field::F
    trajectories::Vector{PGLagrangianTrajectory}
    timings::T
end

function validate_lagrangian_coordinates(x, y, sigma)
    length(x) >= 2 && length(y) >= 2 && length(sigma) >= 2 ||
        throw(ArgumentError("x, y, and sigma must each contain at least two points"))
    issorted(x) && all(diff(x) .> 0) ||
        throw(ArgumentError("x must be strictly increasing"))
    issorted(y) && all(diff(y) .> 0) ||
        throw(ArgumentError("y must be strictly increasing"))
    issorted(sigma) && all(diff(sigma) .> 0) ||
        throw(ArgumentError("sigma must be strictly increasing"))
    iszero(first(sigma)) && last(sigma) == 1 ||
        throw(ArgumentError("sigma must span the complete interval from 0 to 1"))
end

function sampled_depth_derivatives(depth, x, y)
    wet = trues(size(depth))
    depth_x = similar(depth)
    depth_y = similar(depth)
    for j in eachindex(y), i in eachindex(x)
        depth_x[i, j] = horizontal_derivative(depth, wet, x, i, j, 1)
        depth_y[i, j] = horizontal_derivative(depth, wet, y, i, j, 2)
    end
    return depth_x, depth_y
end

function reconstruct_sigma_velocity(u, v, depth, x, y, sigma)
    shape = size(u)
    wet = trues(shape)
    transport_divergence = similar(u)
    column_depth = reshape(depth, size(depth)..., 1)
    depth_integrated_u = column_depth .* u
    depth_integrated_v = column_depth .* v
    for k in eachindex(sigma), j in eachindex(y), i in eachindex(x)
        transport_divergence[i, j, k] =
            horizontal_derivative(
                depth_integrated_u, wet, x, i, j, k, 1,
            ) +
            horizontal_derivative(
                depth_integrated_v, wet, y, i, j, k, 2,
            )
    end

    sigma_transport = zeros(shape)
    for j in eachindex(y), i in eachindex(x)
        for k in 2:length(sigma)
            sigma_transport[i, j, k] =
                sigma_transport[i, j, k - 1] -
                0.5 * (sigma[k] - sigma[k - 1]) * (
                    transport_divergence[i, j, k - 1] +
                    transport_divergence[i, j, k]
                )
        end
        bottom_residual = sigma_transport[i, j, end]
        for k in eachindex(sigma)
            sigma_transport[i, j, k] -= bottom_residual * sigma[k]
        end
    end
    return sigma_transport ./ column_depth
end

function build_lagrangian_velocity_field(
    x,
    y,
    sigma,
    u,
    v,
    sigma_velocity,
    depth;
    timings=(;),
)
    depth_x, depth_y = sampled_depth_derivatives(depth, x, y)
    w = similar(u)
    for k in eachindex(sigma), j in eachindex(y), i in eachindex(x)
        w[i, j, k] =
            -depth[i, j] * sigma_velocity[i, j, k] -
            sigma[k] * (
                u[i, j, k] * depth_x[i, j] +
                v[i, j, k] * depth_y[i, j]
            )
    end

    nodes = (x, y, sigma)
    interpolation_mode = Interpolations.Gridded(Interpolations.Linear())
    u_interpolant = Interpolations.interpolate(nodes, u, interpolation_mode)
    v_interpolant = Interpolations.interpolate(nodes, v, interpolation_mode)
    w_interpolant = Interpolations.interpolate(nodes, w, interpolation_mode)
    sigma_velocity_interpolant = Interpolations.interpolate(
        nodes, sigma_velocity, interpolation_mode,
    )
    depth_interpolant = Interpolations.interpolate(
        (x, y), depth, interpolation_mode,
    )
    return PGLagrangianVelocityField(
        x, y, sigma, u, v, w, sigma_velocity, depth,
        u_interpolant, v_interpolant, w_interpolant,
        sigma_velocity_interpolant, depth_interpolant, timings,
    )
end

"""
    sample_lagrangian_velocity(solution, x, y, sigma)

Sample the steady three-dimensional velocity on the terrain-following
coordinate `sigma = -z/H(x,y)`. The horizontal sampling grid must be a
rectangular all-water patch, and `sigma` must increase from 0 at the rigid lid
to 1 at the seabed. The contravariant vertical velocity is reconstructed from
discrete continuity and corrected to vanish at both material boundaries.
"""
function sample_lagrangian_velocity(
    solution::PGSolution,
    x_values,
    y_values,
    sigma_values,
)
    started = time()
    x = collect(float.(x_values))
    y = collect(float.(y_values))
    sigma = collect(float.(sigma_values))
    validate_lagrangian_coordinates(x, y, sigma)

    depth = Matrix{Float64}(undef, length(x), length(y))
    step = solution.problem.derivative_step
    for j in eachindex(y), i in eachindex(x)
        xx, yy = x[i], y[j]
        stencil = (
            (xx, yy),
            (xx - step, yy),
            (xx + step, yy),
            (xx, yy - step),
            (xx, yy + step),
        )
        all(point -> solution.problem.inside(point...), stencil) ||
            throw(DomainError(
                (xx, yy),
                "the Lagrangian grid and derivative stencil must lie inside the basin",
            ))
        depth[i, j] = solution.problem.depth(xx, yy)
        depth[i, j] > 0 ||
            throw(DomainError((xx, yy), "depth must be positive"))
    end

    shape = (length(x), length(y), length(sigma))
    u = Array{Float64}(undef, shape)
    v = Array{Float64}(undef, shape)
    _, horizontal_timing = timed_stage(
        "sampling terrain-following horizontal velocity",
    ) do
        for j in eachindex(y), i in eachindex(x)
            column = horizontal_velocity_column(solution, x[i], y[j])
            for k in eachindex(sigma)
                velocity = column(-sigma[k] * depth[i, j])
                u[i, j, k] = float(real(velocity))
                v[i, j, k] = float(imag(velocity))
            end
        end
    end

    sigma_velocity, vertical_timing = timed_stage(
        "reconstructing terrain-following vertical velocity",
    ) do
        reconstruct_sigma_velocity(u, v, depth, x, y, sigma)
    end
    timings = (
        horizontal_velocity=horizontal_timing,
        vertical_velocity=vertical_timing,
        total_seconds=time() - started,
    )
    return build_lagrangian_velocity_field(
        x, y, sigma, u, v, sigma_velocity, depth;
        timings,
    )
end

function require_lagrangian_point(field::PGLagrangianVelocityField, x, y, sigma)
    first(field.x) <= x <= last(field.x) ||
        throw(DomainError(x, "x lies outside the Lagrangian velocity grid"))
    first(field.y) <= y <= last(field.y) ||
        throw(DomainError(y, "y lies outside the Lagrangian velocity grid"))
    first(field.sigma) <= sigma <= last(field.sigma) ||
        throw(DomainError(
            sigma, "sigma lies outside the Lagrangian velocity grid",
        ))
end

"""
    lagrangian_velocity(field, x, y, sigma)

Interpolate `(u, v, w, sigma_velocity)` at a terrain-following point.
"""
function lagrangian_velocity(
    field::PGLagrangianVelocityField,
    x,
    y,
    sigma,
)
    require_lagrangian_point(field, x, y, sigma)
    return (
        u=field.u_interpolant(x, y, sigma),
        v=field.v_interpolant(x, y, sigma),
        w=field.w_interpolant(x, y, sigma),
        sigma_velocity=field.sigma_velocity_interpolant(x, y, sigma),
    )
end

function trajectory_callback(condition, reason, termination)
    affect!(integrator) = begin
        termination[] = reason
        terminate!(integrator)
    end
    return ContinuousCallback(
        condition, affect!;
        save_positions=(true, false),
    )
end

function integrate_trajectory(
    field::PGLagrangianVelocityField,
    initial_position,
    tspan;
    saveat,
    reltol,
    abstol,
)
    x0, y0, z0 = float.(initial_position)
    first(field.x) < x0 < last(field.x) ||
        throw(DomainError(x0, "initial x must lie inside the velocity grid"))
    first(field.y) < y0 < last(field.y) ||
        throw(DomainError(y0, "initial y must lie inside the velocity grid"))
    H0 = field.depth_interpolant(x0, y0)
    sigma0 = -z0 / H0
    0 < sigma0 < 1 ||
        throw(DomainError(
            z0, "initial z must lie strictly between the seabed and rigid lid",
        ))

    function velocity!(derivative, state, _, _)
        xx = clamp(state[1], first(field.x), last(field.x))
        yy = clamp(state[2], first(field.y), last(field.y))
        ss = clamp(state[3], first(field.sigma), last(field.sigma))
        derivative[1] = field.u_interpolant(xx, yy, ss)
        derivative[2] = field.v_interpolant(xx, yy, ss)
        derivative[3] = field.sigma_velocity_interpolant(xx, yy, ss)
    end

    termination = Ref(:completed)
    callbacks = CallbackSet(
        trajectory_callback(
            (state, _, _) -> state[1] - first(field.x),
            :x_min,
            termination,
        ),
        trajectory_callback(
            (state, _, _) -> last(field.x) - state[1],
            :x_max,
            termination,
        ),
        trajectory_callback(
            (state, _, _) -> state[2] - first(field.y),
            :y_min,
            termination,
        ),
        trajectory_callback(
            (state, _, _) -> last(field.y) - state[2],
            :y_max,
            termination,
        ),
        trajectory_callback(
            (state, _, _) -> state[3] - first(field.sigma),
            :surface,
            termination,
        ),
        trajectory_callback(
            (state, _, _) -> last(field.sigma) - state[3],
            :seabed,
            termination,
        ),
    )
    problem = ODEProblem(velocity!, [x0, y0, sigma0], float.(tspan))
    result = OrdinaryDiffEqTsit5.solve(
        problem,
        Tsit5();
        callback=callbacks,
        saveat,
        reltol,
        abstol,
    )
    t = collect(float.(result.t))
    x = [float(state[1]) for state in result.u]
    y = [float(state[2]) for state in result.u]
    sigma = [float(state[3]) for state in result.u]
    z = [
        -sigma[index] * field.depth_interpolant(x[index], y[index])
        for index in eachindex(t)
    ]
    return PGLagrangianTrajectory(t, x, y, z, sigma, termination[])
end

"""
    compute_trajectories(field, initial_positions, tspan; ...)

Integrate steady three-dimensional particle trajectories from `(x, y, z)`
initial positions. Times and velocities are non-dimensional. Integration uses
adaptive `Tsit5`; particles stop cleanly if they reach an edge of the sampled
velocity patch, the rigid lid, or the seabed.
"""
function compute_trajectories(
    field::PGLagrangianVelocityField,
    initial_positions,
    tspan;
    saveat=nothing,
    reltol=1e-7,
    abstol=1e-9,
)
    length(tspan) == 2 || throw(ArgumentError("tspan must have two endpoints"))
    tspan[1] != tspan[2] || throw(ArgumentError("tspan endpoints must differ"))
    reltol > 0 || throw(ArgumentError("reltol must be positive"))
    abstol > 0 || throw(ArgumentError("abstol must be positive"))
    isempty(initial_positions) ||
        all(position -> length(position) == 3, initial_positions) ||
        throw(ArgumentError("each initial position must contain (x, y, z)"))
    output_times = isnothing(saveat) ?
        range(float(tspan[1]), float(tspan[2]); length=201) :
        saveat
    started = time()
    trajectories = [
        integrate_trajectory(
            field,
            initial_position,
            tspan;
            saveat=output_times,
            reltol,
            abstol,
        )
        for initial_position in initial_positions
    ]
    timings = (
        total_seconds=time() - started,
        particles=length(trajectories),
    )
    return PGLagrangianTrajectories(field, trajectories, timings)
end

"""
    plot_trajectories(trajectories; output=nothing)

Plot three-dimensional paths above bathymetry, horizontal paths over depth
contours, and particle depth versus time.
"""
function plot_trajectories(
    trajectories::PGLagrangianTrajectories;
    output=nothing,
)
    field = trajectories.field
    colors = palette(:tab10, max(length(trajectories.trajectories), 1))
    show_legend = length(trajectories.trajectories) <= 12
    three_dimensional = surface(
        field.x,
        field.y,
        -field.depth';
        color=:gray,
        alpha=0.3,
        colorbar=false,
        label="seabed",
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="3D Lagrangian trajectories",
    )
    horizontal = contourf(
        field.x,
        field.y,
        field.depth';
        color=:blues,
        colorbar_title="H",
        xlabel="x",
        ylabel="y",
        aspect_ratio=:equal,
        title="Horizontal paths",
    )
    vertical = plot(
        ;
        xlabel="time",
        ylabel="z",
        title="Particle depth",
        legend=show_legend ? :outerright : false,
    )
    for (index, trajectory) in enumerate(trajectories.trajectories)
        color = colors[index]
        label = show_legend ? "particle $index" : false
        plot!(
            three_dimensional,
            trajectory.x,
            trajectory.y,
            trajectory.z;
            color,
            linewidth=2,
            label,
        )
        scatter!(
            three_dimensional,
            [first(trajectory.x)],
            [first(trajectory.y)],
            [first(trajectory.z)];
            color,
            markersize=4,
            label=false,
        )
        plot!(
            horizontal,
            trajectory.x,
            trajectory.y;
            color,
            linewidth=2,
            label=false,
        )
        scatter!(
            horizontal,
            [first(trajectory.x)],
            [first(trajectory.y)];
            color,
            markersize=4,
            label=false,
        )
        plot!(
            vertical,
            trajectory.t,
            trajectory.z;
            color,
            linewidth=2,
            label,
        )
    end
    figure = plot(
        three_dimensional,
        horizontal,
        vertical;
        layout=(1, 3),
        size=(1500, 450),
    )
    if !isnothing(output)
        savefig(figure, output)
        @info "Lagrangian trajectory plot saved" path=string(output)
    end
    return figure
end

function trajectory_frame(
    trajectories::PGLagrangianTrajectories,
    time,
    colors,
)
    field = trajectories.field
    three_dimensional = surface(
        field.x,
        field.y,
        -field.depth';
        color=:gray,
        alpha=0.3,
        colorbar=false,
        label=false,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="3D trajectories, t=$(round(time; digits=1))",
    )
    horizontal = contourf(
        field.x,
        field.y,
        field.depth';
        color=:blues,
        colorbar_title="H",
        xlabel="x",
        ylabel="y",
        aspect_ratio=:equal,
        title="Horizontal paths",
    )
    for (index, trajectory) in enumerate(trajectories.trajectories)
        point_index = searchsortedlast(trajectory.t, time)
        point_index == 0 && continue
        path = 1:point_index
        color = colors[index]
        plot!(
            three_dimensional,
            trajectory.x[path],
            trajectory.y[path],
            trajectory.z[path];
            color,
            linewidth=1.5,
            label=false,
        )
        scatter!(
            three_dimensional,
            [trajectory.x[point_index]],
            [trajectory.y[point_index]],
            [trajectory.z[point_index]];
            color,
            markersize=3,
            label=false,
        )
        plot!(
            horizontal,
            trajectory.x[path],
            trajectory.y[path];
            color,
            linewidth=1.5,
            label=false,
        )
        scatter!(
            horizontal,
            [trajectory.x[point_index]],
            [trajectory.y[point_index]];
            color,
            markersize=3,
            label=false,
        )
    end
    return plot(
        three_dimensional,
        horizontal;
        layout=(1, 2),
        size=(1000, 450),
    )
end

"""
    animate_trajectories(
        trajectories;
        output="trajectories.gif",
        frame_stride=1,
        fps=20,
    )

Animate accumulated three-dimensional and horizontal particle paths. Frames
follow the longest trajectory's saved times, retaining every `frame_stride`th
sample, and the animation is written to `output`.
"""
function animate_trajectories(
    trajectories::PGLagrangianTrajectories;
    output="trajectories.gif",
    frame_stride=1,
    fps=20,
)
    isempty(trajectories.trajectories) &&
        throw(ArgumentError("at least one trajectory is required"))
    frame_stride > 0 || throw(ArgumentError("frame_stride must be positive"))
    fps > 0 || throw(ArgumentError("fps must be positive"))
    all(trajectory -> issorted(trajectory.t), trajectories.trajectories) ||
        throw(ArgumentError("animation requires forward trajectories"))

    reference = trajectories.trajectories[
        argmax(length(trajectory.t) for trajectory in trajectories.trajectories)
    ]
    frame_indices = collect(1:frame_stride:length(reference.t))
    last(frame_indices) == length(reference.t) ||
        push!(frame_indices, length(reference.t))
    frame_times = reference.t[frame_indices]
    colors = palette(:tab20, length(trajectories.trajectories))
    animation = Animation()
    for time in frame_times
        frame(animation, trajectory_frame(trajectories, time, colors))
    end
    gif(animation, output; fps)
    @info "Lagrangian trajectory animation saved" path=string(output) frames=length(frame_times) fps
    return string(output)
end

"""
    compute_diagnostics(
        solution, x, y;
        streamfunction_order=2,
        bottom_stress_projection_order=pressure order,
    )

Sample bottom stress and depth-integrated horizontal current, and solve and
sample its streamfunction, on an increasing horizontal rectilinear grid.
Bottom stress is interpolated into a continuous `H1` finite-element space before
sampling so subsequent curl diagnostics do not differentiate jumps in the
raw pressure gradient.
"""
function compute_diagnostics(
    solution::PGSolution,
    x_values,
    y_values;
    streamfunction_order=2,
    bottom_stress_projection_order=solution.pressure.order,
)
    diagnostic_started = time()
    x, y = collect(float.(x_values)), collect(float.(y_values))
    length(x) >= 2 && length(y) >= 2 ||
        throw(ArgumentError("x and y must each contain at least two points"))
    issorted(x) && all(diff(x) .> 0) ||
        throw(ArgumentError("x must be strictly increasing"))
    issorted(y) && all(diff(y) .> 0) ||
        throw(ArgumentError("y must be strictly increasing"))
    @info "PG diagnostic calculation started" Nx=length(x) Ny=length(y) horizontal_points=length(x) * length(y)
    ψ = solve_streamfunction(solution; order=streamfunction_order)
    projected_stress = project_bottom_stress(
        solution; order=bottom_stress_projection_order,
    )

    shape = (length(x), length(y))
    τx, τy = fill(NaN, shape), fill(NaN, shape)
    U, V = fill(NaN, shape), fill(NaN, shape)
    analytical_U, analytical_V = fill(NaN, shape), fill(NaN, shape)
    Ψ = fill(NaN, shape)
    wet_values = fill(false, shape)
    step = solution.problem.derivative_step
    progress_stride = max(1, cld(length(y), 10))
    rows_completed = Threads.Atomic{Int}(0)
    _, sampling_timing = timed_stage(
        "sampling bottom stress, transport, and streamfunction",
    ) do
        Threads.@threads for j in eachindex(y)
            for i in eachindex(x)
                xx, yy = x[i], y[j]
                solution.problem.inside(xx, yy) || continue
                solution.problem.depth(xx, yy) > 0 || continue
                stencil_is_inside = all(
                    solution.problem.inside(xq, yq) for (xq, yq) in (
                        (xx - step, yy), (xx + step, yy),
                        (xx, yy - step), (xx, yy + step),
                    )
                )
                stencil_is_inside || continue
                stress_vector = projected_stress.value(xx, yy)
                analytical_current = depth_integrated_current(solution, xx, yy)
                current = streamfunction_current(ψ, xx, yy)
                τx[i, j], τy[i, j] =
                    float(stress_vector[1]), float(stress_vector[2])
                U[i, j], V[i, j] = float(current.U), float(current.V)
                analytical_U[i, j] = float(analytical_current.U)
                analytical_V[i, j] = float(analytical_current.V)
                Ψ[i, j] = float(streamfunction(ψ, xx, yy))
                wet_values[i, j] = true
            end

            completed = Threads.atomic_add!(rows_completed, 1) + 1
            if completed == 1 || completed == length(y) || completed % progress_stride == 0
                @info "PG diagnostic sampling progress" rows_completed=completed rows_total=length(y) percent=round(100completed / length(y); digits=1)
            end
        end
    end
    wet = BitMatrix(wet_values)

    total_seconds = time() - diagnostic_started
    wet_points = count(wet)
    transport_projection_error = begin
        differences = hypot.(
            U[wet] .- analytical_U[wet],
            V[wet] .- analytical_V[wet],
        )
        analytical_magnitudes = hypot.(analytical_U[wet], analytical_V[wet])
        sqrt(sum(abs2, differences)) /
        max(sqrt(sum(abs2, analytical_magnitudes)), eps())
    end
    timings = (
        streamfunction=ψ.timings,
        bottom_stress_projection=projected_stress.timings,
        sampling=sampling_timing,
        total_seconds,
        wet_points,
        points_per_second=wet_points / max(sampling_timing.seconds, eps()),
        transport_projection_relative_l2=transport_projection_error,
    )
    @info "PG diagnostic calculation completed" seconds=round(total_seconds; digits=3) wet_points points_per_second=round(timings.points_per_second; digits=1) transport_projection_relative_l2=round(transport_projection_error; digits=4)
    return PGDiagnostics(
        x, y, τx, τy, U, V, analytical_U, analytical_V, Ψ, wet, timings,
    )
end

"""
    compute_dim_diagnostics(solution, diagnostics)

Compute maps of all signed terms in the depth-integrated momentum equations
corresponding to (32)--(33), together with bottom pressure, baroclinic
potential energy `χ`, and componentwise closure residuals.
"""
function compute_dim_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics,
)
    started = time()
    x, y = diagnostics.x, diagnostics.y
    shape = (length(x), length(y))
    arrays = [fill(NaN, shape) for _ in 1:14]
    χ, pb = arrays[1], arrays[2]
    x_coriolis, x_pb, x_χ, x_τb, x_τs, x_residual = arrays[3:8]
    y_coriolis, y_pb, y_χ, y_τb, y_τs, y_residual = arrays[9:14]
    wet = copy(diagnostics.wet)
    progress_stride = max(1, cld(length(y), 10))

    @info "DIM diagnostic calculation started" Nx=length(x) Ny=length(y) wet_points=count(wet)
    rows_completed = Threads.Atomic{Int}(0)
    _, sampling_timing = timed_stage(
        "sampling depth-integrated momentum terms",
    ) do
        Threads.@threads for j in eachindex(y)
            for i in eachindex(x)
                wet[i, j] || continue
                xx, yy = x[i], y[j]
                χ[i, j] = float(baroclinic_potential_energy(solution, xx, yy))
                pb[i, j] = float(bottom_pressure(solution, xx, yy))
                terms = depth_integrated_momentum_terms(solution, xx, yy)
                x_coriolis[i, j] = terms.x.coriolis
                x_pb[i, j] = terms.x.bottom_pressure
                x_χ[i, j] = terms.x.baroclinic
                x_τb[i, j] = terms.x.bottom_stress
                x_τs[i, j] = terms.x.surface_stress
                x_residual[i, j] = terms.x_residual
                y_coriolis[i, j] = terms.y.coriolis
                y_pb[i, j] = terms.y.bottom_pressure
                y_χ[i, j] = terms.y.baroclinic
                y_τb[i, j] = terms.y.bottom_stress
                y_τs[i, j] = terms.y.surface_stress
                y_residual[i, j] = terms.y_residual
            end
            completed = Threads.atomic_add!(rows_completed, 1) + 1
            if completed == 1 || completed == length(y) || completed % progress_stride == 0
                @info "DIM diagnostic sampling progress" rows_completed=completed rows_total=length(y) percent=round(100completed / length(y); digits=1)
            end
        end
    end

    finite_l2(values) = sqrt(sum(abs2, values[wet]))
    x_scale = max(finite_l2(x_coriolis), eps())
    y_scale = max(finite_l2(y_coriolis), eps())
    total_seconds = time() - started
    timings = (
        sampling=sampling_timing,
        total_seconds,
        x_relative_l2_residual=finite_l2(x_residual) / x_scale,
        y_relative_l2_residual=finite_l2(y_residual) / y_scale,
    )
    @info "DIM diagnostic calculation completed" seconds=round(total_seconds; digits=3) x_relative_l2_residual=round(timings.x_relative_l2_residual; digits=6) y_relative_l2_residual=round(timings.y_relative_l2_residual; digits=6)
    return PGDIMDiagnostics(
        x, y, χ, pb,
        x_coriolis, x_pb, x_χ, x_τb, x_τs, x_residual,
        y_coriolis, y_pb, y_χ, y_τb, y_τs, y_residual,
        wet, timings,
    )
end

function compute_dim_diagnostics(solution::PGSolution, x, y; kwargs...)
    return compute_dim_diagnostics(
        solution, compute_diagnostics(solution, x, y; kwargs...),
    )
end

"""
    centered_barotropic_vorticity_diagnostics(solution, diagnostics)

Centered-difference form of `compute_barotropic_vorticity_diagnostics(...; method=:centered)`.

Compute maps of all signed terms in the barotropic vorticity equation (36),
together with its closure residual. Bottom stress is sampled from its
continuous finite-element interpolant, surface-stress curl is differentiated
analytically, and remaining derivatives use centered diagnostic-grid
differences. Edge points and points adjoining land are omitted because a
centered stencil is unavailable. The transport-divergence term is set to its
exact fixed-lid value of zero.
"""
function centered_barotropic_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics,
)
    started = time()
    x, y = diagnostics.x, diagnostics.y
    shape = (length(x), length(y))
    arrays = [fill(NaN, shape) for _ in 1:6]
    divergence, planetary, pressure_torque =
        arrays[1], arrays[2], arrays[3]
    bottom_curl, surface_curl, residual =
        arrays[4], arrays[5], arrays[6]
    bottom_pressure_values = fill(NaN, shape)
    wet = copy(diagnostics.wet)
    budget_wet = falses(shape)
    progress_stride = max(1, cld(length(y), 10))

    @info "BV diagnostic calculation started" Nx=length(x) Ny=length(y) wet_points=count(wet)
    _, sampling_timing = timed_stage(
        "sampling barotropic vorticity terms",
    ) do
        for j in eachindex(y)
            for i in eachindex(x)
                wet[i, j] || continue
                xx, yy = x[i], y[j]
                bottom_pressure_values[i, j] =
                    bottom_pressure(solution, xx, yy)
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "BV diagnostic sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1)
            end
        end

        Ek = solution.problem.parameters.Ek
        for j in eachindex(y), i in eachindex(x)
            centered_stencil_is_wet(wet, i, j) || continue
            xx, yy = x[i], y[j]
            divergence[i, j] = 0.0
            planetary[i, j] = (
                diagnostics.analytical_transport_U[i, j] *
                solution.problem.coriolis_x(xx, yy) +
                diagnostics.analytical_transport_V[i, j] *
                solution.problem.coriolis_y(xx, yy)
            )
            pressure_x = centered_horizontal_derivative(
                bottom_pressure_values, wet, x, i, j, 1,
            )
            pressure_y = centered_horizontal_derivative(
                bottom_pressure_values, wet, y, i, j, 2,
            )
            pressure_torque[i, j] = (
                pressure_x * solution.problem.depth_y(xx, yy) -
                pressure_y * solution.problem.depth_x(xx, yy)
            )
            bottom_curl[i, j] = Ek * (
                centered_horizontal_derivative(
                    diagnostics.bottom_stress_y, wet, x, i, j, 1,
                ) -
                centered_horizontal_derivative(
                    diagnostics.bottom_stress_x, wet, y, i, j, 2,
                )
            )
            surface_curl[i, j] = Ek * analytical_surface_stress_curl(
                solution.problem, xx, yy,
            )
            residual[i, j] = divergence[i, j] + planetary[i, j] -
                pressure_torque[i, j] - bottom_curl[i, j] -
                surface_curl[i, j]
            budget_wet[i, j] = true
        end
    end

    budget_points = count(budget_wet)
    budget_points > 0 ||
        @warn "BV diagnostic grid has no points with a centered wet stencil"
    finite_l2(values) = sqrt(sum(abs2, values[budget_wet]))
    balance_scale = max(
        finite_l2(divergence),
        finite_l2(planetary),
        finite_l2(pressure_torque),
        finite_l2(bottom_curl),
        finite_l2(surface_curl),
        eps(),
    )
    total_seconds = time() - started
    timings = (
        sampling=sampling_timing,
        total_seconds,
        relative_l2_residual=budget_points > 0 ?
            finite_l2(residual) / balance_scale : NaN,
    )
    @info "BV diagnostic calculation completed" seconds=round(total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; digits=6)
    return PGBarotropicVorticityDiagnostics(
        x, y, divergence, planetary, pressure_torque, bottom_curl,
        surface_curl, residual, budget_wet, timings,
    )
end

function compute_barotropic_vorticity_diagnostics(
    solution::PGSolution,
    x,
    y;
    method=:cell_average,
    quadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
    kwargs...,
)
    return compute_barotropic_vorticity_diagnostics(
        solution, compute_diagnostics(solution, x, y; kwargs...);
        method, quadrature,
    )
end

"""
    centered_vorticity_diagnostics(solution, diagnostics)

Centered-difference form of `compute_vorticity_diagnostics(...; method=:centered)`.

Compute maps of all signed terms in the depth-integrated vorticity equation
(41), together with its closure residual. Bottom stress is sampled from its
continuous finite-element interpolant, and bottom pressure is sampled on the
diagnostic grid. Their centered derivatives include the chain-rule correction
`f w_b + J(p_b,H)` that converts the curl following a sloping seabed into the
fixed-`z` bottom vorticity flux required by (41). Surface-stress curl is
differentiated analytically. Edge points and points adjoining land are omitted
because a centered stencil is unavailable.
"""
function centered_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics,
)
    started = time()
    x, y = diagnostics.x, diagnostics.y
    shape = (length(x), length(y))
    arrays = [fill(NaN, shape) for _ in 1:5]
    planetary, stretching, bottom_curl, surface_curl, residual = arrays
    terrain_following_curl = fill(NaN, shape)
    pressure_torque = fill(NaN, shape)
    bottom_pressure_values = fill(NaN, shape)
    wet = copy(diagnostics.wet)
    budget_wet = falses(shape)
    progress_stride = max(1, cld(length(y), 10))

    @info "Vorticity diagnostic calculation started" Nx=length(x) Ny=length(y) wet_points=count(wet)
    _, sampling_timing = timed_stage(
        "sampling depth-integrated vorticity terms",
    ) do
        for j in eachindex(y)
            for i in eachindex(x)
                wet[i, j] || continue
                xx, yy = x[i], y[j]
                H = solution.problem.depth(xx, yy)
                bottom_velocity = horizontal_velocity(
                    solution, xx, yy, -H,
                )
                bottom_vertical_velocity = -(
                    bottom_velocity.u * solution.problem.depth_x(xx, yy) +
                    bottom_velocity.v * solution.problem.depth_y(xx, yy)
                )
                planetary[i, j] = (
                    diagnostics.analytical_transport_U[i, j] *
                    solution.problem.coriolis_x(xx, yy) +
                    diagnostics.analytical_transport_V[i, j] *
                    solution.problem.coriolis_y(xx, yy)
                )
                stretching[i, j] =
                    solution.problem.coriolis(xx, yy) *
                    bottom_vertical_velocity
                bottom_pressure_values[i, j] =
                    bottom_pressure(solution, xx, yy)
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "Vorticity diagnostic sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1)
            end
        end

        Ek = solution.problem.parameters.Ek
        for j in eachindex(y), i in eachindex(x)
            centered_stencil_is_wet(wet, i, j) || continue
            terrain_following_bottom_curl = Ek * (
                centered_horizontal_derivative(
                    diagnostics.bottom_stress_y, wet, x, i, j, 1,
                ) -
                centered_horizontal_derivative(
                    diagnostics.bottom_stress_x, wet, y, i, j, 2,
                )
            )
            bottom_pressure_x = centered_horizontal_derivative(
                bottom_pressure_values, wet, x, i, j, 1,
            )
            bottom_pressure_y = centered_horizontal_derivative(
                bottom_pressure_values, wet, y, i, j, 2,
            )
            bottom_pressure_torque = (
                bottom_pressure_x *
                solution.problem.depth_y(x[i], y[j]) -
                bottom_pressure_y *
                solution.problem.depth_x(x[i], y[j])
            )
            terrain_following_curl[i, j] = terrain_following_bottom_curl
            pressure_torque[i, j] = bottom_pressure_torque
            bottom_curl[i, j] = terrain_following_bottom_curl +
                stretching[i, j] + bottom_pressure_torque
            surface_curl[i, j] = Ek * analytical_surface_stress_curl(
                solution.problem, x[i], y[j],
            )
            residual[i, j] = planetary[i, j] + stretching[i, j] -
                bottom_curl[i, j] - surface_curl[i, j]
            budget_wet[i, j] = true
        end
        planetary[.!budget_wet] .= NaN
        stretching[.!budget_wet] .= NaN
    end

    budget_points = count(budget_wet)
    budget_points > 0 ||
        @warn "Vorticity diagnostic grid has no points with a centered wet stencil"
    finite_l2(values) = sqrt(sum(abs2, values[budget_wet]))
    # `bottom_curl` already bundles the chain-rule correction (eq 41 docstring):
    # terrain-following stress curl, bottom vortex stretching, and bottom
    # pressure torque. The first and third of those largely cancel in the
    # true solution (compare the barotropic-vorticity balance, eq 36, which
    # keeps them as separate terms), so normalizing by the combined,
    # already-cancelled `bottom_curl` array understates the balance scale and
    # inflates the reported relative residual. Use the uncombined
    # constituent terms instead, matching the convention used by the
    # barotropic- and geostrophic-contour-vorticity diagnostics.
    balance_scale = max(
        finite_l2(planetary),
        finite_l2(stretching),
        finite_l2(terrain_following_curl),
        finite_l2(pressure_torque),
        finite_l2(surface_curl),
        eps(),
    )
    total_seconds = time() - started
    timings = (
        sampling=sampling_timing,
        total_seconds,
        relative_l2_residual=budget_points > 0 ?
            finite_l2(residual) / balance_scale : NaN,
    )
    @info "Vorticity diagnostic calculation completed" seconds=round(total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; digits=6)
    return PGVorticityDiagnostics(
        x, y, planetary, stretching, bottom_curl, surface_curl,
        residual, budget_wet, timings,
    )
end

function compute_vorticity_diagnostics(
    solution::PGSolution,
    x,
    y;
    method=:cell_average,
    quadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
    kwargs...,
)
    return compute_vorticity_diagnostics(
        solution, compute_diagnostics(solution, x, y; kwargs...);
        method, quadrature,
    )
end

"""
    centered_geostrophic_contour_vorticity_diagnostics(solution, diagnostics)

Centered-difference form of `compute_geostrophic_contour_vorticity_diagnostics(...; method=:centered)`.

Compute maps of all signed terms in the geostrophic-contour vorticity balance
(43), together with its closure residual. Gradients of `f/H` and `1/H` use
the specified analytical derivatives of `f` and `H`. The JEBAR and
bottom-stress derivatives use centered diagnostic-grid differences, with
bottom stress sampled from its continuous finite-element interpolant. The
surface-stress curl divided by depth is evaluated analytically. Edge points
and points adjoining land are omitted because a centered stencil is
unavailable. As in the mapped BV and DIV diagnostics, the balance uses the
direct analytical transport; in the continuum this is equivalently
`(U,V)=(-Ψ_y,Ψ_x)`.
"""
function centered_geostrophic_contour_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics,
)
    started = time()
    x, y = diagnostics.x, diagnostics.y
    shape = (length(x), length(y))
    arrays = [fill(NaN, shape) for _ in 1:5]
    contour_advection, jebar, bottom_curl, surface_curl, residual = arrays
    χ = fill(NaN, shape)
    bottom_stress_x_over_depth = fill(NaN, shape)
    bottom_stress_y_over_depth = fill(NaN, shape)
    wet = copy(diagnostics.wet)
    budget_wet = falses(shape)
    progress_stride = max(1, cld(length(y), 10))

    @info "GCVB diagnostic calculation started" Nx=length(x) Ny=length(y) wet_points=count(wet)
    _, sampling_timing = timed_stage(
        "sampling geostrophic-contour vorticity terms",
    ) do
        for j in eachindex(y)
            for i in eachindex(x)
                wet[i, j] || continue
                xx, yy = x[i], y[j]
                H = solution.problem.depth(xx, yy)
                χ[i, j] = baroclinic_potential_energy(solution, xx, yy)
                bottom_stress_x_over_depth[i, j] =
                    diagnostics.bottom_stress_x[i, j] / H
                bottom_stress_y_over_depth[i, j] =
                    diagnostics.bottom_stress_y[i, j] / H
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "GCVB diagnostic sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1)
            end
        end

        problem = solution.problem
        Ek = problem.parameters.Ek
        for j in eachindex(y), i in eachindex(x)
            centered_stencil_is_wet(wet, i, j) || continue
            xx, yy = x[i], y[j]
            H = problem.depth(xx, yy)
            H_x, H_y = problem.depth_x(xx, yy), problem.depth_y(xx, yy)
            f = problem.coriolis(xx, yy)
            f_over_H_x = (problem.coriolis_x(xx, yy) * H - f * H_x) / H^2
            f_over_H_y = (problem.coriolis_y(xx, yy) * H - f * H_y) / H^2
            contour_advection[i, j] = -(
                f_over_H_x * diagnostics.analytical_transport_U[i, j] +
                f_over_H_y * diagnostics.analytical_transport_V[i, j]
            )

            χ_x = centered_horizontal_derivative(χ, wet, x, i, j, 1)
            χ_y = centered_horizontal_derivative(χ, wet, y, i, j, 2)
            jebar[i, j] = (H_y * χ_x - H_x * χ_y) / H^2
            bottom_curl[i, j] = Ek * (
                centered_horizontal_derivative(
                    bottom_stress_y_over_depth, wet, x, i, j, 1,
                ) -
                centered_horizontal_derivative(
                    bottom_stress_x_over_depth, wet, y, i, j, 2,
                )
            )
            surface_curl[i, j] =
                Ek * analytical_surface_stress_over_depth_curl(
                    problem, xx, yy,
                )
            residual[i, j] = contour_advection[i, j] - jebar[i, j] +
                bottom_curl[i, j] + surface_curl[i, j]
            budget_wet[i, j] = true
        end
    end

    budget_points = count(budget_wet)
    budget_points > 0 ||
        @warn "GCVB diagnostic grid has no points with a centered wet stencil"
    finite_l2(values) = sqrt(sum(abs2, values[budget_wet]))
    balance_scale = max(
        finite_l2(contour_advection),
        finite_l2(jebar),
        finite_l2(bottom_curl),
        finite_l2(surface_curl),
        eps(),
    )
    total_seconds = time() - started
    timings = (
        sampling=sampling_timing,
        total_seconds,
        relative_l2_residual=budget_points > 0 ?
            finite_l2(residual) / balance_scale : NaN,
    )
    @info "GCVB diagnostic calculation completed" seconds=round(total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; digits=6)
    return PGGeostrophicContourVorticityDiagnostics(
        x, y, contour_advection, jebar, bottom_curl, surface_curl,
        residual, budget_wet, timings,
    )
end

function compute_geostrophic_contour_vorticity_diagnostics(
    solution::PGSolution,
    x,
    y;
    method=:cell_average,
    quadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
    kwargs...,
)
    return compute_geostrophic_contour_vorticity_diagnostics(
        solution, compute_diagnostics(solution, x, y; kwargs...);
        method, quadrature,
    )
end

"""
    centered_transport_divergence_vorticity_diagnostics(solution, diagnostics)

Centered-difference form of `compute_transport_divergence_vorticity_diagnostics(...; method=:centered)`.

Compute maps of all signed terms in the transport-divergence vorticity balance
(47), together with its closure residual. Gradients of `H/f` and `1/f` use
the specified analytical derivatives of `H` and `f`. Bottom-pressure,
baroclinic-potential-energy, and bottom-stress derivatives use centered
diagnostic-grid differences, with bottom stress sampled from its continuous
finite-element interpolant. The curl of the surface stress divided by `f`,
`curl(τ/f)`, is evaluated analytically. Edge points and points adjoining land
are omitted because a centered stencil is unavailable.
"""
function centered_transport_divergence_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics,
)
    started = time()
    x, y = diagnostics.x, diagnostics.y
    shape = (length(x), length(y))
    arrays = [fill(NaN, shape) for _ in 1:5]
    pressure_torque, baroclinic_torque, bottom_curl, surface_curl, residual =
        arrays
    bottom_pressure_values = fill(NaN, shape)
    χ = fill(NaN, shape)
    bottom_stress_x_over_coriolis = fill(NaN, shape)
    bottom_stress_y_over_coriolis = fill(NaN, shape)
    wet = copy(diagnostics.wet)
    budget_wet = falses(shape)
    progress_stride = max(1, cld(length(y), 10))

    @info "TDVB diagnostic calculation started" Nx=length(x) Ny=length(y) wet_points=count(wet)
    _, sampling_timing = timed_stage(
        "sampling transport-divergence vorticity terms",
    ) do
        for j in eachindex(y)
            for i in eachindex(x)
                wet[i, j] || continue
                xx, yy = x[i], y[j]
                f = solution.problem.coriolis(xx, yy)
                bottom_pressure_values[i, j] =
                    bottom_pressure(solution, xx, yy)
                χ[i, j] = baroclinic_potential_energy(solution, xx, yy)
                bottom_stress_x_over_coriolis[i, j] =
                    diagnostics.bottom_stress_x[i, j] / f
                bottom_stress_y_over_coriolis[i, j] =
                    diagnostics.bottom_stress_y[i, j] / f
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "TDVB diagnostic sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1)
            end
        end

        problem = solution.problem
        Ek = problem.parameters.Ek
        for j in eachindex(y), i in eachindex(x)
            centered_stencil_is_wet(wet, i, j) || continue
            xx, yy = x[i], y[j]
            H = problem.depth(xx, yy)
            f = problem.coriolis(xx, yy)
            H_x, H_y = problem.depth_x(xx, yy), problem.depth_y(xx, yy)
            f_x, f_y = problem.coriolis_x(xx, yy), problem.coriolis_y(xx, yy)

            pb_x = centered_horizontal_derivative(
                bottom_pressure_values, wet, x, i, j, 1,
            )
            pb_y = centered_horizontal_derivative(
                bottom_pressure_values, wet, y, i, j, 2,
            )
            H_over_f_x = (H_x * f - H * f_x) / f^2
            H_over_f_y = (H_y * f - H * f_y) / f^2
            pressure_torque[i, j] =
                pb_x * H_over_f_y - pb_y * H_over_f_x

            χ_x = centered_horizontal_derivative(χ, wet, x, i, j, 1)
            χ_y = centered_horizontal_derivative(χ, wet, y, i, j, 2)
            baroclinic_torque[i, j] = (-f_y * χ_x + f_x * χ_y) / f^2
            bottom_curl[i, j] = Ek * (
                centered_horizontal_derivative(
                    bottom_stress_y_over_coriolis, wet, x, i, j, 1,
                ) -
                centered_horizontal_derivative(
                    bottom_stress_x_over_coriolis, wet, y, i, j, 2,
                )
            )
            surface_curl[i, j] =
                Ek * analytical_surface_stress_over_coriolis_curl(
                    problem, xx, yy,
                )
            residual[i, j] = -(
                pressure_torque[i, j] + baroclinic_torque[i, j] +
                bottom_curl[i, j] + surface_curl[i, j]
            )
            budget_wet[i, j] = true
        end
    end

    budget_points = count(budget_wet)
    budget_points > 0 ||
        @warn "TDVB diagnostic grid has no points with a centered wet stencil"
    finite_l2(values) = sqrt(sum(abs2, values[budget_wet]))
    balance_scale = max(
        finite_l2(pressure_torque),
        finite_l2(baroclinic_torque),
        finite_l2(bottom_curl),
        finite_l2(surface_curl),
        eps(),
    )
    total_seconds = time() - started
    timings = (
        sampling=sampling_timing,
        total_seconds,
        relative_l2_residual=budget_points > 0 ?
            finite_l2(residual) / balance_scale : NaN,
    )
    @info "TDVB diagnostic calculation completed" seconds=round(total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; digits=6)
    return PGTransportDivergenceVorticityDiagnostics(
        x, y, pressure_torque, baroclinic_torque, bottom_curl,
        surface_curl, residual, budget_wet, timings,
    )
end

function compute_transport_divergence_vorticity_diagnostics(
    solution::PGSolution,
    x,
    y;
    method=:cell_average,
    quadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
    kwargs...,
)
    return compute_transport_divergence_vorticity_diagnostics(
        solution, compute_diagnostics(solution, x, y; kwargs...);
        method, quadrature,
    )
end

"""
    compute_surface_pressure_equation_diagnostics(solution, diagnostics)

Sample the velocity, diffusivity, and source fields in the surface-pressure
advection-diffusion equation (75). The exact analytical transport coefficients
`A` and `B` used by the pressure solve are recovered from the already sampled
analytical transport and pressure gradient; their derivatives are then
evaluated with centered diagnostic-grid differences. The total advection
velocity is accompanied by its inviscid `A_i = H/f` and frictional
components. The residual is the sampled transport divergence, equivalent to
the left-hand side minus source in (75). Edge points and points adjoining land
are omitted because a centered stencil is unavailable.
"""
function compute_surface_pressure_equation_diagnostics(
    solution::PGSolution,
    diagnostics,
)
    started = time()
    x, y = diagnostics.x, diagnostics.y
    shape = (length(x), length(y))
    arrays = [fill(NaN, shape) for _ in 1:11]
    velocity_x, velocity_y, inviscid_velocity_x, inviscid_velocity_y,
        frictional_velocity_x, frictional_velocity_y, diffusivity,
        baroclinic_source, stress_source, source, residual = arrays
    transport_x_divergence = fill(NaN, shape)
    transport_y_divergence = fill(NaN, shape)
    real_A, imag_A = fill(NaN, shape), fill(NaN, shape)
    inviscid_A_i, frictional_A_i =
        fill(NaN, shape), fill(NaN, shape)
    real_baroclinic_B, imag_baroclinic_B =
        fill(NaN, shape), fill(NaN, shape)
    real_stress_B, imag_stress_B = fill(NaN, shape), fill(NaN, shape)
    sampled_wet = copy(diagnostics.wet)
    wet = falses(shape)
    progress_stride = max(1, cld(length(y), 10))

    @info "Surface-pressure equation diagnostic calculation started" Nx=length(x) Ny=length(y) wet_points=count(sampled_wet)
    _, sampling_timing = timed_stage(
        "sampling surface-pressure equation coefficients",
    ) do
        for j in eachindex(y)
            for i in eachindex(x)
                sampled_wet[i, j] || continue
                point = (x[i], y[j])
                A = solution.pressure.A_fn(point)
                current = (
                    diagnostics.analytical_transport_U[i, j] +
                    im * diagnostics.analytical_transport_V[i, j]
                )
                B = current - A * solution.pressure.gradps(x[i], y[j])
                stress_B = solution.pressure.stress_B_fn(point)
                baroclinic_B = B - stress_B
                real_A[i, j], imag_A[i, j] = real(A), imag(A)
                A_i_components =
                    surface_pressure_imaginary_coefficient_components(
                        solution.problem, x[i], y[j],
                    )
                inviscid_A_i[i, j] = A_i_components.inviscid
                frictional_A_i[i, j] = A_i_components.frictional
                real_baroclinic_B[i, j], imag_baroclinic_B[i, j] =
                    real(baroclinic_B), imag(baroclinic_B)
                real_stress_B[i, j], imag_stress_B[i, j] =
                    real(stress_B), imag(stress_B)
                diffusivity[i, j] = real(A)
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "Surface-pressure equation diagnostic sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1)
            end
        end

        for j in eachindex(y), i in eachindex(x)
            centered_stencil_is_wet(sampled_wet, i, j) || continue
            velocity_x[i, j] = centered_horizontal_derivative(
                imag_A, sampled_wet, y, i, j, 2,
            )
            velocity_y[i, j] = -centered_horizontal_derivative(
                imag_A, sampled_wet, x, i, j, 1,
            )
            inviscid_velocity_x[i, j] = centered_horizontal_derivative(
                inviscid_A_i, sampled_wet, y, i, j, 2,
            )
            inviscid_velocity_y[i, j] = -centered_horizontal_derivative(
                inviscid_A_i, sampled_wet, x, i, j, 1,
            )
            frictional_velocity_x[i, j] = centered_horizontal_derivative(
                frictional_A_i, sampled_wet, y, i, j, 2,
            )
            frictional_velocity_y[i, j] = -centered_horizontal_derivative(
                frictional_A_i, sampled_wet, x, i, j, 1,
            )
            baroclinic_source[i, j] = -(
                centered_horizontal_derivative(
                    real_baroclinic_B, sampled_wet, x, i, j, 1,
                ) +
                centered_horizontal_derivative(
                    imag_baroclinic_B, sampled_wet, y, i, j, 2,
                )
            )
            stress_source[i, j] = -(
                centered_horizontal_derivative(
                    real_stress_B, sampled_wet, x, i, j, 1,
                ) +
                centered_horizontal_derivative(
                    imag_stress_B, sampled_wet, y, i, j, 2,
                )
            )
            source[i, j] =
                baroclinic_source[i, j] + stress_source[i, j]
            transport_x_divergence[i, j] =
                centered_horizontal_derivative(
                    diagnostics.analytical_transport_U,
                    sampled_wet, x, i, j, 1,
                )
            transport_y_divergence[i, j] =
                centered_horizontal_derivative(
                    diagnostics.analytical_transport_V,
                    sampled_wet, y, i, j, 2,
                )
            residual[i, j] = (
                transport_x_divergence[i, j] +
                transport_y_divergence[i, j]
            )
            wet[i, j] = true
        end
    end

    wet_points = count(wet)
    wet_points > 0 ||
        @warn "Surface-pressure equation diagnostic grid has no points with a centered wet stencil"
    finite_l2(values) = sqrt(sum(abs2, values[wet]))
    residual_scale = max(
        finite_l2(transport_x_divergence),
        finite_l2(transport_y_divergence),
        eps(),
    )
    total_seconds = time() - started
    timings = (
        sampling=sampling_timing,
        total_seconds,
        sampled_wet_points=count(sampled_wet),
        wet_points,
        points_per_second=wet_points / max(sampling_timing.seconds, eps()),
        relative_l2_residual=wet_points > 0 ?
            finite_l2(residual) / residual_scale : NaN,
    )
    @info "Surface-pressure equation diagnostic calculation completed" seconds=round(total_seconds; digits=3) wet_points relative_l2_residual=round(timings.relative_l2_residual; digits=6)
    return PGSurfacePressureEquationDiagnostics(
        x, y, velocity_x, velocity_y,
        inviscid_velocity_x, inviscid_velocity_y,
        frictional_velocity_x, frictional_velocity_y,
        diffusivity, baroclinic_source, stress_source, source, residual, wet,
        timings,
    )
end

function compute_surface_pressure_equation_diagnostics(
    solution::PGSolution,
    x_values,
    y_values,
)
    x, y = collect(float.(x_values)), collect(float.(y_values))
    length(x) >= 3 && length(y) >= 3 ||
        throw(ArgumentError("x and y must each contain at least three points"))
    issorted(x) && all(diff(x) .> 0) ||
        throw(ArgumentError("x must be strictly increasing"))
    issorted(y) && all(diff(y) .> 0) ||
        throw(ArgumentError("y must be strictly increasing"))
    shape = (length(x), length(y))
    U, V = fill(NaN, shape), fill(NaN, shape)
    wet = falses(shape)
    for j in eachindex(y), i in eachindex(x)
        xx, yy = x[i], y[j]
        solution.problem.inside(xx, yy) || continue
        solution.problem.depth(xx, yy) > 0 || continue
        current = depth_integrated_current(solution, xx, yy)
        U[i, j], V[i, j] = current.U, current.V
        wet[i, j] = true
    end
    return compute_surface_pressure_equation_diagnostics(
        solution,
        (; x, y, analytical_transport_U=U, analytical_transport_V=V, wet),
    )
end

"""
    write_solution(fields, prefix)

Write `<prefix>_3D_solution.vtr` containing `u`, `v`, `w`, and `b`, and
`<prefix>_surface_solution.vtr` containing surface pressure and depth.
"""
function write_solution(fields::PGFields, prefix)
    write_started = time()
    file_prefix = string(prefix)
    volume_base = file_prefix * "_3D_solution"
    surface_base = file_prefix * "_surface_solution"
    @info "Writing sampled PG solution" prefix=file_prefix
    _, volume_timing = timed_stage("writing 3D VTK fields") do
        vtk_grid(volume_base, fields.x, fields.y, fields.z) do vtk
            vtk["u"] = fields.u
            vtk["v"] = fields.v
            vtk["w"] = fields.w
            vtk["b"] = fields.b
        end
    end
    _, surface_timing = timed_stage("writing surface VTK fields") do
        vtk_grid(surface_base, fields.x, fields.y) do vtk
            vtk["surface_pressure"] = fields.surface_pressure
            vtk["depth"] = fields.depth
        end
    end
    total_seconds = time() - write_started
    @info "Solution output completed" seconds=round(total_seconds; digits=3)
    return (
        volume=volume_base * ".vtr",
        surface=surface_base * ".vtr",
        timings=(volume=volume_timing, surface=surface_timing, total_seconds),
    )
end

function horizontal_slice(fields, values, z_index)
    checkbounds(fields.z, z_index)
    return values[:, :, z_index], "z=$(round(fields.z[z_index]; digits=3))"
end

function terrain_following_slice(fields, values, sigma_level)
    0 <= sigma_level < 1 ||
        throw(ArgumentError("sigma_level must satisfy 0 ≤ sigma_level < 1"))
    slice = fill(NaN, length(fields.x), length(fields.y))
    for j in eachindex(fields.y), i in eachindex(fields.x)
        target_z = -sigma_level * fields.depth[i, j]
        upper = searchsortedfirst(fields.z, target_z)
        if upper == 1
            fields.wet[i, j, 1] && (slice[i, j] = values[i, j, 1])
            continue
        elseif upper > length(fields.z)
            fields.wet[i, j, end] && (slice[i, j] = values[i, j, end])
            continue
        end
        lower = upper - 1
        fields.wet[i, j, lower] && fields.wet[i, j, upper] || continue
        weight = (
            (target_z - fields.z[lower]) /
            (fields.z[upper] - fields.z[lower])
        )
        slice[i, j] = (
            (1 - weight) * values[i, j, lower] +
            weight * values[i, j, upper]
        )
    end
    return slice, "z/H=-$(round(sigma_level; digits=3))"
end

function finite_slice_absmax(values)
    finite = filter(isfinite, vec(values))
    return isempty(finite) ? 1.0 : max(maximum(abs, finite), eps())
end

function field_plot(fields, values, title, level_label; signed=false)
    options = signed ? (
        color=:balance,
        clims=(-finite_slice_absmax(values), finite_slice_absmax(values)),
    ) : (;)
    heatmap(
        fields.x, fields.y, values';
        xlabel="x", ylabel="y", title="$title at $level_label",
        aspect_ratio=:equal, options...,
    )
end

"""
    plot_solution(fields; z_index=nothing, sigma_level=nothing, output=nothing)

Plot one slice of all four 3D fields together with surface pressure and depth.
Pass `z_index` for a fixed-height slice or `sigma_level` for a
terrain-following slice at `z/H = -sigma_level`. If neither is supplied, the
surface level is plotted. Fixed-height slices below the local bottom are left
blank. Pass `output="figure.png"` to save the figure.
"""
function plot_solution(
    fields::PGFields;
    z_index=nothing,
    sigma_level=nothing,
    output=nothing,
)
    !isnothing(z_index) && !isnothing(sigma_level) &&
        throw(ArgumentError("specify either z_index or sigma_level, not both"))
    if isnothing(sigma_level)
        selected_index = isnothing(z_index) ? length(fields.z) : z_index
        slice_field = values -> horizontal_slice(
            fields, values, selected_index,
        )
        @info "Plotting PG solution" z_index=selected_index z=fields.z[selected_index] output
    else
        slice_field = values -> terrain_following_slice(
            fields, values, sigma_level,
        )
        @info "Plotting PG solution" sigma_level output
    end
    figure, timing = timed_stage("rendering solution plot") do
        u, level_label = slice_field(fields.u)
        v, _ = slice_field(fields.v)
        w, _ = slice_field(fields.w)
        b, _ = slice_field(fields.b)
        panels = [
            field_plot(fields, u, "u", level_label; signed=true),
            field_plot(fields, v, "v", level_label; signed=true),
            field_plot(fields, w, "w", level_label; signed=true),
            field_plot(fields, b, "buoyancy", level_label),
            heatmap(fields.x, fields.y, fields.surface_pressure';
                    xlabel="x", ylabel="y", title="surface pressure", aspect_ratio=:equal),
            heatmap(fields.x, fields.y, fields.depth';
                    xlabel="x", ylabel="y", title="depth", aspect_ratio=:equal),
        ]
        plot(panels...; layout=(2, 3), size=(1200, 750))
    end
    if !isnothing(output)
        _, save_timing = timed_stage("saving solution plot") do
            savefig(figure, output)
        end
        @info "Solution plot saved" path=string(output) seconds=round(save_timing.seconds; digits=3)
    end
    return figure
end

function plot_surface_pressure(fields::PGFields; output=nothing)
    @info "Plotting surface pressure" output
    figure, _ = timed_stage("rendering surface-pressure plot") do
        heatmap(
            fields.x, fields.y, fields.surface_pressure';
            xlabel="x", ylabel="y", title="surface pressure", aspect_ratio=:equal,
        )
    end
    if !isnothing(output)
        _, save_timing = timed_stage("saving surface-pressure plot") do
            savefig(figure, output)
        end
        @info "Surface-pressure plot saved" path=string(output) seconds=round(save_timing.seconds; digits=3)
    end
    return figure
end

function sampled_vectors(diagnostics, x_component, y_component, stride)
    stride > 0 || throw(ArgumentError("stride must be positive"))
    indices = CartesianIndex{2}[]
    for j in 1:stride:length(diagnostics.y), i in 1:stride:length(diagnostics.x)
        diagnostics.wet[i, j] && push!(indices, CartesianIndex(i, j))
    end
    return (
        x=[diagnostics.x[index[1]] for index in indices],
        y=[diagnostics.y[index[2]] for index in indices],
        u=[x_component[index] for index in indices],
        v=[y_component[index] for index in indices],
    )
end

function vector_paths(vectors, vector_scale, head_length)
    shaft_x, shaft_y = Float64[], Float64[]
    head_x, head_y = Float64[], Float64[]
    head_width = 0.6head_length
    for (x, y, u, v) in zip(vectors.x, vectors.y, vectors.u, vectors.v)
        dx, dy = vector_scale * u, vector_scale * v
        length = hypot(dx, dy)
        length > 0 || continue
        tip_x, tip_y = x + dx, y + dy
        unit_x, unit_y = dx / length, dy / length
        base_x, base_y = tip_x - head_length * unit_x, tip_y - head_length * unit_y
        perpendicular_x, perpendicular_y = -unit_y, unit_x
        left_x = base_x + head_width * perpendicular_x
        left_y = base_y + head_width * perpendicular_y
        right_x = base_x - head_width * perpendicular_x
        right_y = base_y - head_width * perpendicular_y
        append!(shaft_x, (x, tip_x, NaN))
        append!(shaft_y, (y, tip_y, NaN))
        append!(head_x, (left_x, tip_x, right_x, NaN))
        append!(head_y, (left_y, tip_y, right_y, NaN))
    end
    return (; shaft_x, shaft_y, head_x, head_y)
end

function vector_diagnostic_plot(
    diagnostics,
    x_component,
    y_component,
    title;
    stride=max(1, cld(max(length(diagnostics.x), length(diagnostics.y)), 16)),
    arrow_scale=0.35,
    arrowhead_size=0.08,
    contours=nothing,
    magnitude_scale=nothing,
    color=:inferno,
)
    arrow_scale > 0 || throw(ArgumentError("arrow_scale must be positive"))
    arrowhead_size > 0 || throw(ArgumentError("arrowhead_size must be positive"))
    magnitude = hypot.(x_component, y_component)
    figure = heatmap(
        diagnostics.x, diagnostics.y, magnitude';
        xlabel="x", ylabel="y", title, colorbar_title="magnitude",
        clims=isnothing(magnitude_scale) ? :auto : (0, magnitude_scale),
        color,
        aspect_ratio=:equal,
    )
    if !isnothing(contours)
        contour!(
            figure, diagnostics.x, diagnostics.y, contours';
            color=:black, linewidth=0.7, colorbar=false, label=false,
        )
    end
    vectors = sampled_vectors(diagnostics, x_component, y_component, stride)
    maximum_magnitude = isnothing(magnitude_scale) ?
        (isempty(vectors.u) ? 0.0 : maximum(hypot.(vectors.u, vectors.v))) :
        magnitude_scale
    sampled_dx = stride * minimum(diff(diagnostics.x))
    sampled_dy = stride * minimum(diff(diagnostics.y))
    maximum_arrow_length = arrow_scale * min(sampled_dx, sampled_dy)
    vector_scale = maximum_magnitude > 0 ?
        maximum_arrow_length / maximum_magnitude : 0.0
    head_length = arrowhead_size * min(sampled_dx, sampled_dy)
    paths = vector_paths(vectors, vector_scale, head_length)
    plot!(
        figure, paths.shaft_x, paths.shaft_y;
        color=:white, linewidth=1, label=false,
    )
    plot!(
        figure, paths.head_x, paths.head_y;
        color=:white, linewidth=1, label=false,
    )
    return figure
end

function save_diagnostic_plot(figure, output, description)
    if !isnothing(output)
        _, timing = timed_stage("saving $description") do
            savefig(figure, output)
        end
        @info "$description saved" path=string(output) seconds=round(timing.seconds; digits=3)
    end
    return figure
end

"""
    plot_surface_pressure_equation_diagnostics(
        diagnostics; stride=nothing, output=nothing,
    )

Plot the inviscid, frictional, and total advection-velocity magnitudes and
vectors on a common scale, plus diffusivity and the separate baroclinic and
surface-stress source terms in the surface-pressure equation (75).
"""
function plot_surface_pressure_equation_diagnostics(
    diagnostics::PGSurfacePressureEquationDiagnostics;
    stride=nothing,
    arrow_scale=0.35,
    arrowhead_size=0.08,
    output=nothing,
)
    vector_stride = isnothing(stride) ?
        max(1, cld(max(length(diagnostics.x), length(diagnostics.y)), 16)) :
        stride
    source_scale = max(
        finite_absmax(diagnostics.baroclinic_source),
        finite_absmax(diagnostics.stress_source),
    )
    velocity_scale = maximum((
        finite_absmax(hypot.(
            diagnostics.inviscid_advection_velocity_x,
            diagnostics.inviscid_advection_velocity_y,
        )),
        finite_absmax(hypot.(
            diagnostics.frictional_advection_velocity_x,
            diagnostics.frictional_advection_velocity_y,
        )),
        finite_absmax(hypot.(
            diagnostics.advection_velocity_x,
            diagnostics.advection_velocity_y,
        )),
    ))
    @info "Plotting surface-pressure equation diagnostics" stride=vector_stride arrow_scale arrowhead_size output
    figure, _ = timed_stage(
        "rendering surface-pressure equation diagnostic plot",
    ) do
        inviscid_velocity = vector_diagnostic_plot(
            diagnostics,
            diagnostics.inviscid_advection_velocity_x,
            diagnostics.inviscid_advection_velocity_y,
            "inviscid advection velocity";
            stride=vector_stride,
            arrow_scale,
            arrowhead_size,
            magnitude_scale=velocity_scale,
        )
        frictional_velocity = vector_diagnostic_plot(
            diagnostics,
            diagnostics.frictional_advection_velocity_x,
            diagnostics.frictional_advection_velocity_y,
            "frictional advection velocity";
            stride=vector_stride,
            arrow_scale,
            arrowhead_size,
            magnitude_scale=velocity_scale,
        )
        total_velocity = vector_diagnostic_plot(
            diagnostics,
            diagnostics.advection_velocity_x,
            diagnostics.advection_velocity_y,
            "total advection velocity";
            stride=vector_stride,
            arrow_scale,
            arrowhead_size,
            magnitude_scale=velocity_scale,
        )
        diffusivity = heatmap(
            diagnostics.x, diagnostics.y, diagnostics.diffusivity';
            xlabel="x",
            ylabel="y",
            title="diffusivity Aᵣ",
            colorbar_title="Aᵣ",
            color=:viridis,
            aspect_ratio=:equal,
        )
        baroclinic_source = heatmap(
            diagnostics.x, diagnostics.y, diagnostics.baroclinic_source';
            xlabel="x",
            ylabel="y",
            title="baroclinic source",
            colorbar_title="source",
            color=:balance,
            clims=(-source_scale, source_scale),
            aspect_ratio=:equal,
        )
        stress_source = heatmap(
            diagnostics.x, diagnostics.y, diagnostics.stress_source';
            xlabel="x",
            ylabel="y",
            title="stress source",
            colorbar_title="source",
            color=:balance,
            clims=(-source_scale, source_scale),
            aspect_ratio=:equal,
        )
        plot(
            inviscid_velocity, frictional_velocity, total_velocity,
            diffusivity, baroclinic_source, stress_source;
            layout=(2, 3),
            size=(1350, 850),
            plot_title="surface-pressure equation coefficients",
            plot_titlefontsize=16,
            margin=2Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "surface-pressure equation diagnostic plot",
    )
end

function plot_bottom_stress(
    diagnostics::PGDiagnostics;
    stride=nothing,
    arrow_scale=0.35,
    arrowhead_size=0.08,
    output=nothing,
)
    vector_stride = isnothing(stride) ?
        max(1, cld(max(length(diagnostics.x), length(diagnostics.y)), 16)) : stride
    @info "Plotting bottom stress" stride=vector_stride arrow_scale arrowhead_size output
    figure, _ = timed_stage("rendering bottom-stress plot") do
        vector_diagnostic_plot(
            diagnostics, diagnostics.bottom_stress_x, diagnostics.bottom_stress_y,
            "bottom stress"; stride=vector_stride, arrow_scale, arrowhead_size,
        )
    end
    return save_diagnostic_plot(figure, output, "bottom-stress plot")
end

function plot_depth_integrated_current(
    diagnostics::PGDiagnostics;
    stride=nothing,
    arrow_scale=0.35,
    arrowhead_size=0.08,
    output=nothing,
)
    vector_stride = isnothing(stride) ?
        max(1, cld(max(length(diagnostics.x), length(diagnostics.y)), 16)) : stride
    @info "Plotting depth-integrated current" stride=vector_stride arrow_scale arrowhead_size output
    figure, _ = timed_stage("rendering depth-integrated-current plot") do
        vector_diagnostic_plot(
            diagnostics, diagnostics.transport_U, diagnostics.transport_V,
            "depth-integrated current"; stride=vector_stride, arrow_scale,
            arrowhead_size, contours=diagnostics.streamfunction,
        )
    end
    return save_diagnostic_plot(
        figure, output, "depth-integrated-current plot",
    )
end

function plot_streamfunction(diagnostics::PGDiagnostics; output=nothing)
    @info "Plotting transport streamfunction" output
    figure, _ = timed_stage("rendering transport-streamfunction plot") do
        contourf(
            diagnostics.x, diagnostics.y, diagnostics.streamfunction';
            xlabel="x", ylabel="y", title="transport streamfunction",
            colorbar_title="streamfunction", aspect_ratio=:equal,
        )
    end
    return save_diagnostic_plot(figure, output, "transport-streamfunction plot")
end

function plot_diagnostics(
    diagnostics::PGDiagnostics;
    stride=nothing,
    arrow_scale=0.35,
    arrowhead_size=0.08,
    output=nothing,
)
    vector_stride = isnothing(stride) ?
        max(1, cld(max(length(diagnostics.x), length(diagnostics.y)), 16)) : stride
    @info "Plotting PG diagnostics" stride=vector_stride arrow_scale arrowhead_size output
    figure, _ = timed_stage("rendering combined diagnostic plot") do
        stress = vector_diagnostic_plot(
            diagnostics, diagnostics.bottom_stress_x, diagnostics.bottom_stress_y,
            "bottom stress"; stride=vector_stride, arrow_scale, arrowhead_size,
        )
        current = vector_diagnostic_plot(
            diagnostics, diagnostics.transport_U, diagnostics.transport_V,
            "depth-integrated current"; stride=vector_stride, arrow_scale,
            arrowhead_size, contours=diagnostics.streamfunction,
        )
        Ψ = contourf(
            diagnostics.x, diagnostics.y, diagnostics.streamfunction';
            xlabel="x", ylabel="y", title="transport streamfunction",
            colorbar_title="streamfunction", aspect_ratio=:equal,
        )
        plot(stress, current, Ψ; layout=(1, 3), size=(1350, 420))
    end
    return save_diagnostic_plot(figure, output, "combined diagnostic plot")
end

function scalar_diagnostic_plot(dim::PGDIMDiagnostics, values, title, colorbar_title)
    return heatmap(
        dim.x, dim.y, values';
        xlabel="x", ylabel="y", title, colorbar_title, aspect_ratio=:equal,
    )
end

"""
    plot_baroclinic_potential_energy(dim; output=nothing)

Plot `χ = -∫_{-H}^0 z b dz` from equation (34).
"""
function plot_baroclinic_potential_energy(
    dim::PGDIMDiagnostics;
    output=nothing,
)
    @info "Plotting baroclinic potential energy" output
    figure, _ = timed_stage("rendering baroclinic-potential-energy plot") do
        scalar_diagnostic_plot(
            dim, dim.chi, "baroclinic potential energy", "χ",
        )
    end
    return save_diagnostic_plot(
        figure, output, "baroclinic-potential-energy plot",
    )
end

"""
    plot_bottom_pressure(dim; output=nothing)

Plot `p_b = p_s - ∫_{-H}^0 b dz`.
"""
function plot_bottom_pressure(dim::PGDIMDiagnostics; output=nothing)
    @info "Plotting bottom pressure" output
    figure, _ = timed_stage("rendering bottom-pressure plot") do
        scalar_diagnostic_plot(dim, dim.bottom_pressure, "bottom pressure", "p_b")
    end
    return save_diagnostic_plot(figure, output, "bottom-pressure plot")
end

finite_absmax(values) = begin
    finite = filter(isfinite, vec(values))
    isempty(finite) ? 1.0 : max(maximum(abs, finite), eps())
end

function dim_component_fields(dim::PGDIMDiagnostics, component)
    if component === :x
        return (
            ("-f V", dim.x_coriolis),
            ("-H ∂x p_b", dim.x_bottom_pressure),
            ("-∂x χ", dim.x_baroclinic),
            ("Ek τ_bˣ", dim.x_bottom_stress),
            ("Ek τˣ", dim.x_surface_stress),
            ("residual", dim.x_residual),
        )
    elseif component === :y
        return (
            ("f U", dim.y_coriolis),
            ("-H ∂y p_b", dim.y_bottom_pressure),
            ("-∂y χ", dim.y_baroclinic),
            ("Ek τ_bʸ", dim.y_bottom_stress),
            ("Ek τʸ", dim.y_surface_stress),
            ("residual", dim.y_residual),
        )
    end
    throw(ArgumentError("component must be :x or :y"))
end

function dim_component_plots(dim::PGDIMDiagnostics, component; show_xlabel=true)
    fields = dim_component_fields(dim, component)
    scale = maximum(finite_absmax(values) for (_, values) in fields[1:5])
    panels = [
        heatmap(
            dim.x, dim.y, values';
            xlabel=show_xlabel ? "x" : "",
            ylabel=index == 1 ? "y" : "",
            title,
            clims=(-scale, scale),
            color=:balance,
            colorbar=false,
            aspect_ratio=:equal,
            xticks=show_xlabel ? :auto : false,
            yticks=index == 1 ? :auto : false,
            titlefontsize=11,
            guidefontsize=10,
            tickfontsize=8,
            margin=1.5Plots.mm,
        )
        for (index, (title, values)) in enumerate(fields)
    ]
    return panels, scale
end

function dim_balance_colorbar(scale)
    ticks = collect(range(-scale, scale; length=5))
    labels = [@sprintf("%.2g", value) for value in ticks]
    levels = collect(range(-scale, scale; length=256))
    return heatmap(
        [0.0, 1.0], levels, repeat(levels, 1, 2);
        clims=(-scale, scale),
        color=:balance,
        colorbar=false,
        title="value",
        titlefontsize=10,
        xticks=false,
        yticks=(ticks, labels),
        tickfontsize=9,
        xaxis=false,
        ymirror=true,
        margin=2Plots.mm,
    )
end

"""
    plot_dim_balance(dim; component=:both, output=nothing)

Plot every signed term and the closure residual corresponding to equations
(32)--(33). `component` may be `:x`, `:y`, or `:both`. Terms in each momentum
component share symmetric color limits so their magnitudes can be compared
directly.
"""
function plot_dim_balance(
    dim::PGDIMDiagnostics;
    component=:both,
    output=nothing,
)
    component in (:x, :y, :both) ||
        throw(ArgumentError("component must be :x, :y, or :both"))
    @info "Plotting depth-integrated momentum balance" component output
    figure, _ = timed_stage("rendering DIM balance plot") do
        if component === :both
            x_panels, x_scale = dim_component_plots(
                dim, :x; show_xlabel=false,
            )
            y_panels, y_scale = dim_component_plots(dim, :y)
            layout = @layout [
                grid(1, 6) x_colorbar{0.07w}
                grid(1, 6) y_colorbar{0.07w}
            ]
            plot(
                x_panels..., dim_balance_colorbar(x_scale),
                y_panels..., dim_balance_colorbar(y_scale);
                layout,
                size=(1900, 700),
                plot_title="depth-integrated momentum balance",
                plot_titlefontsize=17,
                margin=2Plots.mm,
            )
        else
            panels, scale = dim_component_plots(dim, component)
            layout = @layout [grid(1, 6) colorbar{0.07w}]
            plot(
                panels..., dim_balance_colorbar(scale);
                layout,
                size=(1900, 370),
                plot_title="$(component)-momentum balance",
                plot_titlefontsize=17,
                margin=2Plots.mm,
            )
        end
    end
    return save_diagnostic_plot(figure, output, "DIM balance plot")
end

"""
    plot_barotropic_vorticity_balance(diagnostics; output=nothing)

Plot the nonzero signed terms and closure residual in the barotropic vorticity
equation (36). The exact fixed-lid transport divergence is retained in the
diagnostic data and residual but omitted from the figure. All plotted physical
terms and the residual share symmetric color limits.
"""
function plot_barotropic_vorticity_balance(
    diagnostics::PGBarotropicVorticityDiagnostics;
    output=nothing,
)
    fields = (
        ("U ∂x f + V ∂y f", diagnostics.planetary_vorticity_advection),
        ("J(p_b, H)", diagnostics.bottom_pressure_torque),
        ("Ek curl τ_b", diagnostics.bottom_stress_curl),
        ("Ek curl τ", diagnostics.surface_stress_curl),
        ("residual", diagnostics.residual),
    )
    scale = maximum(finite_absmax(values) for (_, values) in fields[1:4])
    @info "Plotting barotropic vorticity balance" output
    figure, _ = timed_stage("rendering barotropic vorticity balance plot") do
        panels = [
            heatmap(
                diagnostics.x, diagnostics.y, values';
                xlabel="x",
                ylabel=index == 1 ? "y" : "",
                title,
                clims=(-scale, scale),
                color=:balance,
                colorbar=false,
                aspect_ratio=:equal,
                yticks=index == 1 ? :auto : false,
                titlefontsize=11,
                guidefontsize=10,
                tickfontsize=8,
                margin=1.5Plots.mm,
            )
            for (index, (title, values)) in enumerate(fields)
        ]
        layout = @layout [grid(1, 5) colorbar{0.07w}]
        plot(
            panels..., dim_balance_colorbar(scale);
            layout,
            size=(1650, 390),
            plot_title="barotropic vorticity balance",
            plot_titlefontsize=17,
            margin=2Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "barotropic vorticity balance plot",
    )
end

"""
    plot_vorticity_balance(diagnostics; output=nothing)

Plot every signed term and the closure residual in the depth-integrated
vorticity equation (41). All physical terms and the residual share symmetric
color limits.
"""
function plot_vorticity_balance(
    diagnostics::PGVorticityDiagnostics;
    output=nothing,
)
    fields = (
        ("U ∂x f + V ∂y f", diagnostics.planetary_vorticity_advection),
        ("f w_b", diagnostics.bottom_vortex_stretching),
        ("Ek curl τ_b", diagnostics.bottom_stress_curl),
        ("Ek curl τ", diagnostics.surface_stress_curl),
        ("residual", diagnostics.residual),
    )
    scale = maximum(finite_absmax(values) for (_, values) in fields[1:4])
    @info "Plotting depth-integrated vorticity balance" output
    figure, _ = timed_stage("rendering vorticity balance plot") do
        panels = [
            heatmap(
                diagnostics.x, diagnostics.y, values';
                xlabel="x",
                ylabel=index == 1 ? "y" : "",
                title,
                clims=(-scale, scale),
                color=:balance,
                colorbar=false,
                aspect_ratio=:equal,
                yticks=index == 1 ? :auto : false,
                titlefontsize=11,
                guidefontsize=10,
                tickfontsize=8,
                margin=1.5Plots.mm,
            )
            for (index, (title, values)) in enumerate(fields)
        ]
        layout = @layout [grid(1, 5) colorbar{0.08w}]
        plot(
            panels..., dim_balance_colorbar(scale);
            layout,
            size=(1650, 390),
            plot_title="depth-integrated vorticity balance",
            plot_titlefontsize=17,
            margin=2Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "depth-integrated vorticity balance plot",
    )
end

"""
    plot_geostrophic_contour_vorticity_balance(
        diagnostics; output=nothing,
    )

Plot every signed term and the closure residual in the geostrophic-contour
vorticity balance (43). The two stress panels include the equation's minus
sign, so the first panel equals the sum of panels two through four. All
physical terms and the residual share symmetric color limits.
"""
function plot_geostrophic_contour_vorticity_balance(
    diagnostics::PGGeostrophicContourVorticityDiagnostics;
    output=nothing,
)
    fields = (
        ("J(f/H, Ψ)", diagnostics.geostrophic_contour_advection),
        ("J(1/H, χ)", diagnostics.jebar),
        ("-Ek curl(τ_b/H)", .-diagnostics.bottom_stress_curl),
        ("-Ek curl(τ/H)", .-diagnostics.surface_stress_curl),
        ("residual", diagnostics.residual),
    )
    scale = maximum(finite_absmax(values) for (_, values) in fields[1:4])
    @info "Plotting geostrophic-contour vorticity balance" output
    figure, _ = timed_stage(
        "rendering geostrophic-contour vorticity balance plot",
    ) do
        panels = [
            heatmap(
                diagnostics.x, diagnostics.y, values';
                xlabel="x",
                ylabel=index == 1 ? "y" : "",
                title,
                clims=(-scale, scale),
                color=:balance,
                colorbar=false,
                aspect_ratio=:equal,
                yticks=index == 1 ? :auto : false,
                titlefontsize=11,
                guidefontsize=10,
                tickfontsize=8,
                margin=1.5Plots.mm,
            )
            for (index, (title, values)) in enumerate(fields)
        ]
        layout = @layout [grid(1, 5) colorbar{0.08w}]
        plot(
            panels..., dim_balance_colorbar(scale);
            layout,
            size=(1650, 390),
            plot_title="geostrophic-contour vorticity balance",
            plot_titlefontsize=17,
            margin=2Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "geostrophic-contour vorticity balance plot",
    )
end

"""
    plot_transport_divergence_vorticity_balance(
        diagnostics; output=nothing,
    )

Plot every signed term and the closure residual in the transport-divergence
vorticity balance (47). The four physical panels sum to zero, apart from the
reported residual. All physical terms and the residual share symmetric color
limits.
"""
function plot_transport_divergence_vorticity_balance(
    diagnostics::PGTransportDivergenceVorticityDiagnostics;
    output=nothing,
)
    fields = (
        ("J(p_b, H/f)", diagnostics.bottom_pressure_torque),
        ("J(χ, 1/f)", diagnostics.baroclinic_torque),
        ("Ek curl(τ_b/f)", diagnostics.bottom_stress_curl),
        ("Ek curl(τ/f)", diagnostics.surface_stress_curl),
        ("residual", diagnostics.residual),
    )
    scale = maximum(finite_absmax(values) for (_, values) in fields[1:4])
    @info "Plotting transport-divergence vorticity balance" output
    figure, _ = timed_stage(
        "rendering transport-divergence vorticity balance plot",
    ) do
        panels = [
            heatmap(
                diagnostics.x, diagnostics.y, values';
                xlabel="x",
                ylabel=index == 1 ? "y" : "",
                title,
                clims=(-scale, scale),
                color=:balance,
                colorbar=false,
                aspect_ratio=:equal,
                yticks=index == 1 ? :auto : false,
                titlefontsize=11,
                guidefontsize=10,
                tickfontsize=8,
                margin=1.5Plots.mm,
            )
            for (index, (title, values)) in enumerate(fields)
        ]
        layout = @layout [grid(1, 5) colorbar{0.08w}]
        plot(
            panels..., dim_balance_colorbar(scale);
            layout,
            size=(1650, 390),
            plot_title="transport-divergence vorticity balance",
            plot_titlefontsize=17,
            margin=2Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "transport-divergence vorticity balance plot",
    )
end

forcing_sensitivity_labels() = ("B", "bₛ", "τˣ", "τʸ")

"""
    plot_velocity_sensitivity(sensitivity, x, y, z; output=nothing)

Plot vertical profiles of the `u` and `v` derivatives with respect to each
forcing-pattern multiplier. `z` must be an increasing collection of levels
within the local water column.
"""
function plot_velocity_sensitivity(
    sensitivity::PGForcingSensitivities,
    x,
    y,
    z_values;
    output=nothing,
)
    z = collect(float.(z_values))
    !isempty(z) || throw(ArgumentError("z must contain at least one level"))
    issorted(z) && all(diff(z) .> 0) ||
        throw(ArgumentError("z must be strictly increasing"))
    responses = [
        [horizontal_velocity(solution, x, y, zz) for zz in z]
        for solution in values(forcing_sensitivity_pairs(sensitivity))
    ]
    labels = forcing_sensitivity_labels()
    @info "Plotting velocity forcing sensitivity" x y levels=length(z) output
    figure, _ = timed_stage("rendering velocity sensitivity plot") do
        u = plot(
            xlabel="∂u / ∂ amplitude",
            ylabel="z",
            title="zonal velocity sensitivity",
            legend=:best,
        )
        v = plot(
            xlabel="∂v / ∂ amplitude",
            ylabel="z",
            title="meridional velocity sensitivity",
            legend=:best,
        )
        for (response, label) in zip(responses, labels)
            plot!(u, getproperty.(response, :u), z; label, linewidth=2)
            plot!(v, getproperty.(response, :v), z; label, linewidth=2)
        end
        plot(
            u,
            v;
            layout=(1, 2),
            size=(1000, 430),
            plot_title="forcing sensitivity at (x, y) = ($(x), $(y))",
            plot_titlefontsize=15,
            margin=3Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "velocity forcing sensitivity plot",
    )
end

"""
    plot_transport_sensitivity(sensitivity, x, y; output=nothing)

Plot the `U` and `V` depth-integrated-current derivatives with respect to each
forcing-pattern multiplier at `(x, y)`.
"""
function plot_transport_sensitivity(
    sensitivity::PGForcingSensitivities,
    x,
    y;
    output=nothing,
)
    response = depth_integrated_current_sensitivity(sensitivity, x, y)
    labels = collect(forcing_sensitivity_labels())
    U = [current.U for current in values(response)]
    V = [current.V for current in values(response)]
    scale = max(maximum(abs, U), maximum(abs, V), eps())
    @info "Plotting transport forcing sensitivity" x y output
    figure, _ = timed_stage("rendering transport sensitivity plot") do
        panels = [
            bar(
                labels,
                values;
                xlabel="forcing pattern",
                ylabel="derivative",
                title,
                color=:steelblue,
                legend=false,
                ylims=(-1.05scale, 1.05scale),
            )
            for (title, values) in (
                ("zonal transport sensitivity ∂U/∂a", U),
                ("meridional transport sensitivity ∂V/∂a", V),
            )
        ]
        plot(
            panels...;
            layout=(1, 2),
            size=(1000, 430),
            plot_title="forcing sensitivity at (x, y) = ($(x), $(y))",
            plot_titlefontsize=15,
            margin=3Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "transport forcing sensitivity plot",
    )
end

function adjoint_kernel_plot(
    sensitivity::PGTransportAdjointSensitivity,
    values,
    title,
)
    scale = finite_absmax(values)
    figure = heatmap(
        sensitivity.x,
        sensitivity.y,
        values';
        xlabel="forcing x",
        ylabel="forcing y",
        title,
        titlefontsize=9,
        guidefontsize=9,
        tickfontsize=7,
        colorbar_title="functional derivative",
        colorbar_titlefontsize=9,
        color=:balance,
        clims=(-scale, scale),
        aspect_ratio=:equal,
    )
    target = sensitivity.target
    if hasproperty(target, :kind) && target.kind === :zonal_section
        plot!(
            figure,
            collect(target.x_limits),
            fill(target.y, 2);
            color=:black,
            linewidth=4,
            label=false,
        )
    else
        θ = range(0, 2π; length=181)
        radius = sensitivity.target_radius
        plot!(
            figure,
            target.x .+ radius .* cos.(θ),
            target.y .+ radius .* sin.(θ);
            color=:black,
            linestyle=:dash,
            linewidth=2,
            label=false,
        )
        scatter!(
            figure,
            [target.x],
            [target.y];
            marker=:xcross,
            markersize=7,
            markerstrokewidth=2,
            color=:black,
            label=false,
        )
    end
    return figure
end

"""
    plot_transport_adjoint_sensitivity(
        sensitivity; buoyancy_levels=eachindex(sensitivity.z), output=nothing,
    )

Plot the spatial sensitivity maps for surface stress and surface buoyancy,
together with selected horizontal slices through the three-dimensional
restoring-buoyancy sensitivity kernel.
"""
function plot_transport_adjoint_sensitivity(
    sensitivity::PGTransportAdjointSensitivity;
    buoyancy_levels=eachindex(sensitivity.z),
    output=nothing,
)
    levels = collect(buoyancy_levels)
    all(level -> checkbounds(Bool, sensitivity.z, level), levels) ||
        throw(BoundsError(sensitivity.z, levels))
    cost_label = sensitivity.cost_label
    target = sensitivity.target
    radius = sensitivity.target_radius
    target_description =
        hasproperty(target, :kind) && target.kind === :zonal_section ?
        (
            "section x=($(round(target.x_limits[1]; digits=3)), " *
            "$(round(target.x_limits[2]; digits=3))) at " *
            "y=$(round(target.y; digits=3))"
        ) :
        (
            "target=($(round(target.x; digits=3)), " *
            "$(round(target.y; digits=3))), " *
            "radius=$(round(radius; digits=3))"
        )
    @info "Plotting transport adjoint sensitivity" cost_label target radius levels output
    figure, _ = timed_stage(
        "rendering transport adjoint sensitivity plot",
    ) do
        panels = [
            adjoint_kernel_plot(
                sensitivity,
                sensitivity.wind_stress_x,
                "δ$(cost_label) / δτˣ(x,y)",
            ),
            adjoint_kernel_plot(
                sensitivity,
                sensitivity.wind_stress_y,
                "δ$(cost_label) / δτʸ(x,y)",
            ),
            adjoint_kernel_plot(
                sensitivity,
                sensitivity.surface_buoyancy,
                "δ$(cost_label) / δbₛ(x,y)",
            ),
            adjoint_kernel_plot(
                sensitivity,
                sensitivity.bottom_restoring_buoyancy,
                "bottom-boundary δ$(cost_label) / δB(x,y,-H)",
            ),
        ]
        append!(
            panels,
            [
                adjoint_kernel_plot(
                    sensitivity,
                    sensitivity.restoring_buoyancy[:, :, level],
                    "δ$(cost_label) / δB(x,y,z), z=$(round(
                        sensitivity.z[level]; digits=3,
                    ))",
                )
                for level in levels
            ],
        )
        columns = min(3, length(panels))
        rows = cld(length(panels), columns)
        plot(
            panels...;
            layout=(rows, columns),
            size=(500columns, 420rows),
            plot_title=(
                "$(cost_label) adjoint sensitivity; $target_description"
            ),
            plot_titlefontsize=13,
            margin=4Plots.mm,
        )
    end
    return save_diagnostic_plot(
        figure, output, "transport adjoint sensitivity plot",
    )
end

performance_summary(solution::PGSolution) = solution.timings
performance_summary(fields::PGFields) = fields.timings
performance_summary(field::PGLagrangianVelocityField) = field.timings
performance_summary(trajectories::PGLagrangianTrajectories) =
    trajectories.timings
performance_summary(solution::PGStreamfunction) = solution.timings
performance_summary(diagnostics::PGDiagnostics) = diagnostics.timings
performance_summary(diagnostics::PGDIMDiagnostics) = diagnostics.timings
performance_summary(diagnostics::PGBarotropicVorticityDiagnostics) =
    diagnostics.timings
performance_summary(diagnostics::PGVorticityDiagnostics) = diagnostics.timings
performance_summary(
    diagnostics::PGGeostrophicContourVorticityDiagnostics,
) = diagnostics.timings
performance_summary(
    diagnostics::PGTransportDivergenceVorticityDiagnostics,
) = diagnostics.timings
performance_summary(
    diagnostics::PGSurfacePressureEquationDiagnostics,
) = diagnostics.timings
performance_summary(sensitivity::PGForcingSensitivities) =
    sensitivity.timings
performance_summary(sensitivity::PGTransportAdjointSensitivity) =
    sensitivity.timings


include(joinpath(@__DIR__, "CellBudgets.jl"))

end
