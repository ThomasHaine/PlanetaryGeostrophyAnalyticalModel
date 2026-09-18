module PlanetaryGeostrophy

using ForwardDiff
using Gridap
using Plots
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
include(joinpath(@__DIR__, "..", "..", "Analytical Model", "SurfacePressureTools.jl"))

export PGParameters,
       PGProblem,
       PGSolution,
       PGFields,
       PGStreamfunction,
       PGDiagnostics,
       rectangle_boundary,
       ellipse_boundary,
       solve_pg,
       surface_pressure,
       horizontal_velocity,
       buoyancy,
       bottom_stress,
       depth_integrated_current,
       solve_streamfunction,
       streamfunction,
       sample_solution,
       compute_diagnostics,
       write_solution,
       plot_solution,
       plot_surface_pressure,
       plot_bottom_stress,
       plot_depth_integrated_current,
       plot_streamfunction,
       plot_diagnostics,
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
otherwise centered automatic differentiation is used.
"""
struct PGProblem{P,H,F,FX,FY,T,B,BS,HX,HY,BD,I}
    parameters::P
    depth::H
    coriolis::F
    coriolis_x::FX
    coriolis_y::FY
    wind_stress::T
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

    cx = sum(first, polygon) / length(polygon)
    cy = sum(last, polygon) / length(polygon)
    depth(cx, cy) > 0 || throw(ArgumentError("depth must be positive in the basin"))
    coriolis(cx, cy) != 0 || throw(ArgumentError("coriolis must be nonzero in the basin"))

    return PGProblem(
        parameters, depth, coriolis, fx, fy, wind_stress,
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
             pressure_order=2, overwrite_mesh=false)

Generate or reuse a surface mesh, solve the Gridap surface-pressure equation,
and construct callable Green-function solutions for buoyancy and horizontal
velocity.
"""
function solve_pg(
    problem::PGProblem;
    mesh_file=tempname() * ".msh",
    mesh_resolution=0.1,
    pressure_order=2,
    overwrite_mesh=false,
)
    mesh_resolution > 0 || throw(ArgumentError("mesh_resolution must be positive"))
    pressure_order > 0 || throw(ArgumentError("pressure_order must be positive"))
    solve_started = time()
    reusing_mesh = isfile(mesh_file) && !overwrite_mesh
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
    )
    b = (x, y, z) -> buoyancy_from_forcing(
        p, problem.depth, problem.restoring_buoyancy,
        problem.surface_buoyancy, x, y, z,
    )
    uv = (x, y, z) -> velocity_b23_from_gradps(
        kernels, p, problem.depth, problem.coriolis,
        problem.coriolis_x, problem.coriolis_y, pressure.gradps,
        problem.wind_stress, problem.restoring_buoyancy,
        problem.surface_buoyancy, problem.depth_x, problem.depth_y,
        x, y, z; gradient_step=problem.derivative_step,
    )
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
    return evaluate(solution.pressure.psurf, Point(x, y))
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
buoyancy, and wind-stress contributions, using equations (62)--(64).
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

"""
    depth_integrated_current(solution, x, y)

Evaluate the non-dimensional depth-integrated horizontal current `(U, V)`.
"""
function depth_integrated_current(solution::PGSolution, x, y)
    H = require_water(solution, x, y)
    p = internal_parameters(solution.problem.parameters)
    A = pressure_coefficient(p, H, solution.problem.coriolis(x, y))
    current = A * solution.pressure.gradps(x, y) +
              solution.pressure.diagnostic_B_fn(Point(x, y))
    return (U=real(current), V=imag(current))
end

struct PGStreamfunction{F,V,T}
    finite_element::F
    value::V
    timings::T
end

"""
    solve_streamfunction(solution; order=2)

Solve for the streamfunction of the depth-integrated horizontal current.
"""
function solve_streamfunction(solution::PGSolution; order=2)
    order > 0 || throw(ArgumentError("order must be positive"))
    result = solve_transport_streamfunction(solution.pressure; order)
    value = (x, y) -> evaluate(result.finite_element, Point(x, y))
    return PGStreamfunction(result.finite_element, value, result.timings)
end

function streamfunction(solution::PGStreamfunction, x, y)
    return solution.value(x, y)
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
    streamfunction::Matrix{Float64}
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
    wet = falses(shape)
    step = solution.problem.derivative_step

    @info "Solution sampling started" Nx=length(x) Ny=length(y) Nz=length(z) grid_points=prod(shape)
    _, pressure_timing = timed_stage("sampling surface pressure and bathymetry") do
        for j in eachindex(y), i in eachindex(x)
            xx, yy = x[i], y[j]
            solution.problem.inside(xx, yy) || continue
            H = solution.problem.depth(xx, yy)
            H > 0 || continue
            depth[i, j] = H
            ps[i, j] = float(surface_pressure(solution, xx, yy))
        end
    end

    progress_stride = max(1, cld(length(y), 10))
    _, fields_timing = timed_stage("sampling 3D horizontal velocity and buoyancy") do
        for j in eachindex(y)
            for i in eachindex(x)
                xx, yy = x[i], y[j]
                H = depth[i, j]
                isfinite(H) || continue
                stencil_is_inside = all(solution.problem.inside(xq, yq) for (xq, yq) in (
                    (xx - step, yy), (xx + step, yy), (xx, yy - step), (xx, yy + step),
                ))
                stencil_is_inside || continue
                for k in eachindex(z)
                    -H <= z[k] <= 0 || continue
                    uv = solution.horizontal_velocity_function(xx, yy, z[k])
                    u[i, j, k], v[i, j, k] = float(real(uv)), float(imag(uv))
                    b[i, j, k] = float(solution.buoyancy_function(xx, yy, z[k]))
                    wet[i, j, k] = true
                end
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "3D field sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1) wet_points=count(wet)
            end
        end
    end

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

"""
    compute_diagnostics(solution, x, y; streamfunction_order=2)

Sample bottom stress and depth-integrated horizontal current, and solve and
sample its streamfunction, on an increasing horizontal rectilinear grid.
"""
function compute_diagnostics(
    solution::PGSolution,
    x_values,
    y_values;
    streamfunction_order=2,
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

    shape = (length(x), length(y))
    τx, τy = fill(NaN, shape), fill(NaN, shape)
    U, V = fill(NaN, shape), fill(NaN, shape)
    Ψ = fill(NaN, shape)
    wet = falses(shape)
    step = solution.problem.derivative_step
    progress_stride = max(1, cld(length(y), 10))
    _, sampling_timing = timed_stage(
        "sampling bottom stress, transport, and streamfunction",
    ) do
        for j in eachindex(y)
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
                stress = bottom_stress(solution, xx, yy)
                current = depth_integrated_current(solution, xx, yy)
                τx[i, j], τy[i, j] = float(stress.x), float(stress.y)
                U[i, j], V[i, j] = float(current.U), float(current.V)
                Ψ[i, j] = float(streamfunction(ψ, xx, yy))
                wet[i, j] = true
            end
            if j == 1 || j == length(y) || j % progress_stride == 0
                @info "PG diagnostic sampling progress" rows_completed=j rows_total=length(y) percent=round(100j / length(y); digits=1) wet_points=count(wet)
            end
        end
    end

    total_seconds = time() - diagnostic_started
    wet_points = count(wet)
    timings = (
        streamfunction=ψ.timings,
        sampling=sampling_timing,
        total_seconds,
        wet_points,
        points_per_second=wet_points / max(sampling_timing.seconds, eps()),
    )
    @info "PG diagnostic calculation completed" seconds=round(total_seconds; digits=3) wet_points points_per_second=round(timings.points_per_second; digits=1)
    return PGDiagnostics(x, y, τx, τy, U, V, Ψ, wet, timings)
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

function field_plot(fields, values, title, z_index)
    heatmap(
        fields.x, fields.y, values[:, :, z_index]';
        xlabel="x", ylabel="y", title="$title at z=$(round(fields.z[z_index], digits=3))",
        aspect_ratio=:equal,
    )
end

"""
    plot_solution(fields; z_index=length(fields.z), output=nothing)

Plot one horizontal slice of all four 3D fields together with surface
pressure and depth. Pass `output="figure.png"` to save the figure.
"""
function plot_solution(fields::PGFields; z_index=length(fields.z), output=nothing)
    checkbounds(fields.z, z_index)
    @info "Plotting PG solution" z_index z=fields.z[z_index] output
    figure, timing = timed_stage("rendering solution plot") do
        panels = [
            field_plot(fields, fields.u, "u", z_index),
            field_plot(fields, fields.v, "v", z_index),
            field_plot(fields, fields.w, "w", z_index),
            field_plot(fields, fields.b, "buoyancy", z_index),
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

function sampled_vectors(diagnostics::PGDiagnostics, x_component, y_component, stride)
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
    diagnostics::PGDiagnostics,
    x_component,
    y_component,
    title;
    stride=max(1, cld(max(length(diagnostics.x), length(diagnostics.y)), 16)),
    arrow_scale=0.35,
    arrowhead_size=0.08,
)
    arrow_scale > 0 || throw(ArgumentError("arrow_scale must be positive"))
    arrowhead_size > 0 || throw(ArgumentError("arrowhead_size must be positive"))
    magnitude = hypot.(x_component, y_component)
    figure = heatmap(
        diagnostics.x, diagnostics.y, magnitude';
        xlabel="x", ylabel="y", title, colorbar_title="magnitude",
        aspect_ratio=:equal,
    )
    vectors = sampled_vectors(diagnostics, x_component, y_component, stride)
    maximum_magnitude = isempty(vectors.u) ? 0.0 :
        maximum(hypot.(vectors.u, vectors.v))
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
            "depth-integrated current"; stride=vector_stride, arrow_scale, arrowhead_size,
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
            "depth-integrated current"; stride=vector_stride, arrow_scale, arrowhead_size,
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

performance_summary(solution::PGSolution) = solution.timings
performance_summary(fields::PGFields) = fields.timings
performance_summary(solution::PGStreamfunction) = solution.timings
performance_summary(diagnostics::PGDiagnostics) = diagnostics.timings

end
