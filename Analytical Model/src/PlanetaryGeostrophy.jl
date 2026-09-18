module PlanetaryGeostrophy

using ForwardDiff
using Gridap
using Plots
using WriteVTK

include(joinpath(@__DIR__, "..", "..", "TheoryNotes", "VelocityFieldPrototype.jl"))
using .VelocityFieldPrototype
include(joinpath(@__DIR__, "..", "SurfacePressureTools.jl"))

export PGParameters,
       PGProblem,
       PGSolution,
       PGFields,
       rectangle_boundary,
       ellipse_boundary,
       solve_pg,
       surface_pressure,
       horizontal_velocity,
       buoyancy,
       sample_solution,
       write_solution,
       plot_solution,
       plot_surface_pressure

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

struct PGSolution{P,K,S,B,V}
    problem::P
    kernels::K
    pressure::S
    buoyancy_function::B
    horizontal_velocity_function::V
    mesh_file::String
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
    generate_pressure_mesh(
        mesh_file, problem.boundary;
        resolution=mesh_resolution, overwrite=overwrite_mesh,
    )
    kernels = load_symbolic_kernels()
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
    return PGSolution(problem, kernels, pressure, b, uv, string(mesh_file))
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

struct PGFields{TX<:AbstractVector,TY<:AbstractVector,TZ<:AbstractVector}
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

    for j in eachindex(y), i in eachindex(x)
        xx, yy = x[i], y[j]
        solution.problem.inside(xx, yy) || continue
        H = solution.problem.depth(xx, yy)
        H > 0 || continue
        depth[i, j] = H
        ps[i, j] = float(surface_pressure(solution, xx, yy))
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

    w = reconstruct_vertical_velocity(u, v, wet, depth, x, y, z)
    return PGFields(x, y, z, u, v, w, b, ps, depth, wet)
end

"""
    write_solution(fields, prefix)

Write `<prefix>_3D_solution.vti` containing `u`, `v`, `w`, and `b`, and
`<prefix>_surface_solution.vti` containing surface pressure and depth.
"""
function write_solution(fields::PGFields, prefix)
    file_prefix = string(prefix)
    volume_path = file_prefix * "_3D_solution.vti"
    surface_path = file_prefix * "_surface_solution.vti"
    vtk_grid(volume_path, fields.x, fields.y, fields.z) do vtk
        vtk["u"] = fields.u
        vtk["v"] = fields.v
        vtk["w"] = fields.w
        vtk["b"] = fields.b
    end
    vtk_grid(surface_path, fields.x, fields.y) do vtk
        vtk["surface_pressure"] = fields.surface_pressure
        vtk["depth"] = fields.depth
    end
    return (volume=volume_path, surface=surface_path)
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
    figure = plot(panels...; layout=(2, 3), size=(1200, 750))
    isnothing(output) || savefig(figure, output)
    return figure
end

function plot_surface_pressure(fields::PGFields; output=nothing)
    figure = heatmap(
        fields.x, fields.y, fields.surface_pressure';
        xlabel="x", ylabel="y", title="surface pressure", aspect_ratio=:equal,
    )
    isnothing(output) || savefig(figure, output)
    return figure
end

end
