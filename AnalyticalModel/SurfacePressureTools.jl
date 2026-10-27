using Gmsh
using Gridap
using GridapGmsh

pressure_stage_metrics(measurement) = (
    seconds=measurement.time,
    allocated_mib=measurement.bytes / 2.0^20,
    gc_seconds=measurement.gctime,
)

function pressure_timed_stage(f, label)
    @info "Starting $label"
    measurement = @timed f()
    metrics = pressure_stage_metrics(measurement)
    @info "Completed $label" seconds=round(metrics.seconds; digits=3) allocated_mib=round(metrics.allocated_mib; digits=2) gc_seconds=round(metrics.gc_seconds; digits=3)
    return measurement.value, metrics
end

# Version 2 sizes elements by `resolution` everywhere instead of extending the
# coastline point spacing into the interior.
const PRESSURE_MESH_GENERATOR_VERSION = 2

pressure_mesh_metadata_path(path) = string(path) * ".meta"

pressure_mesh_metadata(boundary, resolution) = string(
    "generator=", PRESSURE_MESH_GENERATOR_VERSION, "\n",
    "resolution=", repr(Float64(resolution)), "\n",
    "boundary=", repr([(Float64(x), Float64(y)) for (x, y) in boundary]), "\n",
)

"""
    pressure_mesh_is_current(path, boundary; resolution)

Whether the mesh file at `path` exists and was written by this version of
`generate_pressure_mesh` for the same `boundary` and `resolution`.
"""
function pressure_mesh_is_current(path, boundary; resolution)
    metadata_path = pressure_mesh_metadata_path(path)
    return isfile(path) && isfile(metadata_path) &&
           read(metadata_path, String) == pressure_mesh_metadata(boundary, resolution)
end

"""
    generate_pressure_mesh(path, boundary; resolution, overwrite=false)

Write a two-dimensional Gmsh mesh for the closed polygon `boundary`, whose
entries are `(x, y)` pairs. The mesh includes the physical line tag
`"boundary"` required by `solve_surface_pressure`.

Elements have size close to `resolution` throughout the basin: the mean edge
length is within a few percent of it when `resolution` is small compared with
the basin. The polygon vertices are mesh nodes, so where the coastline's vertex
spacing is smaller than `resolution` the elements next to the coast are
smaller too.

An existing mesh file is reused only if it was written by this generator for
the same `boundary` and `resolution`, as recorded in the sidecar file
`path * ".meta"`. Otherwise it is regenerated, as it is when `overwrite=true`.
"""
function generate_pressure_mesh(path, boundary; resolution, overwrite=false)
    if !overwrite && pressure_mesh_is_current(path, boundary; resolution)
        return path
    end
    isfile(path) && rm(path)
    gmsh.initialize()
    try
        # Gmsh by default also takes element sizes from the boundary points
        # and extends them inward. The coastline's edges are then split into a
        # whole number of segments, so the realized size followed the polygon
        # vertex spacing rather than `resolution`.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
        gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)
        gmsh.model.add("surface_pressure")
        occ = gmsh.model.occ
        point_tags = [
            occ.addPoint(x, y, 0.0, resolution) for (x, y) in boundary
        ]
        lines = [
            occ.addLine(point_tags[i], point_tags[mod1(i + 1, length(point_tags))])
            for i in eachindex(point_tags)
        ]
        loop = occ.addCurveLoop(lines)
        surface = occ.addPlaneSurface([loop])
        occ.synchronize()
        boundary_tag = gmsh.model.addPhysicalGroup(1, lines)
        gmsh.model.setPhysicalName(1, boundary_tag, "boundary")
        domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, domain_tag, "domain")
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    write(pressure_mesh_metadata_path(path), pressure_mesh_metadata(boundary, resolution))
    return path
end

"""
    mean_edge_length(model)

Mean triangle edge length of a Gridap surface mesh, the realized mesh size.
"""
function mean_edge_length(model)
    total, count = 0.0, 0
    for vertices in get_cell_coordinates(Triangulation(model))
        for k in eachindex(vertices)
            a, b = vertices[k], vertices[mod1(k + 1, length(vertices))]
            total += hypot(a[1] - b[1], a[2] - b[2])
            count += 1
        end
    end
    return total / count
end

rectangle_boundary(Lx, Ly) = [
    (-Lx / 2, -Ly / 2), (Lx / 2, -Ly / 2),
    (Lx / 2, Ly / 2), (-Lx / 2, Ly / 2),
]

ellipse_boundary(ax, ay; points=96) = [
    (ax * cos(θ), ay * sin(θ)) for θ in range(0, 2π; length=points + 1)[1:end-1]
]

function pressure_coefficient(p, H, f)
    q = sqrt(im * f / (p.ν * p.Ek))
    return ((exp(2 * q * H) - 1) / (q * (1 + exp(2 * q * H))) - H) /
           (p.ν * q^2 * p.Ek)
end

function wind_transport_coefficient(p, H, f, τ)
    q = sqrt(im * f / (p.ν * p.Ek))
    return (exp(q * H) - 1)^2 * τ /
           (p.ν * q^2 * (1 + exp(2 * q * H)))
end

function baroclinic_pressure_gradient(
    p, Hfun, B, bs, x, y, ξ;
    gradient_step=1e-5,
)
    return VelocityFieldPrototype.finite_complex_gradient(
        (xx, yy) -> VelocityFieldPrototype.Pi_direct(
            p, Hfun, B, bs, xx, yy, ξ,
        ),
        x, y; step=gradient_step,
    )
end

"""
    bottom_stress_coefficient(p, H, f)

Analytical coefficient `S` in equation (57), such that
`τ_b = S * (∂x + i∂y)p_s + T`.
"""
function bottom_stress_coefficient(p, H, f)
    q = sqrt(im * f / (p.ν * p.Ek))
    return tanh(q * H) / (q * p.Ek)
end

"""
    bottom_stress_forcing(p, H, f, τ, B, bs, x, y; gradient_step)

Analytical forcing `T` in equation (58), including the baroclinic and surface
wind-stress contributions.
"""
function bottom_stress_forcing(
    p, Hfun, ffun, τ, B, bs, x, y;
    gradient_step=1e-5,
)
    H = Hfun(x, y)
    q = sqrt(im * ffun(x, y) / (p.ν * p.Ek))
    baroclinic = VelocityFieldPrototype.split_quadgk(
        ξ -> exp(q * (H - ξ)) * (1 + exp(2 * q * ξ)) /
             (1 + exp(2 * q * H)) *
             baroclinic_pressure_gradient(
                 p, Hfun, B, bs, x, y, ξ; gradient_step,
             ),
        -H, 0.0,
    ) / p.Ek
    wind = -2 * exp(q * H) / (1 + exp(2 * q * H)) * τ(x, y)
    return -baroclinic + wind
end

"""
    transport_forcing(p, H, f, τ, B, bs, x, y; gradient_step)

Analytical forcing `B` in equation (61), with a negative baroclinic
contribution and the unchanged surface wind-stress contribution.
"""
function transport_forcing(
    p, Hfun, ffun, τ, B, bs, x, y;
    gradient_step=1e-5,
)
    H = Hfun(x, y)
    q = sqrt(im * ffun(x, y) / (p.ν * p.Ek))
    baroclinic = VelocityFieldPrototype.split_quadgk(
        ξ -> exp(-q * ξ) *
             (exp(q * ξ) - exp(q * H)) *
             (exp(q * (ξ + H)) - 1) *
             baroclinic_pressure_gradient(
                 p, Hfun, B, bs, x, y, ξ; gradient_step,
             ),
        -H, 0.0,
    ) / p.Ek
    wind = (exp(q * H) - 1)^2 * τ(x, y)
    return (-baroclinic + wind) /
           (p.ν * q^2 * (1 + exp(2 * q * H)))
end

check_coastal_condition(condition) =
    condition in (:no_normal_flow, :dirichlet) || throw(ArgumentError(
        "coastal_condition must be :no_normal_flow or :dirichlet; got $condition",
    ))

"""
    pressure_spaces(model, order, coastal_condition)

Finite-element spaces for the surface-pressure problem.

`:no_normal_flow` imposes no essential condition on `p_s`. The weak form's
natural boundary condition is then `U·n = 0`: integrating
`∫ v Re{(∂x - i∂y)𝔘} dA` by parts leaves `∮ v U·n ds`. This is needed where
the coast is a vertical wall (`H > 0`). `p_s` is fixed up to a constant, which
a Lagrange multiplier sets so that `∫ p_s dA = 0`.

`:dirichlet` sets `p_s = 0` on the coast, as in earlier versions. That
removes only the geostrophic part of the normal transport. The frictional and
baroclinic parts, `𝔄_r ∂_n p_s + 𝔅·n`, can still cross the coast unless
`H → 0` there.
"""
function pressure_spaces(model, order, coastal_condition)
    check_coastal_condition(coastal_condition)
    reffe = ReferenceFE(lagrangian, Float64, order)
    if coastal_condition === :dirichlet
        V = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
        U = TrialFESpace(V)
        return (; V, U, test=V, trial=U)
    end
    V = TestFESpace(model, reffe; conformity=:H1)
    U = TrialFESpace(V)
    Λ = ConstantFESpace(model)
    return (;
        V, U,
        test=MultiFieldFESpace([V, Λ]),
        trial=MultiFieldFESpace([TrialFESpace(V), TrialFESpace(Λ)]),
    )
end

"""
    solve_pressure_problem(a, l, spaces, dΩ, coastal_condition)

Assemble and solve `a(u, v) = l(v)` on `spaces`, adding the mean-zero
constraint for `:no_normal_flow`. Returns the operator assembly and the
single-field solution, each with timing metrics.
"""
function solve_pressure_problem(a, l, spaces, dΩ, coastal_condition, label)
    operator, assembly_timing = pressure_timed_stage("assembling $label operator") do
        if coastal_condition === :dirichlet
            AffineFEOperator(a, l, spaces.trial, spaces.test)
        else
            constrained_a((u, λ), (v, μ)) = a(u, v) + ∫(λ * v)dΩ + ∫(μ * u)dΩ
            constrained_l((v, μ)) = l(v)
            AffineFEOperator(constrained_a, constrained_l, spaces.trial, spaces.test)
        end
    end
    solution, solve_timing = pressure_timed_stage("solving $label linear system") do
        result = Gridap.solve(operator)
        coastal_condition === :dirichlet ? result : first(result)
    end
    return solution, assembly_timing, solve_timing
end

"""
    solve_surface_pressure(mesh_file, kernels, p, H, f, f_x, f_y, τ, B, bs, H_x, H_y;
                           order=2, method=:fast, quadrature=ColumnQuadrature(),
                           gradient_step=1e-5, forcing_assembly=:quadrature,
                           coastal_condition=:no_normal_flow)

Solve the same coupled Gridap surface-pressure problem used in
`PG_Solution_v0.1.ipynb`. The returned `gradps(x, y)` evaluates
`(∂x + i ∂y) p_s` for use in `velocity_b23_from_gradps`.

With `method=:fast` (the default), the transport forcing `𝔅` and the
bottom-stress forcing `T` use the single-integral forms (61) and (58). `𝔅` is
evaluated in parallel before assembly. With `forcing_assembly=:quadrature`
(the default) it is evaluated at every quadrature point, which gives the same
load vector as direct evaluation. With `:nodal` it is evaluated once per node
and assembled as an order-`order` Lagrangian interpolant, with about a third
of the evaluations and an O(h^(order+1)) interpolation error.
`method=:b23` restores the original B23 quadrature, evaluated serially at every
quadrature point.

`coastal_condition=:no_normal_flow` (the default) enforces zero normal
depth-integrated transport through the coast and fixes the mean of `p_s` to
zero. `:dirichlet` restores the earlier `p_s = 0` coastal condition (see
`pressure_spaces`).
"""
function solve_surface_pressure(
    mesh_file, kernels, p, Hfun, ffun, f_x, f_y, τ, B, bs, H_x, H_y;
    order=2,
    method=:fast,
    quadrature=ColumnQuadrature(),
    gradient_step=1e-5,
    forcing_assembly=:quadrature,
    coastal_condition=:no_normal_flow,
)
    check_coastal_condition(coastal_condition)
    method in (:fast, :b23) ||
        throw(ArgumentError("method must be :fast or :b23; got $method"))
    solve_started = time()
    @info "Surface-pressure solve started" mesh_file=string(mesh_file) finite_element_order=order method coastal_condition threads=Threads.nthreads()
    model, mesh_timing = pressure_timed_stage("loading Gridap surface mesh") do
        GmshDiscreteModel(mesh_file)
    end
    mesh_size = mean_edge_length(model)
    @info "Surface mesh" cells=num_cells(model) mean_edge_length=round(mesh_size; sigdigits=3)
    spaces, space_timing = pressure_timed_stage("constructing pressure finite-element spaces") do
        Ω = Triangulation(model)
        dΩ = Measure(Ω, 2 * order)
        (; Ω, dΩ, pressure_spaces(model, order, coastal_condition)...)
    end
    Ω, dΩ = spaces.Ω, spaces.dΩ

    function A_fn(x)
        H = Hfun(x[1], x[2])
        return H > 0 ? pressure_coefficient(p, H, ffun(x[1], x[2])) : 1.0
    end
    function baroclinic_B_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        method === :b23 && return buoyancy_transport_b23(
            kernels, p, Hfun, ffun, f_x, f_y, B, bs, H_x, H_y, x[1], x[2],
        )
        return fast_baroclinic_transport(
            kernels, p, Hfun, ffun, B, bs, x[1], x[2], quadrature;
            step=gradient_step,
        )
    end
    function stress_B_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        return wind_transport_coefficient(
            p, H, ffun(x[1], x[2]), τ(x[1], x[2]),
        )
    end
    B_fn(x) = baroclinic_B_fn(x) + stress_B_fn(x)
    function diagnostic_B_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        method === :b23 && return transport_forcing(
            p, Hfun, ffun, τ, B, bs, x[1], x[2],
        )
        return B_fn(x)
    end
    function T_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        method === :b23 && return bottom_stress_forcing(
            p, Hfun, ffun, τ, B, bs, x[1], x[2],
        )
        return fast_bottom_stress_forcing(
            kernels, p, Hfun, ffun, τ, B, bs, x[1], x[2], quadrature;
            step=gradient_step,
        )
    end

    # On the fast path, 𝔅 is evaluated in parallel before assembly, either at
    # every quadrature point (exact, held in a CellState) or once per
    # finite-element node (an order-`order` interpolant). The node table is
    # reused by the streamfunction projection and is built on first use there.
    forcing_assembly in (:quadrature, :nodal) || throw(ArgumentError(
        "forcing_assembly must be :quadrature or :nodal; got $forcing_assembly",
    ))
    nodal_table = Ref{Any}(nothing)
    function B_nodal(x)
        method === :b23 && return B_fn(x)
        if isnothing(nodal_table[])
            nodal_table[] = first(nodal_lookup(x -> ComplexF64(B_fn(x)), model, order))
        end
        return nodal_table[](x)
    end
    forcing, forcing_timing = pressure_timed_stage(
        method === :b23 ? "deferring transport forcing to quadrature points" :
        "evaluating transport forcing at $(forcing_assembly) points",
    ) do
        method === :b23 && return (; load=B_fn, points=0)
        if forcing_assembly === :nodal
            B_nodal((0.0, 0.0))
            vector_space = TestFESpace(
                model, ReferenceFE(lagrangian, VectorValue{2,Float64}, order);
                conformity=:H1,
            )
            field = interpolate_everywhere(vector_space) do x
                value = B_nodal(x)
                VectorValue(real(value), imag(value))
            end
            load = field ⋅ VectorValue(1.0, im)
            return (; load, points=num_free_dofs(vector_space) ÷ 2)
        end
        physical_points = collect(get_cell_points(dΩ).cell_phys_point)
        state = CellState(0.0 + 0.0im, dΩ)
        Threads.@threads for cell in eachindex(physical_points)
            points, values = physical_points[cell], state.values[cell]
            for k in eachindex(points)
                values[k] = B_fn(points[k])
            end
        end
        (; load=state, points=sum(length, physical_points))
    end
    forcing_points = forcing.points
    @info "Transport forcing representation" method forcing_assembly forcing_points

    a(u, v) = ∫(real((∇(v) ⋅ VectorValue(1.0, -im)) *
                     (A_fn * (∇(u) ⋅ VectorValue(1.0, im)))) )dΩ
    l(v) = ∫(real((∇(v) ⋅ VectorValue(-1.0, im)) * forcing.load))dΩ
    psurf, assembly_timing, linear_solve_timing = solve_pressure_problem(
        a, l, spaces, dΩ, coastal_condition, "surface-pressure",
    )
    ∇psurf = ∇(psurf)
    probe = interior_probe_point(model)
    psurf_at = PooledPointEvaluator(psurf, probe)
    ∇psurf_at = PooledPointEvaluator(∇psurf, probe)
    gradps(x, y) = begin
        gradient = ∇psurf_at(Point(x, y))
        gradient[1] + im * gradient[2]
    end
    total_seconds = time() - solve_started
    timings = (
        mesh_load=mesh_timing,
        finite_element_setup=space_timing,
        forcing_evaluation=forcing_timing,
        operator_assembly=assembly_timing,
        linear_solve=linear_solve_timing,
        total_seconds,
        forcing_points,
        mean_edge_length=mesh_size,
    )
    @info "Surface-pressure solve completed" seconds=round(total_seconds; digits=3)
    return (;
        model, psurf, psurf_at, gradps, A_fn, B_fn, B_nodal,
        baroclinic_B_fn, stress_B_fn, diagnostic_B_fn, T_fn, order, method,
        quadrature, coastal_condition, probe, timings,
    )
end

"""
    solve_surface_pressure_adjoint(
        pressure, target_x, target_y;
        cost, cost_gradient=nothing, cost_value=nothing,
        transport=nothing, target_weight=nothing, target_radius,
    )

Solve the discrete adjoint of the surface-pressure equation for the
user-defined scalar `cost(U, V)`. By default, the objective uses a normalized
Gaussian spatial weight centered on `(target_x, target_y)`. A supplied
`target_weight(x, y)` replaces that Gaussian and is used without additional
normalization, allowing objectives such as regularized section integrals.

The adjoint source uses `cost_gradient(U, V)` when supplied and otherwise falls
back to `ForwardDiff`. A constant `(cost_U, cost_V)` tuple avoids evaluating
transport during assembly for linear costs. `transport` can provide a cached
transport evaluator for nonlinear costs. Supplying a previously sampled
`cost_value` avoids a separate mesh-wide cost integration.
"""
function solve_surface_pressure_adjoint(
    pressure,
    target_x,
    target_y;
    cost,
    cost_gradient=nothing,
    cost_value=nothing,
    transport=nothing,
    target_weight=nothing,
    target_radius,
)
    isnothing(target_weight) && target_radius <= 0 &&
        throw(ArgumentError("target_radius must be positive"))
    solve_started = time()
    order = pressure.order
    spaces, space_timing = pressure_timed_stage(
        "constructing adjoint pressure finite-element spaces",
    ) do
        Ω = Triangulation(pressure.model)
        dΩ = Measure(Ω, 2 * order)
        # The adjoint uses the forward problem's coastal condition. The
        # mean-zero constraint block is symmetric, so it transposes to itself.
        (; Ω, dΩ, pressure_spaces(pressure.model, order, pressure.coastal_condition)...)
    end
    Ω, dΩ = spaces.Ω, spaces.dΩ
    spatial_weight = if isnothing(target_weight)
        unnormalized_target(x) = exp(
            -(
                (x[1] - target_x)^2 +
                (x[2] - target_y)^2
            ) / (2target_radius^2),
        )
        normalization = sum(∫(unnormalized_target)dΩ)
        normalization > 0 ||
            throw(ArgumentError("target averaging kernel does not overlap the basin"))
        x -> unnormalized_target(x) / normalization
    else
        applicable(target_weight, target_x, target_y) ||
            throw(ArgumentError("target_weight must accept (x, y)"))
        center_weight = target_weight(target_x, target_y)
        center_weight isa Real ||
            throw(ArgumentError("target_weight must return real values"))
        x -> begin
            value = target_weight(x[1], x[2])
            value isa Real ||
                throw(ArgumentError("target_weight must return real values"))
            value
        end
    end
    raw_transport(x) = (
        pressure.A_fn(x) * pressure.gradps(x[1], x[2]) +
        pressure.B_fn(x)
    )
    transport_at = isnothing(transport) ? raw_transport : transport
    constant_gradient = cost_gradient isa Tuple
    if constant_gradient
        length(cost_gradient) == 2 ||
            throw(ArgumentError("constant cost_gradient must have two entries"))
        all(value -> value isa Real, cost_gradient) ||
            throw(ArgumentError("constant cost_gradient entries must be real"))
    elseif !isnothing(cost_gradient) && !applicable(cost_gradient, 0.0, 0.0)
        throw(ArgumentError("cost_gradient must accept (U, V)"))
    end
    function cost_derivatives(x)
        if constant_gradient
            return (;
                value=NaN,
                U=NaN,
                V=NaN,
                cost_U=float(cost_gradient[1]),
                cost_V=float(cost_gradient[2]),
            )
        end
        current = transport_at(x)
        transport_U, transport_V = real(current), imag(current)
        value = cost(transport_U, transport_V)
        value isa Real ||
            throw(ArgumentError("cost(U, V) must return a real scalar"))
        cost_U, cost_V = if isnothing(cost_gradient)
            (
                ForwardDiff.derivative(
                    u -> cost(u, transport_V),
                    transport_U,
                ),
                ForwardDiff.derivative(
                    v -> cost(transport_U, v),
                    transport_V,
                ),
            )
        else
            gradient = cost_gradient(transport_U, transport_V)
            length(gradient) == 2 ||
                throw(ArgumentError("cost_gradient must return two entries"))
            gradient
        end
        cost_U isa Real && cost_V isa Real ||
            throw(ArgumentError("cost_gradient entries must be real"))
        return (; value, U=transport_U, V=transport_V, cost_U, cost_V)
    end
    evaluated_cost_value = if isnothing(cost_value)
        sum(∫(
            x -> begin
                current = transport_at(x)
                value = cost(real(current), imag(current))
                value isa Real ||
                    throw(ArgumentError("cost(U, V) must return a real scalar"))
                spatial_weight(x) * value
            end
        )dΩ)
    else
        cost_value isa Real ||
            throw(ArgumentError("cost_value must be a real scalar"))
        float(cost_value)
    end
    cost_U(x) = cost_derivatives(x).cost_U
    cost_V(x) = cost_derivatives(x).cost_V
    adjoint_A(x) = conj(pressure.A_fn(x))
    a(u, v) = ∫(
        real(
            (∇(v) ⋅ VectorValue(1.0, -im)) *
            (adjoint_A * (∇(u) ⋅ VectorValue(1.0, im))),
        )
    )dΩ
    l(v) = ∫(
        spatial_weight * (
            cost_U *
            real(pressure.A_fn * (∇(v) ⋅ VectorValue(1.0, im))) +
            cost_V *
            imag(pressure.A_fn * (∇(v) ⋅ VectorValue(1.0, im)))
        )
    )dΩ
    finite_element, assembly_timing, linear_solve_timing = solve_pressure_problem(
        a, l, spaces, dΩ, pressure.coastal_condition, "adjoint surface-pressure",
    )
    finite_element_gradient = PooledPointEvaluator(
        ∇(finite_element), pressure.probe,
    )
    gradient = (x, y) -> finite_element_gradient(Point(x, y))
    weight = (x, y) -> spatial_weight(Point(x, y))
    timings = (
        finite_element_setup=space_timing,
        operator_assembly=assembly_timing,
        linear_solve=linear_solve_timing,
        total_seconds=time() - solve_started,
    )
    return (;
        finite_element,
        gradient,
        weight,
        cost_derivatives=(x, y) -> cost_derivatives(Point(x, y)),
        cost_value=float(evaluated_cost_value),
        cost_gradient_mode=constant_gradient ? :constant :
            isnothing(cost_gradient) ? :forwarddiff : :analytic,
        target=(x=float(target_x), y=float(target_y)),
        target_radius=float(target_radius),
        timings,
    )
end

"""
    solve_transport_streamfunction(pressure; order=2)

Solve for the streamfunction of the depth-integrated horizontal current
`U + im * V = A * grad(p_s) + B`, with zero streamfunction on the basin
boundary.
"""
function solve_transport_streamfunction(pressure; order=2)
    solve_started = time()
    @info "Transport streamfunction solve started" finite_element_order=order
    spaces, space_timing = pressure_timed_stage(
        "constructing streamfunction finite-element spaces",
    ) do
        Ω = Triangulation(pressure.model)
        dΩ = Measure(Ω, 2 * order)
        reffe = ReferenceFE(lagrangian, Float64, order)
        V = TestFESpace(
            pressure.model, reffe; conformity=:H1, dirichlet_tags="boundary",
        )
        U = TrialFESpace(V)
        # Coefficient fields are interpolated without a coastal Dirichlet
        # condition. Interpolating them into the Ψ space would zero ∇p_s, 𝔄,
        # and 𝔅 on coastal nodes, where the transport need not vanish when
        # H > 0.
        V_free = TestFESpace(pressure.model, reffe; conformity=:H1)
        vector_reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
        V_vector = TestFESpace(pressure.model, vector_reffe; conformity=:H1)
        (; Ω, dΩ, V, U, V_free, V_vector)
    end
    Ω, dΩ, V, U = spaces.Ω, spaces.dΩ, spaces.V, spaces.U

    current, projection_timing = pressure_timed_stage(
        "projecting depth-integrated current",
    ) do
        gradp = interpolate(∇(pressure.psurf), spaces.V_vector)
        real_A = interpolate(x -> real(pressure.A_fn(x)), spaces.V_free)
        imag_A = interpolate(x -> imag(pressure.A_fn(x)), spaces.V_free)
        # On the fast path, B_nodal reads the node table built for the
        # pressure solve, so this projection needs no new forcing evaluations.
        B_cache = Dict{Tuple{Float64,Float64},ComplexF64}()
        B_at(x) = get!(
            () -> ComplexF64(pressure.B_nodal(x)),
            B_cache,
            (Float64(x[1]), Float64(x[2])),
        )
        real_B = interpolate(x -> real(B_at(x)), spaces.V_free)
        imag_B = interpolate(x -> imag(B_at(x)), spaces.V_free)
        (real_A + im * imag_A) *
        (gradp ⋅ VectorValue(1.0, im)) +
        (real_B + im * imag_B)
    end

    a(ψ, v) = ∫(∇(v) ⋅ ∇(ψ))dΩ
    l(v) = ∫(-real((∇(v) ⋅ VectorValue(im, 1.0)) * current))dΩ
    operator, assembly_timing = pressure_timed_stage(
        "assembling transport-streamfunction operator",
    ) do
        AffineFEOperator(a, l, U, V)
    end
    ψ, linear_solve_timing = pressure_timed_stage(
        "solving transport-streamfunction linear system",
    ) do
        Gridap.solve(operator)
    end
    total_seconds = time() - solve_started
    timings = (
        finite_element_setup=space_timing,
        current_projection=projection_timing,
        operator_assembly=assembly_timing,
        linear_solve=linear_solve_timing,
        total_seconds,
    )
    @info "Transport streamfunction solve completed" seconds=round(total_seconds; digits=3)
    return (; finite_element=ψ, timings)
end
