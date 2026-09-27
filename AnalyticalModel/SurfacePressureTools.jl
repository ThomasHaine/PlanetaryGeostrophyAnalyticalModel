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

"""
    generate_pressure_mesh(path, boundary; resolution)

Write a two-dimensional Gmsh mesh for the closed polygon `boundary`, whose
entries are `(x, y)` pairs. The mesh includes the physical line tag
`"boundary"` required by `solve_surface_pressure`.
"""
function generate_pressure_mesh(path, boundary; resolution, overwrite=false)
    if isfile(path)
        overwrite || return path
        rm(path)
    end
    gmsh.initialize()
    try
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
    return path
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

Analytical coefficient `S` in equation (63), such that
`τ_b = S * (∂x + i∂y)p_s + T`.
"""
function bottom_stress_coefficient(p, H, f)
    q = sqrt(im * f / (p.ν * p.Ek))
    return tanh(q * H) / (q * p.Ek)
end

"""
    bottom_stress_forcing(p, H, f, τ, B, bs, x, y; gradient_step)

Analytical forcing `T` in equation (64), including the baroclinic and surface
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

Analytical forcing `B` in equation (68), with a negative baroclinic
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

"""
    solve_surface_pressure(mesh_file, kernels, p, H, f, f_x, f_y, τ, B, bs, H_x, H_y)

Solve the same coupled Gridap surface-pressure problem used in
`PG_Solution_v0.1.ipynb`. The returned `gradps(x, y)` evaluates
`(∂x + i ∂y) p_s` for use in `velocity_b23_from_gradps`.
"""
function solve_surface_pressure(
    mesh_file, kernels, p, Hfun, ffun, f_x, f_y, τ, B, bs, H_x, H_y;
    order=2,
)
    solve_started = time()
    @info "Surface-pressure solve started" mesh_file=string(mesh_file) finite_element_order=order
    model, mesh_timing = pressure_timed_stage("loading Gridap surface mesh") do
        GmshDiscreteModel(mesh_file)
    end
    spaces, space_timing = pressure_timed_stage("constructing pressure finite-element spaces") do
        Ω = Triangulation(model)
        dΩ = Measure(Ω, 2 * order)
        reffe = ReferenceFE(lagrangian, Float64, order)
        V = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
        U = TrialFESpace(V)
        (; Ω, dΩ, V, U)
    end
    Ω, dΩ, V, U = spaces.Ω, spaces.dΩ, spaces.V, spaces.U

    function A_fn(x)
        H = Hfun(x[1], x[2])
        return H > 0 ? pressure_coefficient(p, H, ffun(x[1], x[2])) : 1.0
    end
    function baroclinic_B_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        return buoyancy_transport_b23(
            kernels, p, Hfun, ffun, f_x, f_y, B, bs, H_x, H_y, x[1], x[2],
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
        return transport_forcing(
            p, Hfun, ffun, τ, B, bs, x[1], x[2],
        )
    end
    function T_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        return bottom_stress_forcing(
            p, Hfun, ffun, τ, B, bs, x[1], x[2],
        )
    end

    a(u, v) = ∫(real((∇(v) ⋅ VectorValue(1.0, -im)) *
                     (A_fn * (∇(u) ⋅ VectorValue(1.0, im)))) )dΩ
    l(v) = ∫(real((∇(v) ⋅ VectorValue(-1.0, im)) * B_fn))dΩ
    operator, assembly_timing = pressure_timed_stage("assembling surface-pressure operator") do
        AffineFEOperator(a, l, U, V)
    end
    psurf, linear_solve_timing = pressure_timed_stage("solving surface-pressure linear system") do
        Gridap.solve(operator)
    end
    ∇psurf = ∇(psurf)
    gradps(x, y) = begin
        gradient = evaluate(∇psurf, Point(x, y))
        gradient[1] + im * gradient[2]
    end
    total_seconds = time() - solve_started
    timings = (
        mesh_load=mesh_timing,
        finite_element_setup=space_timing,
        operator_assembly=assembly_timing,
        linear_solve=linear_solve_timing,
        total_seconds,
    )
    @info "Surface-pressure solve completed" seconds=round(total_seconds; digits=3)
    return (;
        model, psurf, gradps, A_fn, B_fn, baroclinic_B_fn, stress_B_fn,
        diagnostic_B_fn, T_fn, order, timings,
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
        reffe = ReferenceFE(lagrangian, Float64, order)
        V = TestFESpace(
            pressure.model,
            reffe;
            conformity=:H1,
            dirichlet_tags="boundary",
        )
        U = TrialFESpace(V)
        (; Ω, dΩ, V, U)
    end
    Ω, dΩ, V, U = spaces.Ω, spaces.dΩ, spaces.V, spaces.U
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
    operator, assembly_timing = pressure_timed_stage(
        "assembling adjoint surface-pressure operator",
    ) do
        AffineFEOperator(a, l, U, V)
    end
    finite_element, linear_solve_timing = pressure_timed_stage(
        "solving adjoint surface-pressure linear system",
    ) do
        Gridap.solve(operator)
    end
    finite_element_gradient = ∇(finite_element)
    gradient = (x, y) -> evaluate(finite_element_gradient, Point(x, y))
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
        vector_reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
        V_vector = TestFESpace(
            pressure.model, vector_reffe;
            conformity=:H1, dirichlet_tags="boundary",
        )
        (; Ω, dΩ, V, U, V_vector)
    end
    Ω, dΩ, V, U = spaces.Ω, spaces.dΩ, spaces.V, spaces.U

    current, projection_timing = pressure_timed_stage(
        "projecting depth-integrated current",
    ) do
        gradp = interpolate(∇(pressure.psurf), spaces.V_vector)
        real_A = interpolate(x -> real(pressure.A_fn(x)), U)
        imag_A = interpolate(x -> imag(pressure.A_fn(x)), U)
        B_cache = Dict{Tuple{Float64,Float64},ComplexF64}()
        B_at(x) = get!(
            () -> ComplexF64(pressure.B_fn(x)),
            B_cache,
            (Float64(x[1]), Float64(x[2])),
        )
        real_B = interpolate(x -> real(B_at(x)), U)
        imag_B = interpolate(x -> imag(B_at(x)), U)
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
