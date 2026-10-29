using PlanetaryGeostrophy
using Gridap: Measure, Triangulation, ∫
using Test

@testset "PG problem specification" begin
    parameters = PGParameters(0.02, 0.2)
    boundary = rectangle_boundary(2.0, 1.0)
    problem = PGProblem(
        parameters=parameters,
        depth=(x, y) -> 1.0 + 0.1x,
        coriolis=(x, y) -> 1.0 + 0.2y,
        wind_stress=(x, y) -> 0.01 + 0.0im,
        restoring_buoyancy=(x, y, z) -> z,
        surface_buoyancy=(x, y) -> 0.0,
        boundary=boundary,
    )

    @test problem.depth_x(0.0, 0.0) ≈ 0.1
    @test problem.coriolis_y(0.0, 0.0) ≈ 0.2
    @test problem.inside(0.0, 0.0)
    @test !problem.inside(2.0, 0.0)
    @test_throws ArgumentError PGParameters(0.0, 0.2)
end

@testset "Analytical diagnostic coefficients" begin
    p = PlanetaryGeostrophy.VelocityParameters(0.02, 0.2, 1.0, 1.0, 1.0, 0.0)
    H, f = 1.0, 1.0
    τ(x, y) = 0.03 - 0.01im
    Hfun(x, y) = H
    ffun(x, y) = f
    B(x, y, z) = 0.0
    bs(x, y) = 0.0
    q = sqrt(im * f / p.Ek)

    T = PlanetaryGeostrophy.bottom_stress_forcing(
        p, Hfun, ffun, τ, B, bs, 0.0, 0.0,
    )
    Btransport = PlanetaryGeostrophy.transport_forcing(
        p, Hfun, ffun, τ, B, bs, 0.0, 0.0,
    )

    @test T ≈ -2exp(q * H) / (1 + exp(2q * H)) * τ(0.0, 0.0)
    @test Btransport ≈
        (exp(q * H) - 1)^2 /
        (p.ν * q^2 * (1 + exp(2q * H))) * τ(0.0, 0.0)

    for signed_f in (-f, f)
        problem = (
            depth=(x, y) -> H,
            coriolis=(x, y) -> signed_f,
            parameters=(; viscosity=p.ν, Ek=p.Ek),
        )
        components =
            PlanetaryGeostrophy.surface_pressure_imaginary_coefficient_components(
                problem, 0.0, 0.0,
            )
        @test components.inviscid == H / signed_f
        @test components.inviscid + components.frictional ≈ imag(
            PlanetaryGeostrophy.pressure_coefficient(
                p, H, signed_f,
            ),
        )
    end

    small_s_problem = (
        depth=(x, y) -> 1e-6,
        coriolis=(x, y) -> -1.0,
        parameters=(; viscosity=1.0, Ek=1.0),
    )
    small_s_components =
        PlanetaryGeostrophy.surface_pressure_imaginary_coefficient_components(
            small_s_problem, 0.0, 0.0,
        )
    @test all(isfinite, small_s_components)
    @test small_s_components.frictional ≈
        -small_s_components.inviscid rtol=1e-12

    large_s_problem = (
        depth=(x, y) -> 1.0,
        coriolis=(x, y) -> 1.0,
        parameters=(; viscosity=1.0, Ek=1e-8),
    )
    large_s_components =
        PlanetaryGeostrophy.surface_pressure_imaginary_coefficient_components(
            large_s_problem, 0.0, 0.0,
        )
    @test all(isfinite, large_s_components)
end

@testset "Green-function velocity formulations" begin
    comparison = PlanetaryGeostrophy.VelocityFieldPrototype.run_manufactured_test(
        samples=1,
    )
    @test comparison.relative_error < 2e-5

    prototype = PlanetaryGeostrophy.VelocityFieldPrototype
    p = prototype.VelocityParameters(0.02, 0.2, 1.0, 1.0, 1.0, 0.0)
    kernels = prototype.load_symbolic_kernels()
    H(x, y) = 1.0 + 0.05x
    f(x, y) = 1.0 + 0.1y
    f_x(x, y) = 0.0
    f_y(x, y) = 0.1
    gradps(x, y) = 0.02x - 0.01im * y
    τ(x, y) = 1e-3cos(y) + 0im
    B(x, y, z) = 1e-2z + 1e-3x
    bs(x, y) = 1e-3cos(x) * cos(y)
    H_x(x, y) = 0.05
    H_y(x, y) = 0.0
    x, y = 0.1, -0.2
    column_velocity = prototype.velocity_symbolic_column_from_gradps(
        kernels, p, H, f, f_x, f_y, gradps, τ, B, bs, H_x, H_y, x, y,
    )
    for z in (-H(x, y), -0.5H(x, y), 0.0)
        @test column_velocity(z) ≈ prototype.velocity_symbolic_from_gradps(
            kernels, p, H, f, f_x, f_y, gradps, τ, B, bs,
            H_x, H_y, x, y, z,
        )
    end

    test_B(xx, yy, z) = exp(z) * (1 + 0.2xx - 0.1yy)
    test_bs(xx, yy) = 0.03 + 0.01xx + 0.02yy
    coefficients = prototype.buoyancy_transport_sensitivity_coefficients(
        kernels, p, H, f, f_x, f_y, H_x, H_y, x, y, -H(x, y) / 2,
    )
    coefficient_transport = prototype.split_quadgk(
        ξ -> begin
            local_coefficients =
                prototype.buoyancy_transport_sensitivity_coefficients(
                    kernels, p, H, f, f_x, f_y, H_x, H_y, x, y, ξ,
                ).interior
            local_coefficients.value * test_B(x, y, ξ) +
                local_coefficients.gradient * exp(ξ) * (0.2 - 0.1im)
        end,
        -H(x, y),
        0.0,
    )
    bottom_coefficients =
        prototype.buoyancy_transport_sensitivity_coefficients(
            kernels, p, H, f, f_x, f_y, H_x, H_y, x, y, -H(x, y),
        ).interior
    coefficient_transport += (
        bottom_coefficients.gradient *
        complex(H_x(x, y), H_y(x, y)) *
        test_B(x, y, -H(x, y))
    ) + coefficients.surface.value * test_bs(x, y) +
        coefficients.surface.gradient * (0.01 + 0.02im)
    direct_transport = prototype.buoyancy_transport_symbolic(
        kernels, p, H, f, f_x, f_y, test_B, test_bs, H_x, H_y, x, y,
    )
    @test coefficient_transport ≈ direct_transport rtol=2e-7

    # κ(0) in (41): b(0) = b_s, and the generated b_s kernels agree with
    # the closed form, for non-uniform-unit κ and ν.
    pκ = prototype.VelocityParameters(0.02, 0.2, 1.3, 1.7, 1.0, 0.1)
    zero_B(xx, yy, z) = 0.0
    @test prototype.buoyancy_from_forcing(pκ, H, zero_B, test_bs, x, y, 0.0) ≈
        test_bs(x, y)
    ψκ = prototype.ψ(pκ)
    for ξ in (-H(x, y), -0.3)
        @test kernels.inner_Gb_xi(ξ, H(x, y), ψκ, pκ.κ) ≈ prototype.split_quadgk(
            z -> prototype.surface_buoyancy_kernel(z, H(x, y), pκ.κ, ψκ),
            ξ, 0.0, (),
        ) rtol=1e-10
    end
    fκ(xx, yy) = prototype.fcor(pκ, yy)
    ps(xx, yy) = sin(0.7xx) * cos(0.4yy)
    Hκ_x(xx, yy) = 0.05
    @test prototype.velocity_direct(pκ, H, fκ, ps, τ, test_B, test_bs, x, y, -0.4) ≈
        prototype.velocity_symbolic(kernels, pκ, H, fκ, ps, τ, test_B, test_bs,
            Hκ_x, H_y, x, y, -0.4) rtol=2e-5
end

@testset "Vertical velocity reconstruction" begin
    x = [-1.0, 0.0, 1.0]
    y = [-1.0, 0.0, 1.0]
    z = [-1.0, -0.5, 0.0]
    u = [x[i] for i in eachindex(x), j in eachindex(y), k in eachindex(z)]
    v = [-y[j] for i in eachindex(x), j in eachindex(y), k in eachindex(z)]
    wet = trues(size(u))
    depth = ones(length(x), length(y))
    w = PlanetaryGeostrophy.reconstruct_vertical_velocity(u, v, wet, depth, x, y, z)

    @test all(iszero, w)

    fields = PGFields(x, y, z, u, v, w, copy(u), zeros(3, 3), depth, wet)
    @test performance_summary(fields) == (;)

    varying_depth = [
        1.0 0.75 0.5
        1.0 0.75 0.5
        1.0 0.75 0.5
    ]
    varying_wet = [
        z[k] >= -varying_depth[i, j]
        for i in eachindex(x), j in eachindex(y), k in eachindex(z)
    ]
    linear_z = [
        varying_wet[i, j, k] ? z[k] : NaN
        for i in eachindex(x), j in eachindex(y), k in eachindex(z)
    ]
    varying_fields = PGFields(
        x, y, z, linear_z, linear_z, linear_z, linear_z,
        zeros(3, 3), varying_depth, BitArray(varying_wet),
    )
    sigma_slice, label = PlanetaryGeostrophy.terrain_following_slice(
        varying_fields, varying_fields.u, 0.5,
    )
    @test sigma_slice ≈ -0.5 .* varying_depth
    @test label == "z/H=-0.5"
    @test_throws ArgumentError PlanetaryGeostrophy.terrain_following_slice(
        varying_fields, varying_fields.u, 1.0,
    )
    @test_throws ArgumentError plot_solution(
        varying_fields; z_index=2, sigma_level=0.5,
    )
end

@testset "Terrain-following Lagrangian trajectories" begin
    x = [-1.0, 0.0, 1.0]
    y = [-1.0, 0.0, 1.0]
    sigma = [0.0, 0.5, 1.0]
    depth = ones(length(x), length(y))

    divergent_u = [
        x[i] * sigma[k]
        for i in eachindex(x), j in eachindex(y), k in eachindex(sigma)
    ]
    zero_velocity = zeros(size(divergent_u))
    reconstructed_sigma_velocity =
        PlanetaryGeostrophy.reconstruct_sigma_velocity(
            divergent_u,
            zero_velocity,
            depth,
            x,
            y,
            sigma,
        )
    @test all(iszero, reconstructed_sigma_velocity[:, :, 1])
    @test all(iszero, reconstructed_sigma_velocity[:, :, end])
    @test reconstructed_sigma_velocity[2, 2, 2] ≈ 0.125

    affine_u = [
        1 + 2x[i] - 3y[j] + 4sigma[k]
        for i in eachindex(x), j in eachindex(y), k in eachindex(sigma)
    ]
    affine_v = [
        -2 + x[i] + 2y[j] - sigma[k]
        for i in eachindex(x), j in eachindex(y), k in eachindex(sigma)
    ]
    affine_sigma_velocity = [
        0.5sigma[k]
        for i in eachindex(x), j in eachindex(y), k in eachindex(sigma)
    ]
    affine_field = PlanetaryGeostrophy.build_lagrangian_velocity_field(
        x,
        y,
        sigma,
        affine_u,
        affine_v,
        affine_sigma_velocity,
        depth,
    )
    point = (0.25, -0.4, 0.3)
    interpolated = lagrangian_velocity(affine_field, point...)
    @test interpolated.u ≈ 1 + 2point[1] - 3point[2] + 4point[3]
    @test interpolated.v ≈ -2 + point[1] + 2point[2] - point[3]
    @test interpolated.sigma_velocity ≈ 0.5point[3]
    @test interpolated.w ≈ -0.5point[3]
    @test_throws DomainError lagrangian_velocity(
        affine_field, 1.1, 0.0, 0.5,
    )

    rotation_u = [
        -y[j]
        for i in eachindex(x), j in eachindex(y), k in eachindex(sigma)
    ]
    rotation_v = [
        x[i]
        for i in eachindex(x), j in eachindex(y), k in eachindex(sigma)
    ]
    rotation_field = PlanetaryGeostrophy.build_lagrangian_velocity_field(
        x,
        y,
        sigma,
        rotation_u,
        rotation_v,
        zero_velocity,
        depth,
    )
    rotation = compute_trajectories(
        rotation_field,
        [(0.4, 0.0, -0.5)],
        (0.0, 2pi);
        saveat=range(0.0, 2pi; length=101),
        reltol=1e-9,
        abstol=1e-11,
    ).trajectories[1]
    @test rotation.termination == :completed
    @test hypot(last(rotation.x) - 0.4, last(rotation.y)) < 1e-7
    @test maximum(abs.(rotation.z .+ 0.5)) < 1e-12
    short_rotation = compute_trajectories(
        rotation_field,
        [(0.4, 0.0, -0.5)],
        (0.0, 1.0);
        saveat=0.0:0.5:1.0,
    )
    mktempdir() do directory
        animation_path = animate_trajectories(
            short_rotation;
            output=joinpath(directory, "trajectories.gif"),
            fps=2,
        )
        @test isfile(animation_path)
        @test filesize(animation_path) > 0
    end
    @test_throws ArgumentError animate_trajectories(
        short_rotation; frame_stride=0,
    )

    boundary_x = [0.0, 0.5, 1.0]
    eastward_u = ones(3, 3, 3)
    boundary_field = PlanetaryGeostrophy.build_lagrangian_velocity_field(
        boundary_x,
        y,
        sigma,
        eastward_u,
        zeros(3, 3, 3),
        zeros(3, 3, 3),
        ones(3, 3),
    )
    exiting = compute_trajectories(
        boundary_field,
        [(0.25, 0.0, -0.5)],
        (0.0, 2.0);
        saveat=range(0.0, 2.0; length=21),
    ).trajectories[1]
    @test exiting.termination == :x_max
    @test last(exiting.t) ≈ 0.75 atol=1e-8
    @test last(exiting.x) ≈ 1.0 atol=1e-8
    @test_throws DomainError compute_trajectories(
        boundary_field,
        [(0.25, 0.0, 0.0)],
        (0.0, 1.0),
    )
end

@testset "Diagnostic arrow geometry" begin
    vectors = (
        x=[0.0],
        y=[0.0],
        u=[3.0],
        v=[4.0],
    )
    paths = PlanetaryGeostrophy.vector_paths(vectors, 0.02, 0.01)

    shaft_length = hypot(
        paths.shaft_x[2] - paths.shaft_x[1],
        paths.shaft_y[2] - paths.shaft_y[1],
    )
    left_head_length = hypot(
        paths.head_x[2] - paths.head_x[1],
        paths.head_y[2] - paths.head_y[1],
    )
    right_head_length = hypot(
        paths.head_x[2] - paths.head_x[3],
        paths.head_y[2] - paths.head_y[3],
    )

    @test shaft_length ≈ 0.1
    @test left_head_length ≈ hypot(0.01, 0.006)
    @test right_head_length ≈ hypot(0.01, 0.006)
end

@testset "Streamfunction-current convention" begin
    ψ = PGStreamfunction(
        nothing,
        (x, y) -> x^2 + 3y,
        (x, y) -> (2x, 3.0),
        (;),
    )
    current = streamfunction_current(ψ, 2.0, -1.0)

    @test current == (U=-3.0, V=4.0)
    @test current.U * 4.0 + current.V * 3.0 == 0.0
end

@testset "Depth-integrated dynamical diagnostics" begin
    Lx, Ly = 2.0, 1.5
    problem = PGProblem(
        parameters=PGParameters(0.02, 0.2),
        depth=(x, y) -> 1.0,
        depth_x=(x, y) -> 0.0,
        depth_y=(x, y) -> 0.0,
        coriolis=(x, y) -> 1.0 + 0.15y,
        coriolis_x=(x, y) -> 0.0,
        coriolis_y=(x, y) -> 0.15,
        wind_stress=(x, y) -> 1e-3cos(pi * y / Ly) + 0im,
        wind_stress_curl=(x, y) -> 1e-3 * pi / Ly * sin(pi * y / Ly),
        restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3sin(pi * x / Lx),
        surface_buoyancy=(x, y) ->
            2e-3cos(pi * x / Lx) * cos(pi * y / Ly),
        boundary=rectangle_boundary(Lx, Ly),
    )
    @test PlanetaryGeostrophy.analytical_surface_stress_curl(
        problem, 0.1, Ly / 4,
    ) ≈ 1e-3 * pi / Ly * sin(pi / 4)

    mktempdir() do directory
        solution = solve_pg(
            problem;
            mesh_file=joinpath(directory, "pressure.msh"),
            mesh_resolution=0.3,
            pressure_order=1,
            overwrite_mesh=true,
        )
        lagrangian_field = sample_lagrangian_velocity(
            solution,
            [-0.3, 0.0, 0.3],
            [-0.3, 0.0, 0.3],
            [0.0, 0.5, 1.0],
        )
        @test all(iszero, lagrangian_field.sigma_velocity[:, :, 1])
        @test all(iszero, lagrangian_field.sigma_velocity[:, :, end])
        initial_depth = -0.5problem.depth(0.0, 0.0)
        pg_trajectory = compute_trajectories(
            lagrangian_field,
            [(0.0, 0.0, initial_depth)],
            (0.0, 1.0);
            saveat=range(0.0, 1.0; length=5),
        ).trajectories[1]
        @test pg_trajectory.termination == :completed
        @test all(isfinite, pg_trajectory.x)
        @test all(isfinite, pg_trajectory.y)
        @test all(isfinite, pg_trajectory.z)
        @test all(
            -problem.depth(xp, yp) <= zp <= 0
            for (xp, yp, zp) in zip(
                pg_trajectory.x,
                pg_trajectory.y,
                pg_trajectory.z,
            )
        )
        x, y = -0.3, 0.0
        projected_stress = PlanetaryGeostrophy.project_bottom_stress(
            solution; order=solution.pressure.order,
        )
        @test all(isfinite, projected_stress.value(x, y))
        terms = depth_integrated_momentum_terms(solution, x, y)
        bv_terms = barotropic_vorticity_terms(solution, x, y)
        vorticity_terms = depth_integrated_vorticity_terms(solution, x, y)
        gcvb_terms = geostrophic_contour_vorticity_terms(solution, x, y)
        tdvb_terms = transport_divergence_vorticity_terms(solution, x, y)
        pressure_coefficients =
            surface_pressure_equation_coefficients(solution, x, y)
        p = PlanetaryGeostrophy.internal_parameters(problem.parameters)
        column_buoyancy =
            PlanetaryGeostrophy.VelocityFieldPrototype.buoyancy_column_integral(
            solution.kernels,
            p,
            problem.depth,
            problem.restoring_buoyancy,
            problem.surface_buoyancy,
            x,
            y,
        )

        @test bottom_pressure(solution, x, y) ≈
            surface_pressure(solution, x, y) - column_buoyancy
        @test abs(terms.x_residual) < 1e-9
        @test abs(terms.y_residual) < 1e-9
        @test bv_terms.bottom_pressure_torque ≈ 0 atol=1e-12
        @test bv_terms.residual ≈
            bv_terms.transport_divergence +
            bv_terms.planetary_vorticity_advection -
            bv_terms.bottom_pressure_torque -
            bv_terms.bottom_stress_curl -
            bv_terms.surface_stress_curl
        @test vorticity_terms.bottom_vortex_stretching ≈ 0 atol=1e-12
        @test vorticity_terms.residual ≈
            vorticity_terms.planetary_vorticity_advection +
            vorticity_terms.bottom_vortex_stretching -
            vorticity_terms.bottom_stress_curl -
            vorticity_terms.surface_stress_curl
        @test vorticity_terms.residual ≈ bv_terms.residual rtol=1e-6
        @test gcvb_terms.jebar ≈ 0 atol=1e-12
        @test gcvb_terms.residual ≈
            gcvb_terms.geostrophic_contour_advection -
            gcvb_terms.jebar +
            gcvb_terms.bottom_stress_curl +
            gcvb_terms.surface_stress_curl
        @test gcvb_terms.geostrophic_contour_advection ≈
            -vorticity_terms.planetary_vorticity_advection
        @test gcvb_terms.bottom_stress_curl ≈
            vorticity_terms.bottom_stress_curl
        @test gcvb_terms.surface_stress_curl ≈
            vorticity_terms.surface_stress_curl
        @test gcvb_terms.residual ≈ -vorticity_terms.residual
        @test tdvb_terms.residual ≈ -(
            tdvb_terms.bottom_pressure_torque +
            tdvb_terms.baroclinic_torque +
            tdvb_terms.bottom_stress_curl +
            tdvb_terms.surface_stress_curl
        )
        @test tdvb_terms.residual ≈
            bv_terms.residual / problem.coriolis(x, y) rtol=1e-6
        coefficient_step = problem.derivative_step
        A_y = (
            solution.pressure.A_fn((x, y + coefficient_step)) -
            solution.pressure.A_fn((x, y - coefficient_step))
        ) / (2coefficient_step)
        B_x = (
            solution.pressure.B_fn((x + coefficient_step, y)) -
            solution.pressure.B_fn((x - coefficient_step, y))
        ) / (2coefficient_step)
        B_y = (
            solution.pressure.B_fn((x, y + coefficient_step)) -
            solution.pressure.B_fn((x, y - coefficient_step))
        ) / (2coefficient_step)
        baroclinic_B_x = (
            solution.pressure.baroclinic_B_fn((
                x + coefficient_step, y,
            )) -
            solution.pressure.baroclinic_B_fn((
                x - coefficient_step, y,
            ))
        ) / (2coefficient_step)
        baroclinic_B_y = (
            solution.pressure.baroclinic_B_fn((
                x, y + coefficient_step,
            )) -
            solution.pressure.baroclinic_B_fn((
                x, y - coefficient_step,
            ))
        ) / (2coefficient_step)
        stress_B_x = (
            solution.pressure.stress_B_fn((x + coefficient_step, y)) -
            solution.pressure.stress_B_fn((x - coefficient_step, y))
        ) / (2coefficient_step)
        stress_B_y = (
            solution.pressure.stress_B_fn((x, y + coefficient_step)) -
            solution.pressure.stress_B_fn((x, y - coefficient_step))
        ) / (2coefficient_step)
        @test pressure_coefficients.advection_velocity_x ≈ imag(A_y)
        @test pressure_coefficients.advection_velocity_y ≈ 0 atol=1e-10
        @test pressure_coefficients.advection_velocity_x ≈
            pressure_coefficients.inviscid_advection_velocity_x +
            pressure_coefficients.frictional_advection_velocity_x
        @test pressure_coefficients.advection_velocity_y ≈
            pressure_coefficients.inviscid_advection_velocity_y +
            pressure_coefficients.frictional_advection_velocity_y atol=1e-10
        @test pressure_coefficients.diffusivity ≈
            real(solution.pressure.A_fn((x, y)))
        @test pressure_coefficients.baroclinic_source ≈
            -real(baroclinic_B_x) - imag(baroclinic_B_y)
        @test pressure_coefficients.stress_source ≈
            -real(stress_B_x) - imag(stress_B_y)
        @test pressure_coefficients.source ≈ -real(B_x) - imag(B_y)
        @test pressure_coefficients.source ≈
            pressure_coefficients.baroclinic_source +
            pressure_coefficients.stress_source
        curl_y = 0.2
        curl_terms = depth_integrated_vorticity_terms(solution, x, curl_y)
        gcvb_curl_terms =
            geostrophic_contour_vorticity_terms(solution, x, curl_y)
        tdvb_curl_terms =
            transport_divergence_vorticity_terms(solution, x, curl_y)
        expected_surface_curl = (
            problem.parameters.Ek * 1e-3 * pi / Ly *
            sin(pi * curl_y / Ly)
        )
        @test curl_terms.surface_stress_curl ≈ expected_surface_curl rtol=1e-8
        @test gcvb_curl_terms.surface_stress_curl ≈
            expected_surface_curl rtol=1e-8
        local_f = problem.coriolis(x, curl_y)
        expected_tdvb_surface_curl = problem.parameters.Ek * (
            1e-3 * pi / Ly * sin(pi * curl_y / Ly) / local_f +
            real(problem.wind_stress(x, curl_y)) *
            problem.coriolis_y(x, curl_y) / local_f^2
        )
        @test tdvb_curl_terms.surface_stress_curl ≈
            expected_tdvb_surface_curl rtol=1e-8

        z, dz = -0.4, 1e-4
        internal_pressure(zz) = surface_pressure(solution, x, y) -
            PlanetaryGeostrophy.VelocityFieldPrototype.Pi_direct(
                p,
                problem.depth,
                problem.restoring_buoyancy,
                problem.surface_buoyancy,
                x,
                y,
                zz,
            )
        pressure_z = (internal_pressure(z + dz) - internal_pressure(z - dz)) / (2dz)
        @test pressure_z ≈ buoyancy(solution, x, y, z) rtol=2e-6

        velocity_complex(zz) = begin
            uv = horizontal_velocity(solution, x, y, zz)
            uv.u + im * uv.v
        end
        velocity_step = 1e-3
        velocity_zz = (
            -velocity_complex(z + 2velocity_step) +
            16velocity_complex(z + velocity_step) -
            30velocity_complex(z) +
            16velocity_complex(z - velocity_step) -
            velocity_complex(z - 2velocity_step)
        ) / (12velocity_step^2)
        pressure_gradient = solution.pressure.gradps(x, y) -
            PlanetaryGeostrophy.VelocityFieldPrototype.finite_complex_gradient(
                (xx, yy) ->
                    PlanetaryGeostrophy.VelocityFieldPrototype.Pi_direct(
                        p,
                        problem.depth,
                        problem.restoring_buoyancy,
                        problem.surface_buoyancy,
                        xx,
                        yy,
                        z,
                    ),
                x,
                y;
                step=problem.derivative_step,
            )
        momentum_lhs =
            p.Ek * p.ν * velocity_zz -
            im * problem.coriolis(x, y) * velocity_complex(z)
        @test momentum_lhs ≈ pressure_gradient rtol=5e-4

        zero_stress(x, y) = 0.0 + 0.0im
        direct_buoyancy_transport = PlanetaryGeostrophy.transport_forcing(
            p,
            problem.depth,
            problem.coriolis,
            zero_stress,
            problem.restoring_buoyancy,
            problem.surface_buoyancy,
            x,
            y,
        )
        symbolic_buoyancy_transport =
            PlanetaryGeostrophy.VelocityFieldPrototype.buoyancy_transport_symbolic(
                solution.kernels,
                p,
                problem.depth,
                problem.coriolis,
                problem.coriolis_x,
                problem.coriolis_y,
                problem.restoring_buoyancy,
                problem.surface_buoyancy,
                problem.depth_x,
                problem.depth_y,
                x,
                y,
            )
        @test direct_buoyancy_transport ≈ symbolic_buoyancy_transport rtol=2e-5

        direct_buoyancy_stress = PlanetaryGeostrophy.bottom_stress_forcing(
            p,
            problem.depth,
            problem.coriolis,
            zero_stress,
            problem.restoring_buoyancy,
            problem.surface_buoyancy,
            x,
            y,
        )
        symbolic_buoyancy_stress =
            PlanetaryGeostrophy.VelocityFieldPrototype.buoyancy_bottom_stress_symbolic(
                solution.kernels,
                p,
                problem.depth,
                problem.coriolis,
                problem.coriolis_x,
                problem.coriolis_y,
                problem.restoring_buoyancy,
                problem.surface_buoyancy,
                problem.depth_x,
                problem.depth_y,
                x,
                y,
            )
        @test direct_buoyancy_stress ≈ symbolic_buoyancy_stress rtol=2e-4

        coordinates = [-0.3, -0.05, 0.2]
        shape = (3, 3)
        zeros2 = zeros(shape)
        analytical_U = fill(NaN, shape)
        analytical_V = fill(NaN, shape)
        for j in eachindex(coordinates), i in eachindex(coordinates)
            current = depth_integrated_current(
                solution, coordinates[i], coordinates[j],
            )
            analytical_U[i, j] = current.U
            analytical_V[i, j] = current.V
        end
        diagnostics = PGDiagnostics(
            coordinates,
            coordinates,
            zeros2,
            zeros2,
            zeros2,
            zeros2,
            analytical_U,
            analytical_V,
            zeros2,
            trues(shape),
            (;),
        )
        dim = compute_dim_diagnostics(solution, diagnostics)
        bv = compute_barotropic_vorticity_diagnostics(solution, diagnostics)
        vorticity = compute_vorticity_diagnostics(solution, diagnostics)
        gcvb = compute_geostrophic_contour_vorticity_diagnostics(
            solution, diagnostics,
        )
        tdvb = compute_transport_divergence_vorticity_diagnostics(
            solution, diagnostics,
        )
        pressure_equation =
            compute_surface_pressure_equation_diagnostics(
                solution, diagnostics,
            )

        @test size(dim.chi) == shape
        @test maximum(abs, dim.x_residual) < 1e-9
        @test maximum(abs, dim.y_residual) < 1e-9
        @test size(bv.transport_divergence) == shape
        @test count(bv.wet) == 1
        @test all(isnan, bv.bottom_stress_curl[.!bv.wet])
        @test all(isnan, bv.residual[.!bv.wet])
        @test bv.bottom_pressure_torque[bv.wet] ≈ zeros(count(bv.wet))
        @test bv.residual[bv.wet] ≈ (
            bv.transport_divergence[bv.wet] .+
            bv.planetary_vorticity_advection[bv.wet] .-
            bv.bottom_pressure_torque[bv.wet] .-
            bv.bottom_stress_curl[bv.wet] .-
            bv.surface_stress_curl[bv.wet]
        )
        @test performance_summary(bv) == bv.timings
        @test isfinite(bv.timings.relative_l2_residual)
        @test size(vorticity.planetary_vorticity_advection) == shape
        @test count(vorticity.wet) == 1
        @test all(isnan, vorticity.bottom_stress_curl[.!vorticity.wet])
        @test all(isnan, vorticity.residual[.!vorticity.wet])
        @test vorticity.residual[vorticity.wet] ≈ (
            vorticity.planetary_vorticity_advection[vorticity.wet] .+
            vorticity.bottom_vortex_stretching[vorticity.wet] .-
            vorticity.bottom_stress_curl[vorticity.wet] .-
            vorticity.surface_stress_curl[vorticity.wet]
        )
        @test vorticity.residual[vorticity.wet] ≈
            bv.residual[bv.wet]
        # The DIV balance is the BV balance: no-slip makes w_b = 0.
        @test vorticity.bottom_vortex_stretching[vorticity.wet] ==
            zeros(count(vorticity.wet))
        @test vorticity.bottom_stress_curl[vorticity.wet] ≈
            bv.bottom_pressure_torque[bv.wet] .+ bv.bottom_stress_curl[bv.wet]
        @test performance_summary(vorticity) == vorticity.timings
        @test isfinite(vorticity.timings.relative_l2_residual)
        @test size(gcvb.geostrophic_contour_advection) == shape
        @test count(gcvb.wet) == 1
        @test all(isnan, gcvb.jebar[.!gcvb.wet])
        @test all(isnan, gcvb.residual[.!gcvb.wet])
        @test gcvb.residual[gcvb.wet] ≈ (
            gcvb.geostrophic_contour_advection[gcvb.wet] .-
            gcvb.jebar[gcvb.wet] .+
            gcvb.bottom_stress_curl[gcvb.wet] .+
            gcvb.surface_stress_curl[gcvb.wet]
        )
        @test performance_summary(gcvb) == gcvb.timings
        @test isfinite(gcvb.timings.relative_l2_residual)
        @test size(tdvb.bottom_pressure_torque) == shape
        @test count(tdvb.wet) == 1
        @test all(isnan, tdvb.bottom_pressure_torque[.!tdvb.wet])
        @test all(isnan, tdvb.residual[.!tdvb.wet])
        @test tdvb.residual[tdvb.wet] ≈ -(
            tdvb.bottom_pressure_torque[tdvb.wet] .+
            tdvb.baroclinic_torque[tdvb.wet] .+
            tdvb.bottom_stress_curl[tdvb.wet] .+
            tdvb.surface_stress_curl[tdvb.wet]
        )
        @test performance_summary(tdvb) == tdvb.timings
        @test isfinite(tdvb.timings.relative_l2_residual)
        @test size(pressure_equation.advection_velocity_x) == shape
        @test size(pressure_equation.inviscid_advection_velocity_x) == shape
        @test size(pressure_equation.frictional_advection_velocity_x) == shape
        @test count(pressure_equation.wet) == 1
        @test all(
            isnan,
            pressure_equation.advection_velocity_x[.!pressure_equation.wet],
        )
        @test all(isnan, pressure_equation.source[.!pressure_equation.wet])
        center = CartesianIndex(2, 2)
        center_A_y = (
            solution.pressure.A_fn((coordinates[2], coordinates[3])) -
            solution.pressure.A_fn((coordinates[2], coordinates[1]))
        ) / (coordinates[3] - coordinates[1])
        center_A_x = (
            solution.pressure.A_fn((coordinates[3], coordinates[2])) -
            solution.pressure.A_fn((coordinates[1], coordinates[2]))
        ) / (coordinates[3] - coordinates[1])
        center_B_x = (
            solution.pressure.B_fn((coordinates[3], coordinates[2])) -
            solution.pressure.B_fn((coordinates[1], coordinates[2]))
        ) / (coordinates[3] - coordinates[1])
        center_B_y = (
            solution.pressure.B_fn((coordinates[2], coordinates[3])) -
            solution.pressure.B_fn((coordinates[2], coordinates[1]))
        ) / (coordinates[3] - coordinates[1])
        center_stress_B_x = (
            solution.pressure.stress_B_fn((
                coordinates[3], coordinates[2],
            )) -
            solution.pressure.stress_B_fn((
                coordinates[1], coordinates[2],
            ))
        ) / (coordinates[3] - coordinates[1])
        center_stress_B_y = (
            solution.pressure.stress_B_fn((
                coordinates[2], coordinates[3],
            )) -
            solution.pressure.stress_B_fn((
                coordinates[2], coordinates[1],
            ))
        ) / (coordinates[3] - coordinates[1])
        @test pressure_equation.advection_velocity_x[center] ≈
            imag(center_A_y)
        @test pressure_equation.advection_velocity_y[center] ≈
            -imag(center_A_x)
        @test pressure_equation.advection_velocity_x[center] ≈
            pressure_equation.inviscid_advection_velocity_x[center] +
            pressure_equation.frictional_advection_velocity_x[center]
        @test pressure_equation.advection_velocity_y[center] ≈
            pressure_equation.inviscid_advection_velocity_y[center] +
            pressure_equation.frictional_advection_velocity_y[center]
        @test pressure_equation.diffusivity[center] ≈
            real(solution.pressure.A_fn((coordinates[2], coordinates[2])))
        @test pressure_equation.source[center] ≈
            -real(center_B_x) - imag(center_B_y) rtol=3e-5
        @test pressure_equation.stress_source[center] ≈
            -real(center_stress_B_x) - imag(center_stress_B_y)
        @test pressure_equation.baroclinic_source[center] +
            pressure_equation.stress_source[center] ≈
            pressure_equation.source[center]
        @test pressure_equation.residual[center] ≈ (
            (analytical_U[3, 2] - analytical_U[1, 2]) /
            (coordinates[3] - coordinates[1]) +
            (analytical_V[2, 3] - analytical_V[2, 1]) /
            (coordinates[3] - coordinates[1])
        )
        @test performance_summary(pressure_equation) ==
            pressure_equation.timings
        @test isfinite(pressure_equation.timings.relative_l2_residual)

        balance_path = joinpath(directory, "dim_balance.png")
        bv_path = joinpath(directory, "barotropic_vorticity_balance.png")
        vorticity_path = joinpath(directory, "vorticity_balance.png")
        gcvb_path = joinpath(directory, "gcvb_balance.png")
        tdvb_path = joinpath(directory, "tdvb_balance.png")
        pressure_equation_path =
            joinpath(directory, "surface_pressure_equation.png")
        chi_path = joinpath(directory, "chi.png")
        pressure_path = joinpath(directory, "bottom_pressure.png")
        balance_figure = plot_dim_balance(dim; output=balance_path)
        x_balance_figure = plot_dim_balance(dim; component=:x)
        bv_figure = plot_barotropic_vorticity_balance(bv; output=bv_path)
        vorticity_figure = plot_vorticity_balance(
            vorticity; output=vorticity_path,
        )
        gcvb_figure = plot_geostrophic_contour_vorticity_balance(
            gcvb; output=gcvb_path,
        )
        tdvb_figure = plot_transport_divergence_vorticity_balance(
            tdvb; output=tdvb_path,
        )
        pressure_equation_figure =
            plot_surface_pressure_equation_diagnostics(
                pressure_equation; output=pressure_equation_path,
            )
        plot_baroclinic_potential_energy(dim; output=chi_path)
        plot_bottom_pressure(dim; output=pressure_path)

        panel_size(figure, index) = Tuple(
            length.value for length in figure.subplots[index].plotarea.a
        )
        x_panel_sizes = [
            panel_size(balance_figure, index) for index in 1:6
        ]
        y_panel_sizes = [
            panel_size(balance_figure, index) for index in 8:13
        ]
        @test all(
            all(isapprox.(dimensions, first(x_panel_sizes)))
            for dimensions in x_panel_sizes
        )
        @test all(
            all(isapprox.(dimensions, first(y_panel_sizes)))
            for dimensions in y_panel_sizes
        )
        @test all(
            all(isapprox.(size, panel_size(x_balance_figure, 1)))
            for size in (panel_size(x_balance_figure, index) for index in 1:6)
        )
        @test all(
            all(isapprox.(size, panel_size(bv_figure, 1)))
            for size in (panel_size(bv_figure, index) for index in 1:5)
        )
        @test all(
            all(isapprox.(size, panel_size(vorticity_figure, 1)))
            for size in (panel_size(vorticity_figure, index) for index in 1:5)
        )
        @test all(
            all(isapprox.(size, panel_size(gcvb_figure, 1)))
            for size in (panel_size(gcvb_figure, index) for index in 1:5)
        )
        @test all(
            all(isapprox.(size, panel_size(tdvb_figure, 1)))
            for size in (panel_size(tdvb_figure, index) for index in 1:5)
        )
        @test all(
            all(isapprox.(size, panel_size(pressure_equation_figure, 1)))
            for size in (
                panel_size(pressure_equation_figure, index) for index in 1:6
            )
        )
        @test isfile(balance_path)
        @test isfile(bv_path)
        @test isfile(vorticity_path)
        @test isfile(gcvb_path)
        @test isfile(tdvb_path)
        @test isfile(pressure_equation_path)
        @test isfile(chi_path)
        @test isfile(pressure_path)

        variable_problem = PGProblem(
            parameters=problem.parameters,
            depth=(x, y) -> 1.0 + 0.04x + 0.03y,
            depth_x=(x, y) -> 0.04,
            depth_y=(x, y) -> 0.03,
            coriolis=problem.coriolis,
            coriolis_x=problem.coriolis_x,
            coriolis_y=problem.coriolis_y,
            wind_stress=problem.wind_stress,
            wind_stress_curl=problem.wind_stress_curl,
            restoring_buoyancy=problem.restoring_buoyancy,
            surface_buoyancy=problem.surface_buoyancy,
            boundary=problem.boundary,
        )
        # Use the same mesh as `solution`. The mesh file is regenerated when
        # the requested resolution differs from the one it was written with.
        variable_solution = solve_pg(
            variable_problem;
            mesh_file=joinpath(directory, "pressure.msh"),
            mesh_resolution=0.3,
            pressure_order=1,
        )
        variable_terms = depth_integrated_momentum_terms(
            variable_solution,
            x,
            y,
        )
        variable_bv_terms = barotropic_vorticity_terms(
            variable_solution,
            x,
            y,
        )
        variable_vorticity_terms = depth_integrated_vorticity_terms(
            variable_solution,
            x,
            y,
        )
        variable_gcvb_terms = geostrophic_contour_vorticity_terms(
            variable_solution,
            x,
            y,
        )
        variable_tdvb_terms = transport_divergence_vorticity_terms(
            variable_solution,
            x,
            y,
        )
        @test abs(variable_terms.x_residual) < 1e-8
        @test abs(variable_terms.y_residual) < 1e-8
        @test variable_bv_terms.residual ≈
            variable_bv_terms.transport_divergence +
            variable_bv_terms.planetary_vorticity_advection -
            variable_bv_terms.bottom_pressure_torque -
            variable_bv_terms.bottom_stress_curl -
            variable_bv_terms.surface_stress_curl
        @test abs(variable_vorticity_terms.bottom_vortex_stretching) < 1e-12
        @test variable_vorticity_terms.residual ≈
            variable_vorticity_terms.planetary_vorticity_advection +
            variable_vorticity_terms.bottom_vortex_stretching -
            variable_vorticity_terms.bottom_stress_curl -
            variable_vorticity_terms.surface_stress_curl
        @test variable_vorticity_terms.residual ≈
            variable_bv_terms.residual rtol=1e-6
        @test variable_gcvb_terms.residual ≈
            variable_gcvb_terms.geostrophic_contour_advection -
            variable_gcvb_terms.jebar +
            variable_gcvb_terms.bottom_stress_curl +
            variable_gcvb_terms.surface_stress_curl
        derivative_step = variable_problem.derivative_step
        surface_stress_over_depth(xx, yy) =
            variable_problem.wind_stress(xx, yy) /
            variable_problem.depth(xx, yy)
        surface_stress_x = (
            surface_stress_over_depth(x + derivative_step, y) -
            surface_stress_over_depth(x - derivative_step, y)
        ) / (2derivative_step)
        surface_stress_y = (
            surface_stress_over_depth(x, y + derivative_step) -
            surface_stress_over_depth(x, y - derivative_step)
        ) / (2derivative_step)
        expected_depth_weighted_surface_curl =
            imag(surface_stress_x) - real(surface_stress_y)
        @test PlanetaryGeostrophy.analytical_surface_stress_over_depth_curl(
            variable_problem, x, y,
        ) ≈ expected_depth_weighted_surface_curl rtol=1e-8

        χ_x = (
            baroclinic_potential_energy(
                variable_solution, x + derivative_step, y,
            ) -
            baroclinic_potential_energy(
                variable_solution, x - derivative_step, y,
            )
        ) / (2derivative_step)
        χ_y = (
            baroclinic_potential_energy(
                variable_solution, x, y + derivative_step,
            ) -
            baroclinic_potential_energy(
                variable_solution, x, y - derivative_step,
            )
        ) / (2derivative_step)
        H = variable_problem.depth(x, y)
        expected_jebar = (
            variable_problem.depth_y(x, y) * χ_x -
            variable_problem.depth_x(x, y) * χ_y
        ) / H^2
        @test variable_gcvb_terms.jebar ≈ expected_jebar rtol=1e-8
        surface_stress_over_coriolis(xx, yy) =
            variable_problem.wind_stress(xx, yy) /
            variable_problem.coriolis(xx, yy)
        surface_stress_over_coriolis_x = (
            surface_stress_over_coriolis(x + derivative_step, y) -
            surface_stress_over_coriolis(x - derivative_step, y)
        ) / (2derivative_step)
        surface_stress_over_coriolis_y = (
            surface_stress_over_coriolis(x, y + derivative_step) -
            surface_stress_over_coriolis(x, y - derivative_step)
        ) / (2derivative_step)
        expected_coriolis_weighted_surface_curl =
            imag(surface_stress_over_coriolis_x) -
            real(surface_stress_over_coriolis_y)
        @test PlanetaryGeostrophy.analytical_surface_stress_over_coriolis_curl(
            variable_problem, x, y,
        ) ≈ expected_coriolis_weighted_surface_curl rtol=1e-8

        pb_x = (
            bottom_pressure(variable_solution, x + derivative_step, y) -
            bottom_pressure(variable_solution, x - derivative_step, y)
        ) / (2derivative_step)
        pb_y = (
            bottom_pressure(variable_solution, x, y + derivative_step) -
            bottom_pressure(variable_solution, x, y - derivative_step)
        ) / (2derivative_step)
        f = variable_problem.coriolis(x, y)
        f_x = variable_problem.coriolis_x(x, y)
        f_y = variable_problem.coriolis_y(x, y)
        H_over_f_x = (variable_problem.depth_x(x, y) * f - H * f_x) / f^2
        H_over_f_y = (variable_problem.depth_y(x, y) * f - H * f_y) / f^2
        @test variable_tdvb_terms.bottom_pressure_torque ≈
            pb_x * H_over_f_y - pb_y * H_over_f_x rtol=1e-8
        @test variable_tdvb_terms.baroclinic_torque ≈
            (-f_y * χ_x + f_x * χ_y) / f^2 rtol=1e-8
        @test variable_tdvb_terms.residual ≈ -(
            variable_tdvb_terms.bottom_pressure_torque +
            variable_tdvb_terms.baroclinic_torque +
            variable_tdvb_terms.bottom_stress_curl +
            variable_tdvb_terms.surface_stress_curl
        )
        @test variable_tdvb_terms.residual ≈
            variable_bv_terms.residual /
            variable_problem.coriolis(x, y) rtol=1e-6
        variable_pressure_coefficients =
            surface_pressure_equation_coefficients(
                variable_solution, x, y,
            )
        variable_A_x = (
            variable_solution.pressure.A_fn((x + derivative_step, y)) -
            variable_solution.pressure.A_fn((x - derivative_step, y))
        ) / (2derivative_step)
        variable_A_y = (
            variable_solution.pressure.A_fn((x, y + derivative_step)) -
            variable_solution.pressure.A_fn((x, y - derivative_step))
        ) / (2derivative_step)
        @test variable_pressure_coefficients.advection_velocity_x ≈
            imag(variable_A_y)
        @test variable_pressure_coefficients.advection_velocity_y ≈
            -imag(variable_A_x)
        @test variable_pressure_coefficients.advection_velocity_x ≈
            variable_pressure_coefficients.inviscid_advection_velocity_x +
            variable_pressure_coefficients.frictional_advection_velocity_x
        @test variable_pressure_coefficients.advection_velocity_y ≈
            variable_pressure_coefficients.inviscid_advection_velocity_y +
            variable_pressure_coefficients.frictional_advection_velocity_y
        @test !isapprox(
            variable_pressure_coefficients.advection_velocity_y,
            0.0;
            atol=1e-8,
        )

        sensitivity = solve_forcing_sensitivities(
            problem;
            wind_stress_y=(x, y) -> 1e-3cos(pi * y / Ly),
            mesh_file=joinpath(directory, "pressure.msh"),
            mesh_resolution=0.3,
            pressure_order=1,
        )
        velocity_response =
            horizontal_velocity_sensitivity(sensitivity, x, y, z)
        transport_response =
            depth_integrated_current_sensitivity(sensitivity, x, y)
        reconstructed_velocity = (
            u=velocity_response.restoring_buoyancy.u +
                velocity_response.surface_buoyancy.u +
                velocity_response.wind_stress_x.u,
            v=velocity_response.restoring_buoyancy.v +
                velocity_response.surface_buoyancy.v +
                velocity_response.wind_stress_x.v,
        )
        reconstructed_transport = (
            U=transport_response.restoring_buoyancy.U +
                transport_response.surface_buoyancy.U +
                transport_response.wind_stress_x.U,
            V=transport_response.restoring_buoyancy.V +
                transport_response.surface_buoyancy.V +
                transport_response.wind_stress_x.V,
        )
        @test reconstructed_velocity.u ≈ horizontal_velocity(
            solution, x, y, z,
        ).u rtol=1e-7
        @test reconstructed_velocity.v ≈ horizontal_velocity(
            solution, x, y, z,
        ).v rtol=1e-7
        @test reconstructed_transport.U ≈ depth_integrated_current(
            solution, x, y,
        ).U rtol=1e-7
        @test reconstructed_transport.V ≈ depth_integrated_current(
            solution, x, y,
        ).V rtol=1e-7
        @test hypot(
            transport_response.wind_stress_y.U,
            transport_response.wind_stress_y.V,
        ) > 0
        @test performance_summary(sensitivity) == sensitivity.timings

        Ω = Triangulation(solution.pressure.model)
        dΩ = Measure(Ω, 2solution.pressure.order)
        transport_field(point) = (
            solution.pressure.A_fn(point) *
            solution.pressure.gradps(point[1], point[2]) +
            solution.pressure.B_fn(point)
        )
        adjoints = Dict{Symbol,Any}()
        for (name, cost, cost_gradient, expected_mode) in (
            (:linear_u, (U, V) -> U, (1.0, 0.0), :constant),
            (:linear_v, (U, V) -> V, nothing, :forwarddiff),
            (
                :quadratic,
                (U, V) -> 0.5 * (U^2 + V^2),
                (U, V) -> (U, V),
                :analytic,
            ),
        )
            adjoint = PlanetaryGeostrophy.solve_surface_pressure_adjoint(
                solution.pressure,
                x,
                y;
                cost,
                cost_gradient,
                target_radius=0.15,
            )
            adjoints[name] = adjoint
            @test adjoint.cost_gradient_mode == expected_mode
            averaged_transport = sum(∫(
                point -> begin
                    transport = transport_field(point)
                    derivatives =
                        adjoint.cost_derivatives(point[1], point[2])
                    adjoint.weight(point[1], point[2]) * (
                        derivatives.cost_U * real(transport) +
                        derivatives.cost_V * imag(transport)
                    )
                end
            )dΩ)
            adjoint_transport = sum(∫(
                point -> begin
                    forcing = solution.pressure.B_fn(point)
                    gradient = adjoint.gradient(point[1], point[2])
                    derivatives =
                        adjoint.cost_derivatives(point[1], point[2])
                    (
                        adjoint.weight(point[1], point[2]) *
                        (
                            derivatives.cost_U * real(forcing) +
                            derivatives.cost_V * imag(forcing)
                        ) -
                        gradient[1] * real(forcing) -
                        gradient[2] * imag(forcing)
                    )
                end
            )dΩ)
            expected_cost = sum(∫(
                point -> begin
                    transport = transport_field(point)
                    adjoint.weight(point[1], point[2]) *
                    cost(real(transport), imag(transport))
                end
            )dΩ)
            @test adjoint_transport ≈ averaged_transport rtol=1e-10
            @test adjoint.cost_value ≈ expected_cost rtol=1e-10
        end
        explicit_v_adjoint =
            PlanetaryGeostrophy.solve_surface_pressure_adjoint(
                solution.pressure,
                x,
                y;
                cost=(U, V) -> V,
                cost_gradient=(0.0, 1.0),
                cost_value=adjoints[:linear_v].cost_value,
                target_radius=0.15,
            )
        @test explicit_v_adjoint.cost_gradient_mode == :constant
        @test explicit_v_adjoint.cost_value == adjoints[:linear_v].cost_value
        @test explicit_v_adjoint.gradient(x, y) ≈
            adjoints[:linear_v].gradient(x, y) rtol=1e-10
        section_width = 0.1
        section_weight(xx, yy) =
            -0.5 <= xx <= 0.0 ?
            exp(-(yy / section_width)^2 / 2) /
                (sqrt(2π) * section_width) : 0.0
        section_adjoint =
            PlanetaryGeostrophy.solve_surface_pressure_adjoint(
                solution.pressure,
                -0.25,
                0.0;
                cost=(U, V) -> V,
                cost_gradient=(0.0, 1.0),
                cost_value=1.25,
                target_weight=section_weight,
                target_radius=section_width,
            )
        @test section_adjoint.weight(-0.25, 0.0) ≈
            1 / (sqrt(2π) * section_width)
        @test section_adjoint.weight(0.25, 0.0) == 0.0
        @test section_adjoint.cost_value == 1.25

        adjoint_sensitivity = compute_transport_adjoint_sensitivity(
            solution,
            range(-0.75, 0.15; length=7),
            range(-0.45, 0.45; length=7),
            [-0.75, -0.5, -0.25];
            target=(x, y),
            cost=(U, V) -> hypot(U, V),
            cost_gradient=(U, V) -> (U, V) ./ hypot(U, V),
            target_weight=section_weight,
            target_metadata=(
                kind=:zonal_section,
                x=-0.25,
                y=0.0,
                x_limits=(-0.5, 0.0),
                width=section_width,
            ),
            cost_label="transport speed",
            target_radius=0.15,
        )
        @test size(adjoint_sensitivity.wind_stress_x) == (7, 7)
        @test size(adjoint_sensitivity.surface_buoyancy) == (7, 7)
        @test size(adjoint_sensitivity.restoring_buoyancy) == (7, 7, 3)
        @test size(adjoint_sensitivity.surface_buoyancy_local) == (7, 7)
        @test size(adjoint_sensitivity.surface_buoyancy_gradient_x) == (7, 7)
        @test size(adjoint_sensitivity.surface_buoyancy_gradient_y) == (7, 7)
        @test size(adjoint_sensitivity.restoring_buoyancy_local) == (7, 7, 3)
        @test size(adjoint_sensitivity.restoring_buoyancy_gradient_x) ==
            (7, 7, 3)
        @test size(adjoint_sensitivity.restoring_buoyancy_gradient_y) ==
            (7, 7, 3)
        @test any(isfinite, adjoint_sensitivity.wind_stress_x)
        @test any(isfinite, adjoint_sensitivity.wind_stress_y)
        @test any(isfinite, adjoint_sensitivity.surface_buoyancy)
        @test any(isfinite, adjoint_sensitivity.restoring_buoyancy)
        @test any(isfinite, adjoint_sensitivity.surface_buoyancy_local)
        @test any(isfinite, adjoint_sensitivity.surface_buoyancy_gradient_x)
        @test any(isfinite, adjoint_sensitivity.surface_buoyancy_gradient_y)
        @test any(isfinite, adjoint_sensitivity.restoring_buoyancy_local)
        @test adjoint_sensitivity.target.kind == :zonal_section
        @test any(
            isfinite,
            adjoint_sensitivity.restoring_buoyancy_gradient_x,
        )
        @test any(
            isfinite,
            adjoint_sensitivity.restoring_buoyancy_gradient_y,
        )
        @test adjoint_sensitivity.cost_label == "transport speed"
        @test isfinite(adjoint_sensitivity.cost_value)
        identity = transport_adjoint_identity(adjoint_sensitivity, solution)
        @test identity.forward ≈ identity.adjoint rtol=1e-10
        @test identity.forward_pressure ≈ identity.adjoint_pressure rtol=1e-10
        @test identity.relative_error < 1e-10
        @test performance_summary(adjoint_sensitivity) ==
            adjoint_sensitivity.timings

        velocity_sensitivity_path =
            joinpath(directory, "velocity_sensitivity.png")
        transport_sensitivity_path =
            joinpath(directory, "transport_sensitivity.png")
        adjoint_sensitivity_path =
            joinpath(directory, "transport_adjoint_sensitivity.png")
        plot_velocity_sensitivity(
            sensitivity,
            x,
            y,
            range(-problem.depth(x, y), 0.0; length=9);
            output=velocity_sensitivity_path,
        )
        plot_transport_sensitivity(
            sensitivity, x, y; output=transport_sensitivity_path,
        )
        plot_transport_adjoint_sensitivity(
            adjoint_sensitivity;
            output=adjoint_sensitivity_path,
        )
        @test isfile(velocity_sensitivity_path)
        @test isfile(transport_sensitivity_path)
        @test isfile(adjoint_sensitivity_path)
    end
end

@testset "Fast single-integral path agrees with the symbolic-kernel path" begin
    # Sloping bottom, a seamount, beta, and depth-dependent horizontal
    # buoyancy forcing exercise every term in (51), (54), and (57).
    problem = PGProblem(
        parameters=PGParameters(0.02, 0.2),
        depth=(x, y) -> 0.8 + 0.25x + 0.1y +
            0.15exp(-((x - 0.2)^2 + y^2) / 0.1),
        coriolis=(x, y) -> 1.0 + 0.15y,
        wind_stress=(x, y) -> 1e-3cos(pi * y / 1.5) + 2e-4im * sin(x),
        restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3x * exp(2z),
        surface_buoyancy=(x, y) -> 2e-3y,
        boundary=ellipse_boundary(1.0, 0.75; points=32),
    )
    @test_throws ArgumentError ColumnQuadrature(points=1)

    mktempdir() do directory
        mesh_file = joinpath(directory, "pressure.msh")
        fast = solve_pg(
            problem; mesh_file, mesh_resolution=0.2, overwrite_mesh=true,
        )
        symbolic = solve_pg(problem; mesh_file, mesh_resolution=0.2, method=:symbolic)
        nodal = solve_pg(
            problem; mesh_file, mesh_resolution=0.2, forcing_assembly=:nodal,
        )
        @test fast.pressure.method === :fast
        @test symbolic.pressure.method === :symbolic
        @test_throws ArgumentError solve_pg(
            problem; mesh_file, mesh_resolution=0.2, method=:other,
        )

        relative(a, b) = abs(a - b) / abs(b)
        for (x, y) in ((0.3, 0.2), (-0.5, -0.1), (0.0, 0.4))
            @test relative(fast.pressure.B_fn((x, y)), symbolic.pressure.B_fn((x, y))) < 1e-7
            @test relative(fast.pressure.T_fn((x, y)), symbolic.pressure.T_fn((x, y))) < 1e-7
            @test relative(
                surface_pressure(fast, x, y), surface_pressure(symbolic, x, y),
            ) < 1e-7
            fast_current = depth_integrated_current(fast, x, y)
            symbolic_current = depth_integrated_current(symbolic, x, y)
            @test relative(
                complex(fast_current.U, fast_current.V),
                complex(symbolic_current.U, symbolic_current.V),
            ) < 1e-6
            # Nodal assembly adds an O(h^3) interpolation error in 𝔅; on this
            # coarse h=0.2 mesh it is about 1-2%, so this is a sanity bound.
            nodal_current = depth_integrated_current(nodal, x, y)
            @test relative(
                complex(nodal_current.U, nodal_current.V),
                complex(symbolic_current.U, symbolic_current.V),
            ) < 5e-2

            H = problem.depth(x, y)
            levels = range(-H, 0.0; length=9)
            velocity(solution, z) =
                complex(values(horizontal_velocity(solution, x, y, z))...)
            scale = maximum(abs(velocity(symbolic, z)) for z in levels)
            @test maximum(
                abs(velocity(fast, z) - velocity(symbolic, z)) for z in levels
            ) < 1e-6scale
            @test maximum(
                abs(buoyancy(fast, x, y, z) - buoyancy(symbolic, x, y, z))
                for z in levels
            ) < 1e-10
            @test bottom_pressure(fast, x, y) ≈ bottom_pressure(symbolic, x, y) rtol=1e-8
            @test baroclinic_potential_energy(fast, x, y) ≈
                  baroclinic_potential_energy(symbolic, x, y) rtol=1e-8

            # τ_b = S (∂x + i∂y)p_b + T*, with T* = T + S (∂x + i∂y)∫b dz
            # from the buoyancy alone, (45) and (52).
            p_internal = PlanetaryGeostrophy.internal_parameters(problem.parameters)
            S = PlanetaryGeostrophy.bottom_stress_coefficient(
                p_internal, H, problem.coriolis(x, y),
            )
            for solution in (fast, symbolic)
                column_gradient =
                    PlanetaryGeostrophy.VelocityFieldPrototype.finite_complex_gradient(
                        (xx, yy) -> surface_pressure(solution, xx, yy) -
                                    bottom_pressure(solution, xx, yy),
                        x, y,
                    )
                gradpb = solution.pressure.gradps(x, y) - column_gradient
                stress = bottom_stress(solution, x, y)
                @test S * gradpb + bottom_pressure_stress_forcing(solution, x, y) ≈
                      complex(stress.x, stress.y) rtol=1e-6
            end
            @test bottom_pressure_stress_forcing(fast, x, y) ≈
                  bottom_pressure_stress_forcing(symbolic, x, y) rtol=1e-6
        end

        xs, ys = range(-0.8, 0.8; length=9), range(-0.6, 0.6; length=7)
        zs = range(-1.2, 0.0; length=7)
        fast_fields = sample_solution(fast, xs, ys, zs)
        symbolic_fields = sample_solution(symbolic, xs, ys, zs)
        @test fast_fields.wet == symbolic_fields.wet
        wet = fast_fields.wet
        @test maximum(abs, (fast_fields.u .- symbolic_fields.u)[wet]) <
              1e-6maximum(abs, symbolic_fields.u[wet])
        @test maximum(abs, (fast_fields.w .- symbolic_fields.w)[wet]) <
              1e-6maximum(abs, symbolic_fields.w[wet])
    end
end

@testset "Cell-averaged vorticity budgets close" begin
    # Sloping bottom with a seamount: centered differences of the
    # finite-element bottom stress do not close here, but Stokes cell
    # averages should, up to the finite-element mass-conservation error.
    problem = PGProblem(
        parameters=PGParameters(0.01, 0.2),
        depth=(x, y) -> 0.8 + 0.25x + 0.1y +
            0.3exp(-((x - 0.2)^2 + y^2) / 0.05),
        coriolis=(x, y) -> 1.0 + 0.15y,
        wind_stress=(x, y) -> 1e-3cos(pi * y / 1.5) + 2e-4im * sin(x),
        restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3x * exp(2z),
        surface_buoyancy=(x, y) -> 2e-3y,
        boundary=ellipse_boundary(1.0, 0.75; points=32),
    )
    @test_throws ArgumentError CellBudgetQuadrature(edge_panels=0)

    mktempdir() do directory
        solution = solve_pg(
            problem;
            mesh_file=joinpath(directory, "pressure.msh"),
            mesh_resolution=0.05,
            overwrite_mesh=true,
        )
        diagnostics = compute_diagnostics(
            solution, range(-0.8, 0.8; length=17), range(-0.6, 0.6; length=13),
        )
        @test_throws ArgumentError compute_barotropic_vorticity_diagnostics(
            solution, diagnostics; method=:other,
        )
        budgets(method) = (
            bv=compute_barotropic_vorticity_diagnostics(solution, diagnostics; method),
            vorticity=compute_vorticity_diagnostics(solution, diagnostics; method),
            gcvb=compute_geostrophic_contour_vorticity_diagnostics(
                solution, diagnostics; method,
            ),
            tdvb=compute_transport_divergence_vorticity_diagnostics(
                solution, diagnostics; method,
            ),
        )
        cell = budgets(:cell_average)
        centered = budgets(:centered)
        for name in keys(cell)
            budget = cell[name]
            residual = budget.timings.relative_l2_residual
            @test count(budget.wet) > 20
            @test all(isnan, budget.residual[.!budget.wet])
            @test all(isfinite, budget.residual[budget.wet])
            # The residual is the cell-averaged finite-element mass error.
            # With U·n = 0 at this basin's vertical coastal wall, a thin
            # coastal frictional layer forms that mesh 0.05 resolves only
            # roughly, hence the loose bound; it falls as the mesh is refined.
            @test residual < 0.1
            @test residual < centered[name].timings.relative_l2_residual
        end
        @test cell.bv.wet == cell.gcvb.wet == cell.tdvb.wet
        @test cell.vorticity.residual[cell.vorticity.wet] ≈
              cell.bv.residual[cell.bv.wet]
        @test cell.vorticity.wet == cell.bv.wet
        # No-slip makes the bottom vortex stretching f w_b vanish, which is why
        # the DIV balance reduces to the BV balance.
        stretching = [PlanetaryGeostrophy.bottom_stretching(solution, x, y)
                      for x in (-0.5, 0.0, 0.4), y in (-0.3, 0.0, 0.25)]
        planetary_scale = maximum(abs, filter(isfinite, cell.bv.planetary_vorticity_advection))
        @test maximum(abs, stretching) < 1e-10 * planetary_scale
        @test cell.bv.transport_divergence[cell.bv.wet] ==
              zeros(count(cell.bv.wet))
    end
end

@testset "No-normal-flow coastal condition" begin
    # A vertical coastal wall (H = 0.4 at the coast) with stress and
    # buoyancy forcing: p_s = 0 lets frictional and buoyancy-forced transport
    # cross the coast, while the natural condition U·n = 0 does not.
    ax, ay = 1.0, 0.75
    problem = PGProblem(
        parameters=PGParameters(0.01, 0.2),
        depth=(x, y) -> 0.4 + 0.6exp(-(x^2 + y^2) / 0.2),
        coriolis=(x, y) -> 1.0 + 0.15y,
        wind_stress=(x, y) -> 1e-3cos(pi * y / 1.5) + 2e-4im * sin(x),
        restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3x * exp(2z),
        surface_buoyancy=(x, y) -> 2e-3y,
        boundary=ellipse_boundary(ax, ay; points=48),
    )
    mktempdir() do directory
        mesh_file = joinpath(directory, "pressure.msh")
        @test_throws ArgumentError solve_pg(
            problem; mesh_file, mesh_resolution=0.08, overwrite_mesh=true,
            coastal_condition=:other,
        )
        natural = solve_pg(
            problem; mesh_file, mesh_resolution=0.08, overwrite_mesh=true,
        )
        dirichlet = solve_pg(
            problem; mesh_file, mesh_resolution=0.08,
            coastal_condition=:dirichlet,
        )
        @test natural.pressure.coastal_condition === :no_normal_flow
        @test dirichlet.pressure.coastal_condition === :dirichlet

        function coastal_ratio(solution, radius)
            normal, tangential = Float64[], Float64[]
            for θ in range(0, 2π; length=257)[1:end-1]
                x, y = radius * ax * cos(θ), radius * ay * sin(θ)
                nx, ny = cos(θ) / ax, sin(θ) / ay
                norm = hypot(nx, ny)
                current = depth_integrated_current(solution, x, y)
                push!(normal, (current.U * nx + current.V * ny) / norm)
                push!(tangential, (-current.U * ny + current.V * nx) / norm)
            end
            return sqrt(sum(abs2, normal) / sum(abs2, tangential))
        end
        @test coastal_ratio(natural, 0.99) < 0.1
        @test coastal_ratio(natural, 0.99) < 0.5coastal_ratio(dirichlet, 0.99)

        # Mean-zero surface pressure on the natural-condition solution.
        Ω = Triangulation(natural.pressure.model)
        dΩ = Measure(Ω, 4)
        @test abs(sum(∫(natural.pressure.psurf)dΩ)) < 1e-10

        # The direct transport and the streamfunction transport agree better
        # once the transport does not cross the coast.
        xs, ys = range(-0.85, 0.85; length=13), range(-0.6, 0.6; length=11)
        natural_diagnostics = compute_diagnostics(natural, xs, ys)
        dirichlet_diagnostics = compute_diagnostics(dirichlet, xs, ys)
        @test natural_diagnostics.timings.transport_projection_relative_l2 <
              dirichlet_diagnostics.timings.transport_projection_relative_l2
    end
end

@testset "Surface mesh generation" begin
    PG = PlanetaryGeostrophy
    coastline = ellipse_boundary(1.0, 0.75; points=64)
    mktempdir() do directory
        mesh_file = joinpath(directory, "pressure.msh")
        PG.generate_pressure_mesh(mesh_file, coastline; resolution=0.05)
        written = mtime(mesh_file)
        @test PG.pressure_mesh_is_current(mesh_file, coastline; resolution=0.05)
        @test !PG.pressure_mesh_is_current(mesh_file, coastline; resolution=0.04)
        @test !PG.pressure_mesh_is_current(
            mesh_file, ellipse_boundary(1.0, 0.7; points=64); resolution=0.05,
        )
        sleep(1.1)
        PG.generate_pressure_mesh(mesh_file, coastline; resolution=0.05)
        @test mtime(mesh_file) == written

        # The realized size follows `resolution`, not the coastline vertex
        # spacing (0.086 for 64 points, 0.022 for 256 points).
        for (points, resolution) in ((64, 0.04), (64, 0.02), (256, 0.02))
            path = joinpath(directory, "size_$(points)_$(resolution).msh")
            PG.generate_pressure_mesh(
                path, ellipse_boundary(1.0, 0.75; points); resolution,
            )
            ratio = PG.mean_edge_length(PG.GmshDiscreteModel(path)) / resolution
            @test 0.9 < ratio < 1.05
        end
    end
end
