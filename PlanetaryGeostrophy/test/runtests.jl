using PlanetaryGeostrophy
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
    column_velocity = prototype.velocity_b23_column_from_gradps(
        kernels, p, H, f, f_x, f_y, gradps, τ, B, bs, H_x, H_y, x, y,
    )
    for z in (-H(x, y), -0.5H(x, y), 0.0)
        @test column_velocity(z) ≈ prototype.velocity_b23_from_gradps(
            kernels, p, H, f, f_x, f_y, gradps, τ, B, bs,
            H_x, H_y, x, y, z,
        )
    end
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
        b23_buoyancy_transport =
            PlanetaryGeostrophy.VelocityFieldPrototype.buoyancy_transport_b23(
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
        @test direct_buoyancy_transport ≈ b23_buoyancy_transport rtol=2e-5

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
        b23_buoyancy_stress =
            PlanetaryGeostrophy.VelocityFieldPrototype.buoyancy_bottom_stress_b23(
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
        @test direct_buoyancy_stress ≈ b23_buoyancy_stress rtol=2e-4

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
                panel_size(pressure_equation_figure, index) for index in 1:4
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
        variable_solution = solve_pg(
            variable_problem;
            mesh_file=joinpath(directory, "pressure.msh"),
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
        @test !isapprox(
            variable_pressure_coefficients.advection_velocity_y,
            0.0;
            atol=1e-8,
        )
    end
end
