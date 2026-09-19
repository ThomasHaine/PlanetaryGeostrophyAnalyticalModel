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

@testset "Depth-integrated momentum diagnostics" begin
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
        restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3sin(pi * x / Lx),
        surface_buoyancy=(x, y) ->
            2e-3cos(pi * x / Lx) * cos(pi * y / Ly),
        boundary=rectangle_boundary(Lx, Ly),
    )

    mktempdir() do directory
        solution = solve_pg(
            problem;
            mesh_file=joinpath(directory, "pressure.msh"),
            mesh_resolution=0.3,
            pressure_order=1,
            overwrite_mesh=true,
        )
        x, y = -0.3, 0.0
        terms = depth_integrated_momentum_terms(solution, x, y)
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

        coordinates = [-0.3, 0.2]
        shape = (2, 2)
        zeros2 = zeros(shape)
        diagnostics = PGDiagnostics(
            coordinates,
            coordinates,
            zeros2,
            zeros2,
            zeros2,
            zeros2,
            zeros2,
            trues(shape),
            (;),
        )
        dim = compute_dim_diagnostics(solution, diagnostics)

        @test size(dim.chi) == shape
        @test maximum(abs, dim.x_residual) < 1e-9
        @test maximum(abs, dim.y_residual) < 1e-9

        balance_path = joinpath(directory, "dim_balance.png")
        chi_path = joinpath(directory, "chi.png")
        pressure_path = joinpath(directory, "bottom_pressure.png")
        plot_dim_balance(dim; output=balance_path)
        plot_baroclinic_potential_energy(dim; output=chi_path)
        plot_bottom_pressure(dim; output=pressure_path)

        @test isfile(balance_path)
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
        @test abs(variable_terms.x_residual) < 1e-8
        @test abs(variable_terms.y_residual) < 1e-8
    end
end
