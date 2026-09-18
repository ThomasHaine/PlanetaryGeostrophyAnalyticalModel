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
