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
end
