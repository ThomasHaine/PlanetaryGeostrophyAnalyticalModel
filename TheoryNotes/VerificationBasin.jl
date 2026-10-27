# The verification basin of manuscript Section 6b (Fig. 4, Tables 2 and 4),
# shared by VerificationConvergence.jl, CostScaling.jl and ColumnMethodTimings.jl:
# an ellipse on a β-plane with a sloping bottom and a seamount, forced by
# surface stress, surface buoyancy, and depth-dependent restoring buoyancy.
# Include after `using PlanetaryGeostrophy`.

const ax, ay = 1.0, 0.75
verification_depth(x, y) = 0.8 + 0.25x + 0.1y + 0.3exp(-((x - 0.2)^2 + y^2) / 0.05)
verification_problem(Ek=0.01) = PGProblem(
    parameters=PGParameters(Ek, 0.2),
    depth=verification_depth,
    coriolis=(x, y) -> 1.0 + 0.15y,
    wind_stress=(x, y) -> 1e-3cos(pi * y / 1.5) + 2e-4im * sin(x),
    restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3x * exp(2z),
    surface_buoyancy=(x, y) -> 2e-3y,
    boundary=ellipse_boundary(ax, ay; points=64),
)
