# How the cost of the PG solve scales with the surface-pressure mesh, the Ekman
# number, and the number of threads (manuscript Section 6e), in the
# verification basin of VerificationConvergence.jl.
#
# The cost has two parts. The forcing evaluation computes the column integrals
# for the coefficients at every quadrature point of the mesh; each column uses
# ceil(H / (4 δ_E)) Gauss–Legendre panels, so its cost grows like s = H/δ_E.
# The finite-element assembly and linear solve depend only on the mesh.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/CostScaling.jl            # mesh and Ek sweeps
#   for t in 1 2 4 8 18; do
#       julia --project=PlanetaryGeostrophy -t $t TheoryNotes/CostScaling.jl threads
#   done                                                                              # thread scaling

using Logging
using PlanetaryGeostrophy
using Printf

global_logger(ConsoleLogger(stderr, Logging.Warn))

const ax, ay = 1.0, 0.75
depth(x, y) = 0.8 + 0.25x + 0.1y + 0.3exp(-((x - 0.2)^2 + y^2) / 0.05)
problem(Ek) = PGProblem(
    parameters=PGParameters(Ek, 0.2),
    depth=depth,
    coriolis=(x, y) -> 1.0 + 0.15y,
    wind_stress=(x, y) -> 1e-3cos(pi * y / 1.5) + 2e-4im * sin(x),
    restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3x * exp(2z),
    surface_buoyancy=(x, y) -> 2e-3y,
    boundary=ellipse_boundary(ax, ay; points=64),
)

solve(Ek, h) = solve_pg(problem(Ek); mesh_resolution=h, overwrite_mesh=true,
    mesh_file=joinpath(tempdir(), "cost_scaling_$(Threads.nthreads()).msh"))

# The best of `repeats` solves, with its stage breakdown.
function timed(Ek, h; repeats=3)
    best = nothing
    for _ in 1:repeats
        seconds = @elapsed solution = solve(Ek, h)
        if best === nothing || seconds < best.seconds
            t = solution.timings.surface_pressure
            best = (; seconds, forcing=t.forcing_evaluation.seconds,
                assembly=t.operator_assembly.seconds, linear=t.linear_solve.seconds,
                points=t.forcing_points, edge=t.mean_edge_length)
        end
    end
    return best
end

solve(1e-2, 0.3)                                     # compile

if get(ARGS, 1, "") == "threads"
    r = timed(1e-2, 0.02)
    @printf("threads %2d: solve %.2f s (forcing %.2f, assembly %.2f, linear solve %.2f)\n",
        Threads.nthreads(), r.seconds, r.forcing, r.assembly, r.linear)
else
    println("Mesh sweep, Ek = 0.01 ($(Threads.nthreads()) threads; best of 3):")
    @printf("  %6s %8s %12s %9s %9s %9s %9s\n", "h", "edge", "quad points", "total", "forcing", "assembly", "solve")
    for h in (0.08, 0.04, 0.02, 0.01)
        r = timed(1e-2, h)
        @printf("  %6.3f %8.4f %12d %8.2fs %8.2fs %8.2fs %8.2fs\n",
            h, r.edge, r.points, r.seconds, r.forcing, r.assembly, r.linear)
    end
    println("Ek sweep, h = 0.04 (maximum s = H/δ_E in parentheses):")
    Hmax = 1.4
    for Ek in (1e-2, 1e-3, 1e-4, 1e-5)
        r = timed(Ek, 0.04)
        @printf("  Ek %.0e (s ≤ %5.0f): total %7.2f s, forcing %7.2f s, assembly %.2f s, solve %.2f s\n",
            Ek, Hmax / sqrt(2Ek), r.seconds, r.forcing, r.assembly, r.linear)
    end
end
