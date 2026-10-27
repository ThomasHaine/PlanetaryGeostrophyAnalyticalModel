# Cost of the three implementations of the column solution (manuscript
# Section 6b, "Column solution"):
#   (1) the direct double-quadrature form (B8 in VelocityFieldPrototype),
#   (2) the symbolic-kernel form (B19; solve_pg method=:b23),
#   (3) the single-integral production form (solve_pg method=:fast).
# (1) and (2) are timed per velocity sample at the manufactured state of
# VelocityFieldPrototype.run_manufactured_test. (2) and (3) are timed for the
# whole surface-pressure solve and for velocity sampling in the verification
# basin of VerificationConvergence.jl, on the same mesh and points.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/ColumnMethodTimings.jl

using Logging
using PlanetaryGeostrophy
using Printf

global_logger(ConsoleLogger(stderr, Logging.Warn))
const prototype = PlanetaryGeostrophy.VelocityFieldPrototype

println("(1) against (2), one velocity sample at the manufactured state (single thread):")
manufactured = prototype.run_manufactured_test(samples=5)

const ax, ay = 1.0, 0.75
problem = PGProblem(
    parameters=PGParameters(0.01, 0.2),
    depth=(x, y) -> 0.8 + 0.25x + 0.1y + 0.3exp(-((x - 0.2)^2 + y^2) / 0.05),
    coriolis=(x, y) -> 1.0 + 0.15y,
    wind_stress=(x, y) -> 1e-3cos(pi * y / 1.5) + 2e-4im * sin(x),
    restoring_buoyancy=(x, y, z) -> 1e-2z + 2e-3x * exp(2z),
    surface_buoyancy=(x, y) -> 2e-3y,
    boundary=ellipse_boundary(ax, ay; points=64),
)
const mesh_resolution = 0.04
# Velocity is sampled in whole columns, as the production path is designed to be
# used: it tabulates ∇Π once per column and interpolates in z.
sample_x = range(-0.9ax, 0.9ax; length=25)
sample_y = range(-0.9ay, 0.9ay; length=19)
sample_z = range(-1.4, 0.0; length=49)

function timed_solve(method)
    solve() = solve_pg(problem; method, mesh_resolution, overwrite_mesh=true,
        mesh_file=joinpath(tempdir(), "column_timings_$(method).msh"))
    solve_pg(problem; method, mesh_resolution=0.3, overwrite_mesh=true,
        mesh_file=joinpath(tempdir(), "column_timings_warm.msh"))      # compile
    seconds = @elapsed solution = solve()
    sample_solution(solution, sample_x[1:2], sample_y[1:2], sample_z)        # compile
    sampling = @elapsed fields = sample_solution(solution, sample_x, sample_y, sample_z)
    columns = count(fields.wet[:, :, end])
    velocity = complex.(fields.u[fields.wet], fields.v[fields.wet])
    return (; solution, seconds, sampling, columns, velocity)
end

fast = timed_solve(:fast)
b23 = timed_solve(:b23)
difference = sqrt(sum(abs2, fast.velocity .- b23.velocity) / sum(abs2, b23.velocity))

@printf("\n(2) against (3), verification basin, mesh %.2f, %d threads:\n", mesh_resolution, Threads.nthreads())
@printf("  %-28s %12s %12s\n", "", "(2) :b23", "(3) :fast")
@printf("  %-28s %10.1f s %10.2f s   (%.0f×)\n", "solve_pg", b23.seconds, fast.seconds, b23.seconds / fast.seconds)
@printf("  %-28s %10.1f s %10.2f s   (%.0f×)\n", "sample_solution, $(fast.columns) columns × $(length(sample_z)) levels",
    b23.sampling, fast.sampling, b23.sampling / fast.sampling)
@printf("  %-28s %10.2f ms %10.3f ms\n", "  per column", 1e3b23.sampling / b23.columns, 1e3fast.sampling / fast.columns)
@printf("  relative difference in u, v over all wet samples: %.1e\n", difference)
@printf("\n(1) %.2f ms and (2) %.2f ms per sample on one thread (%.0f×).\n",
    1e3manufactured.b10_time, 1e3manufactured.b23_time, manufactured.speedup)
