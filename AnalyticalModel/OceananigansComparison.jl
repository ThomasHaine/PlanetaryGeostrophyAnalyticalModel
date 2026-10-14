# Comparison of the PG solution with Oceananigans.jl, with wall-clock timings
# (manuscript Section 6b and Appendix C).
#
# Oceananigans' hydrostatic free-surface model is spun up from rest to steady
# state with momentum and tracer advection off. Its steady state then solves
# the same linear, hydrostatic, robust-diagnostic equations as the PG
# solution, apart from discretization. The basin is an ellipse with a vertical
# coastal wall on a beta plane, forced by zonal surface stress, surface
# buoyancy, and buoyancy restoring with horizontal gradients. Buoyancy,
# horizontal velocity, and depth-integrated transport are compared at
# Oceananigans cell centres, away from the coast and the bottom.
#
# Experiment A (flat bottom) is the validation: it converges with the
# Oceananigans resolution. Experiment B (the Gaussian-bowl bottom of
# Oceananigans_Variable_Basin_Validation.ipynb) shows the sensitivity of the
# z-level model to its bottom representation: stair-stepped or partial cells,
# and grid-scale noise that only horizontal viscosity (absent from PG) damps.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto AnalyticalModel/OceananigansComparison.jl
# Oceananigans is loaded from the AnalyticalModel environment, stacked on the
# package environment. Runtime is about 40 minutes on 18 threads; timings need an otherwise idle machine. Writes
# TheoryNotes/oceananigans_comparison.{pdf,png}.

push!(LOAD_PATH, joinpath(@__DIR__))
ENV["GKSwstype"] = "100"

using Logging
using Oceananigans
using PlanetaryGeostrophy
using Plots
using Printf
using Serialization
using Statistics

global_logger(ConsoleLogger(stderr, Logging.Warn))
const PG = PlanetaryGeostrophy

# Time is in units of 1/f0. The Oceananigans viscosity and diffusivity are Ek
# times the nondimensional ones (here 1).
const Lx, Ly = 2.0, 1.5
const ax, ay = 0.85, 0.60
const f0, beta = 1.0, 0.15
const Ek, Γ = 0.02, 0.20
const tau0 = 1e-4
const Hmax = 1.0
# OCEANANIGANS_COMPARISON_QUICK=1 runs tiny grids briefly, to test the script.
const quick = get(ENV, "OCEANANIGANS_COMPARISON_QUICK", "0") == "1"
const spinup_time = quick ? 10.0 : 200.0
const steadiness_window = 5.0

inside_ellipse(x, y) = (x / ax)^2 + (y / ay)^2 <= 1.0
flat_depth(x, y) = inside_ellipse(x, y) ? 0.65 : 0.0
bowl_depth(x, y) = inside_ellipse(x, y) ?
    0.3 + 0.7 * exp(-2 * ((x / ax)^2 + (y / ay)^2)) : 0.0
tau_x(x, y) = tau0 * cos(pi * y / ay)
surface_buoyancy(x, y) = 2e-3 * exp(-((x / (0.55ax))^2 + (y / (0.55ay))^2))
restoring_buoyancy(x, y, z) = 1e-2 * z + 3e-3 * sin(pi * x / ax) * (1 + z / Hmax)

pg_problem(depth) = PGProblem(
    parameters=PGParameters(Ek, Γ),
    depth=depth,
    coriolis=(x, y) -> f0 + beta * y,
    wind_stress=(x, y) -> tau_x(x, y) + 0im,
    restoring_buoyancy=restoring_buoyancy,
    surface_buoyancy=surface_buoyancy,
    boundary=ellipse_boundary(ax, ay; points=256),
)
const pg_mesh_resolution = 0.02

# ---------------------------------------------------------------------------
# Oceananigans
# ---------------------------------------------------------------------------
# Explicit boundary fluxes (no-slip bottom, prescribed surface buoyancy) limit
# the time step to about Δz²/κ, with Δz the thinnest cell, even though the
# interior vertical diffusion is implicit.
time_step(Nz; minimum_fraction=1.0) =
    min(0.05, 0.4 * (minimum_fraction * Hmax / Nz)^2 / Ek)

function oceananigans_model(depth, Nx, Ny, Nz; partial_cells=false, ν_h=0.0)
    underlying = RectilinearGrid(CPU();
        size=(Nx, Ny, Nz), x=(-Lx / 2, Lx / 2), y=(-Ly / 2, Ly / 2), z=(-Hmax, 0),
        topology=(Bounded, Bounded, Bounded), halo=(4, 4, 4),
    )
    bottom(x, y) = inside_ellipse(x, y) ? -depth(x, y) : 0.0
    minimum_fraction = partial_cells ? 0.5 : 1.0
    immersed = partial_cells ?
        PartialCellBottom(bottom; minimum_fractional_cell_height=minimum_fraction) :
        GridFittedBottom(bottom)
    grid = ImmersedBoundaryGrid(underlying, immersed)
    # Positive surface stress τ (ν ∂z u = τ) is a negative upward momentum flux.
    u_bcs = FieldBoundaryConditions(
        top=FluxBoundaryCondition((x, y, t) -> -Ek * tau_x(x, y)),
        bottom=ValueBoundaryCondition(0), immersed=ValueBoundaryCondition(0),
    )
    v_bcs = FieldBoundaryConditions(
        bottom=ValueBoundaryCondition(0), immersed=ValueBoundaryCondition(0),
    )
    b_bcs = FieldBoundaryConditions(
        top=ValueBoundaryCondition((x, y, t) -> surface_buoyancy(x, y)),
    )
    vertical = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=Ek, κ=Ek)
    closure = ν_h > 0 ? (vertical, HorizontalScalarDiffusivity(ν=ν_h)) : vertical
    Δt = time_step(Nz; minimum_fraction)
    model = HydrostaticFreeSurfaceModel(grid;
        momentum_advection=nothing, tracer_advection=nothing,
        buoyancy=BuoyancyTracer(), tracers=:b, coriolis=BetaPlane(f₀=f0, β=beta), closure,
        free_surface=SplitExplicitFreeSurface(grid; gravitational_acceleration=10.0, cfl=0.7, fixed_Δt=Δt),
        boundary_conditions=(u=u_bcs, v=v_bcs, b=b_bcs),
        forcing=(b=Relaxation(rate=Γ, target=(x, y, z, t) -> restoring_buoyancy(x, y, z)),),
    )
    set!(model, b=restoring_buoyancy)
    return model, Δt
end

function run_oceananigans(depth, Nx, Ny, Nz; kwargs...)
    model, Δt = oceananigans_model(depth, Nx, Ny, Nz; kwargs...)
    seconds = @elapsed begin
        run!(Simulation(model; Δt, stop_time=spinup_time - steadiness_window))
        u_before = copy(Array(interior(model.velocities.u)))
        run!(Simulation(model; Δt, stop_time=spinup_time))
    end
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    b = Array(interior(model.tracers.b))
    grid = model.grid
    return (; Nx, Ny, Nz, Δt, steps=round(Int, spinup_time / Δt), seconds,
            drift=maximum(abs, u .- u_before) / maximum(abs, u),
            x=collect(xnodes(grid, Center())), y=collect(ynodes(grid, Center())),
            z=collect(znodes(grid, Center())), dz=Hmax / Nz,
            # Velocities at cell centres from the two adjacent faces.
            u=0.5 .* (u[1:end-1, :, :] .+ u[2:end, :, :]),
            v=0.5 .* (v[:, 1:end-1, :] .+ v[:, 2:end, :]), b)
end

# ---------------------------------------------------------------------------
# Comparison on the Oceananigans cell centres
# ---------------------------------------------------------------------------
wet(depth, x, y, z) = inside_ellipse(x, y) && z > -depth(x, y)

function sample_pg(solution, depth, ocean)
    x, y, z = ocean.x, ocean.y, ocean.z
    shape = (length(x), length(y), length(z))
    u, v, b = fill(NaN, shape), fill(NaN, shape), fill(NaN, shape)
    U = fill(complex(NaN, NaN), length(x), length(y))
    problem = solution.problem
    Threads.@threads for j in eachindex(y)
        for i in eachindex(x)
            problem.inside(x[i], y[j]) || continue
            column = PG.horizontal_velocity_column(solution, x[i], y[j])
            current = depth_integrated_current(solution, x[i], y[j])
            U[i, j] = complex(current.U, current.V)
            for k in eachindex(z)
                wet(depth, x[i], y[j], z[k]) || continue
                velocity = column(z[k])
                u[i, j, k], v[i, j, k] = real(velocity), imag(velocity)
                b[i, j, k] = buoyancy(solution, x[i], y[j], z[k])
            end
        end
    end
    return (; u, v, b, U)
end

# Bulk cells: wet, with wet horizontal neighbours at the same depth, at least
# one cell above the bottom, and inside 95% of the coastal radius.
function bulk_masks(depth, ocean, inside)
    x, y, z, dz = ocean.x, ocean.y, ocean.z, ocean.dz
    cells = falses(length(x), length(y), length(z))
    columns = falses(length(x), length(y))
    for j in 2:length(y)-1, i in 2:length(x)-1
        (x[i] / ax)^2 + (y[j] / ay)^2 <= 0.95^2 && inside(x[i], y[j]) || continue
        columns[i, j] = true
        neighbours = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        for k in eachindex(z)
            cells[i, j, k] = wet(depth, x[i], y[j], z[k] - dz) &&
                all(wet(depth, x[p], y[q], z[k]) for (p, q) in neighbours)
        end
    end
    return cells, columns
end

relative_l2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

function ocean_transport(depth, ocean)
    U = fill(complex(NaN, NaN), length(ocean.x), length(ocean.y))
    for j in eachindex(ocean.y), i in eachindex(ocean.x)
        levels = [k for k in eachindex(ocean.z) if wet(depth, ocean.x[i], ocean.y[j], ocean.z[k])]
        isempty(levels) && continue
        U[i, j] = ocean.dz * sum(complex(ocean.u[i, j, k], ocean.v[i, j, k]) for k in levels)
    end
    return U
end

function compare(depth, solution, ocean)
    sampling_seconds = @elapsed pg = sample_pg(solution, depth, ocean)
    cells, columns = bulk_masks(depth, ocean, solution.problem.inside)
    U_ocean = ocean_transport(depth, ocean)
    return (;
        b=relative_l2(ocean.b[cells], pg.b[cells]),
        u=relative_l2(complex.(ocean.u, ocean.v)[cells], complex.(pg.u, pg.v)[cells]),
        U=relative_l2(U_ocean[columns], pg.U[columns]),
        sampling_seconds, pg, U_ocean,
    )
end

report(label, o, e) = @printf(
    "%-34s %3d×%3d×%2d  Δt %.4f %6d steps %7.1f s (%.2f ms/step, drift %.0e) | PG sampling %.2f s | b %.2e  u %.2e  U %.2e\n",
    label, o.Nx, o.Ny, o.Nz, o.Δt, o.steps, o.seconds, 1e3 * o.seconds / o.steps, o.drift,
    e.sampling_seconds, e.b, e.u, e.U)

# OCEANANIGANS_COMPARISON_REPLOT=1 redraws the figure from the saved results.
const results_file = joinpath(tempdir(), "oceananigans_comparison.jls")
if get(ENV, "OCEANANIGANS_COMPARISON_REPLOT", "0") == "1"
    (; flat_results, bowl_results, flat_pg_seconds, bowl_pg_seconds) = deserialize(results_file)
else

# ---------------------------------------------------------------------------
# Warm-up, so that no timing includes compilation
# ---------------------------------------------------------------------------
let
    for kwargs in ((;), (; partial_cells=true), (; ν_h=1e-4))
        model, Δt = oceananigans_model(bowl_depth, 8, 8, 8; kwargs...)
        run!(Simulation(model; Δt, stop_time=3Δt))
    end
    # Each depth function is its own type, so warm up the PG solve for both.
    for depth in (flat_depth, bowl_depth)
        solution = solve_pg(pg_problem(depth); mesh_file=joinpath(tempdir(), "pg_warm.msh"),
            mesh_resolution=0.2, overwrite_mesh=true)
        sample_pg(solution, depth, (; x=[0.0, 0.1], y=[0.0, 0.1], z=[-0.5, -0.2]))
    end
end

# The PG solve (including meshing) is timed as the best of three.
function solve_timed(depth, name)
    solve() = solve_pg(pg_problem(depth);
        mesh_file=joinpath(tempdir(), "pg_oceananigans_$(name).msh"),
        mesh_resolution=pg_mesh_resolution, overwrite_mesh=true)
    seconds = minimum(@elapsed(solve()) for _ in 1:3)
    solution = solve()
    @printf("PG solve (%s): %.2f s, mean edge %.4f, %d threads\n", name, seconds,
        solution.timings.surface_pressure.mean_edge_length, Threads.nthreads())
    return solution, seconds
end

# ---------------------------------------------------------------------------
# Experiment A: flat bottom, three resolutions
# ---------------------------------------------------------------------------
flat_solution, flat_pg_seconds = solve_timed(flat_depth, "flat")
flat_results = map(quick ? [(16, 12, 8), (24, 18, 8), (32, 24, 8)] :
        [(40, 30, 20), (80, 60, 40), (160, 120, 40)]) do (Nx, Ny, Nz)
    ocean = run_oceananigans(flat_depth, Nx, Ny, Nz)
    errors = compare(flat_depth, flat_solution, ocean)
    report("A flat bottom", ocean, errors)
    (; ocean, errors)
end

# ---------------------------------------------------------------------------
# Experiment B: Gaussian-bowl bottom, bottom representation and viscosity
# ---------------------------------------------------------------------------
bowl_solution, bowl_pg_seconds = solve_timed(bowl_depth, "bowl")
bowl_cases = [
    ("B bowl, stair-stepped bottom", (80, 60, 40), (;)),
    ("B bowl, partial cells", (80, 60, 40), (; partial_cells=true)),
    ("B bowl, stair-stepped bottom", (160, 120, 40), (;)),
    ("B bowl, stair-stepped, ν_h = 1e-4", (160, 120, 40), (; ν_h=1e-4)),
]
bowl_results = map(bowl_cases) do (label, (Nx, Ny, Nz), kwargs)
    quick && ((Nx, Ny, Nz) = (16, 12, 8))
    ocean = run_oceananigans(bowl_depth, Nx, Ny, Nz; kwargs...)
    errors = compare(bowl_depth, bowl_solution, ocean)
    report(label, ocean, errors)
    (; label, ocean, errors)
end

println()
println("Wall-clock time to solution on this machine ($(Threads.nthreads()) threads, after warm-up):")
@printf("  PG solve at mesh_resolution %.2f: %.2f s (flat), %.2f s (bowl)\n",
    pg_mesh_resolution, flat_pg_seconds, bowl_pg_seconds)
for r in flat_results
    o = r.ocean
    @printf("  Oceananigans flat %3d×%3d×%2d: %7.1f s = %5.0f × PG solve + sampling\n",
        o.Nx, o.Ny, o.Nz, o.seconds, o.seconds / (flat_pg_seconds + r.errors.sampling_seconds))
end

serialize(results_file, (; flat_results, bowl_results, flat_pg_seconds, bowl_pg_seconds))
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fine = flat_results[end]
o, e = fine.ocean, fine.errors
plotable(field) = map(value -> isfinite(value) ? value : NaN, field)
speed_ocean, speed_pg = plotable(abs.(e.U_ocean)), plotable(abs.(e.pg.U))
limits = (0.0, maximum(filter(isfinite, vcat(vec(speed_ocean), vec(speed_pg)))))
mapstyle = (; aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", titlefontsize=10,
    guidefontsize=9, framestyle=:box, xlims=(-0.9, 0.9), ylims=(-0.65, 0.65))
panel_a = heatmap(o.x, o.y, speed_ocean'; clims=limits, color=:viridis,
    title="(a) Oceananigans \$|\\mathbf{U}|\$, flat, $(o.Nx)×$(o.Ny)×$(o.Nz)", mapstyle...)
panel_b = heatmap(o.x, o.y, speed_pg'; clims=limits, color=:viridis,
    title="(b) PG \$|\\mathbf{U}|\$, flat", mapstyle...)
# The difference is shown over the bulk columns used in the L2 metric.
_, bulk_columns = bulk_masks(flat_depth, o, inside_ellipse)
difference = [bulk_columns[i, j] ? abs(e.U_ocean[i, j] - e.pg.U[i, j]) : NaN for i in axes(e.pg.U, 1), j in axes(e.pg.U, 2)]
panel_c = heatmap(o.x, o.y, difference'; color=:magma,
    title="(c) \$|\\mathbf{U}_{\\mathrm{Oc}} - \\mathbf{U}_{\\mathrm{PG}}|\$ (bulk), flat", mapstyle...)

dx = [Lx / r.ocean.Nx for r in flat_results]
linestyle = (; titlefontsize=10, guidefontsize=9, framestyle=:box, legendfontsize=7)
panel_d = plot(dx, [r.errors.U for r in flat_results]; marker=:circle, lw=1.5, xscale=:log10,
    yscale=:log10, label="\$\\mathbf{U}\$", xlabel="Oceananigans \$\\Delta x\$",
    ylabel="relative \$L_2\$ difference", title="(d) flat bottom: convergence",
    xticks=(dx, [@sprintf("%.4f", d) for d in dx]), yticks=10.0 .^ (-4:0), legend=:bottomright,
    linestyle...)
plot!(panel_d, dx, [r.errors.u for r in flat_results]; marker=:square, lw=1.5, label="\$\\mathbf{u}\$")
plot!(panel_d, dx, [r.errors.b for r in flat_results]; marker=:diamond, lw=1.5, label="\$b\$")
plot!(panel_d, dx, flat_results[end].errors.U .* dx ./ dx[end]; color=:gray, ls=:dash, label="\$\\propto \\Delta x\$")

labels = ["stair\n80×60×40", "partial\n80×60×40", "stair\n160×120×40", "stair, ν_h\n160×120×40"]
positions = 1:length(bowl_results)
panel_e = bar(positions .- 0.2, [r.errors.u for r in bowl_results]; bar_width=0.4,
    label="\$\\mathbf{u}\$")
bar!(panel_e, positions .+ 0.2, [r.errors.U for r in bowl_results]; bar_width=0.4,
    xticks=(positions, labels), label="\$\\mathbf{U}\$",
    ylabel="relative \$L_2\$ difference", title="(e) bowl bottom: Oceananigans sensitivity",
    ylims=(0, 1.15 * maximum(max(r.errors.u, r.errors.U) for r in bowl_results)),
    legend=:topright, linestyle..., xtickfontsize=7)

grid_labels = [@sprintf("%d×%d×%d", r.ocean.Nx, r.ocean.Ny, r.ocean.Nz) for r in flat_results]
panel_f = plot(1:length(flat_results), [r.ocean.seconds for r in flat_results]; yscale=:log10,
    marker=:circle, lw=1.5, color=:steelblue, xticks=(1:length(flat_results), grid_labels),
    xlims=(0.5, length(flat_results) + 0.5), yticks=10.0 .^ (-1:4), ylims=(0.3, 3 * maximum(r.ocean.seconds for r in flat_results)),
    xlabel="Oceananigans grid", ylabel="wall-clock seconds", title="(f) time to solution, flat",
    label="Oceananigans spin-up", legend=:right, linestyle...)
plot!(panel_f, 1:length(flat_results), [flat_pg_seconds + r.errors.sampling_seconds for r in flat_results];
    color=:black, marker=:circle, lw=1.5, label="PG solve + sampling")
hline!(panel_f, [flat_pg_seconds]; color=:black, ls=:dash,
    label=@sprintf("PG solve (mesh %.2f)", pg_mesh_resolution))

figure = plot(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f; layout=(2, 3),
    size=(1250, 760), left_margin=4Plots.mm, bottom_margin=6Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "..", "TheoryNotes", "oceananigans_comparison.$extension"))
end
