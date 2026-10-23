# Double-basin showcase (manuscript Section 6d): two deep bowls on an f-plane,
# separated by a shallow ridge and joined by a western strait, with resolved,
# overlapping Ekman layers over the shelf. The geometry and forcing are those of
# AnalyticalModel/PG_Double_Basin_Prototype.ipynb (high-resolution setting).
#
# The script times each stage of the calculation and reports:
#   - the 3-D solution and the northward transport through the strait;
#   - closure of the five dynamical balances (DIM, BV, depth-integrated
#     vorticity, GCVB, TDVB), as relative L2 residuals on the diagnostic grid;
#   - the Nøst–Isachsen (2003) bottom velocity on closed f/H contours with an
#     assumed (calibrated, isotropic) linear or quadratic drag, and with the
#     drag derived from the PG bottom stress (no closure);
#   - the adjoint sensitivity of the strait transport to the local wind stress,
#     surface buoyancy, and restoring buoyancy, from one adjoint solve, and the
#     exact finite-element check of the adjoint identity.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/DoubleBasinShowcase.jl
# Writes TheoryNotes/double_basin_showcase.{pdf,png}.

ENV["GKSwstype"] = "100"
using Logging
using PlanetaryGeostrophy
using Plots
using Printf
using Statistics

global_logger(ConsoleLogger(stderr, Logging.Warn))
include(joinpath(@__DIR__, "..", "AnalyticalModel", "NostIsachsenDiagnostics.jl"))

# ---------------------------------------------------------------------------
# Geometry and forcing (as PG_Double_Basin_Prototype.ipynb)
# ---------------------------------------------------------------------------
const Lx, Ly = 2.0, 2.8
const ax, ay = Lx / 2, Ly / 2
const Ek, Γ = 0.002, 0.20
const f0 = 1.0
const mesh_resolution = 0.02
const section_x, section_y = (-0.60, -0.35), 0.0

gaussian(x, y, cx, cy, sx, sy) = exp(-((x - cx) / sx)^2 - ((y - cy) / sy)^2)
gaussian_x(x, y, cx, cy, sx, sy) = -2(x - cx) / sx^2 * gaussian(x, y, cx, cy, sx, sy)
gaussian_y(x, y, cx, cy, sx, sy) = -2(y - cy) / sy^2 * gaussian(x, y, cx, cy, sx, sy)

const shelf_depth, shelf_width = 0.26, 0.18
const western_extra, western_transition, western_transition_width = 0.50, -0.45, 0.15
const basins = ((amplitude=0.96, cx=0.01, cy=0.69, sx=0.62, sy=0.455),
                (amplitude=0.96, cx=0.01, cy=-0.69, sx=0.62, sy=0.455))
const fram_amplitude, fram_transition, fram_width = 1.30, -0.54, 0.10

sigmoid(x, x0, w) = 1 / (1 + exp((x - x0) / w))
sigmoid_x(x, x0, w) = -exp((x - x0) / w) / (w * (1 + exp((x - x0) / w))^2)
fram(x) = fram_amplitude * sigmoid(x, fram_transition, fram_width)
fram_x(x) = fram_amplitude * sigmoid_x(x, fram_transition, fram_width)
shelf(x) = shelf_width + western_extra * sigmoid(x, western_transition, western_transition_width)
shelf_x(x) = western_extra * sigmoid_x(x, western_transition, western_transition_width)
radius(x, y) = 1 - (x / ax)^2 - (y / ay)^2
taper(x, y) = 1 - exp(-(radius(x, y) / shelf(x))^2)
taper_x(x, y) = exp(-(radius(x, y) / shelf(x))^2) *
    (-4x * radius(x, y) / (ax^2 * shelf(x)^2) - 2radius(x, y)^2 * shelf_x(x) / shelf(x)^3)
taper_y(x, y) = -4y * radius(x, y) / (ay^2 * shelf(x)^2) * exp(-(radius(x, y) / shelf(x))^2)
deepening(x, y) = sum(b.amplitude * gaussian(x, y, b.cx, b.cy, b.sx, b.sy) for b in basins) + fram(x)
deepening_x(x, y) = sum(b.amplitude * gaussian_x(x, y, b.cx, b.cy, b.sx, b.sy) for b in basins) + fram_x(x)
deepening_y(x, y) = sum(b.amplitude * gaussian_y(x, y, b.cx, b.cy, b.sx, b.sy) for b in basins)
depth(x, y) = shelf_depth + taper(x, y) * deepening(x, y)
depth_x(x, y) = taper_x(x, y) * deepening(x, y) + taper(x, y) * deepening_x(x, y)
depth_y(x, y) = taper_y(x, y) * deepening(x, y) + taper(x, y) * deepening_y(x, y)

const stress_amplitude = 1e-3
const cells = Tuple((cx=b.cx, cy=b.cy, sx=0.56, sy=0.51) for b in basins)
wind_stress(x, y) = sum(stress_amplitude * gaussian(x, y, c.cx, c.cy, c.sx, c.sy) *
    (-(y - c.cy) / c.sy + im * (x - c.cx) / c.sx) for c in cells)
wind_stress_curl(x, y) = sum(stress_amplitude * gaussian(x, y, c.cx, c.cy, c.sx, c.sy) *
    ((1 - 2((x - c.cx) / c.sx)^2) / c.sx + (1 - 2((y - c.cy) / c.sy)^2) / c.sy) for c in cells)

problem = PGProblem(
    parameters=PGParameters(Ek, Γ),
    depth=depth, depth_x=depth_x, depth_y=depth_y,
    coriolis=(x, y) -> f0, coriolis_x=(x, y) -> 0.0, coriolis_y=(x, y) -> 0.0,
    wind_stress=wind_stress, wind_stress_curl=wind_stress_curl,
    restoring_buoyancy=(x, y, z) -> 1e-2z,
    surface_buoyancy=(x, y) -> 2e-3 * y / Ly,
    boundary=ellipse_boundary(ax, ay; points=64),
)

# ---------------------------------------------------------------------------
# Calculation, timed stage by stage after a coarse warm-up
# ---------------------------------------------------------------------------
let
    warm = solve_pg(problem; mesh_file=joinpath(tempdir(), "double_basin_warm.msh"),
        mesh_resolution=0.16, overwrite_mesh=true)
    d = compute_diagnostics(warm, range(-0.9ax, 0.9ax; length=9), range(-0.9ay, 0.9ay; length=9))
    compute_dim_diagnostics(warm, d)
    sample_solution(warm, range(-0.5, 0.5; length=3), [-0.01, 0.0, 0.01], [-0.5, -0.2, 0.0])
end

timings = Pair{String,Float64}[]
timed(label, f) = (t = @elapsed(value = f()); push!(timings, label => t); value)

solution = timed("PG solve (mesh 0.02)", () -> solve_pg(problem;
    mesh_file=joinpath(tempdir(), "double_basin_showcase.msh"), mesh_resolution, overwrite_mesh=true))

δ_E = sqrt(2Ek / f0)
x3 = collect(range(-0.98ax, 0.98ax; length=241))
z3 = collect(range(-maximum(depth(xx, 0.0) for xx in x3 if problem.inside(xx, 0.0)), 0.0; length=121))
section = timed("3-D sampling, zonal section (241 × 3 × 121)",
    () -> sample_solution(solution, x3, [section_y - 0.01, section_y, section_y + 0.01], z3))

dx = range(-0.94ax, 0.94ax; length=65)
dy = range(-0.94ay, 0.94ay; length=65)
diagnostics = timed("transport, bottom stress, Ψ (65 × 65)", () -> compute_diagnostics(solution, dx, dy))
dim = timed("DIM balance", () -> compute_dim_diagnostics(solution, diagnostics))
bv = timed("BV balance (cell averages)", () -> compute_barotropic_vorticity_diagnostics(solution, diagnostics))
vorticity = timed("depth-integrated vorticity", () -> compute_vorticity_diagnostics(solution, diagnostics))
gcvb = timed("GCVB", () -> compute_geostrophic_contour_vorticity_diagnostics(solution, diagnostics))
tdvb = timed("TDVB", () -> compute_transport_divergence_vorticity_diagnostics(solution, diagnostics))
nost_isachsen = timed("Nøst–Isachsen comparison",
    () -> NostIsachsenDiagnostics.compute_nost_isachsen_diagnostics(solution, diagnostics, dim, tdvb))

# Northward transport through the strait section, by the trapezoid rule.
section_points = collect(range(section_x...; length=201))
section_V = [depth_integrated_current(solution, xx, section_y).V for xx in section_points]
strait_transport = sum(0.5 .* (section_V[1:end-1] .+ section_V[2:end]) .* diff(section_points))

width = max(mesh_resolution, 0.04)
weight(x, y) = section_x[1] <= x <= section_x[2] ?
    exp(-((y - section_y) / width)^2 / 2) / (sqrt(2pi) * width) : 0.0
kernel_z = [-0.5]
adjoint = timed("adjoint sensitivity (one adjoint solve + kernels)",
    () -> compute_transport_adjoint_sensitivity(solution, dx, dy, kernel_z;
        target=(sum(section_x) / 2, section_y), cost=(U, V) -> V, cost_gradient=(0.0, 1.0),
        cost_value=strait_transport, target_weight=weight,
        target_metadata=(kind=:zonal_section, x=sum(section_x) / 2, y=section_y,
            x_limits=section_x, width),
        cost_label="northward strait transport", target_radius=width))
identity = transport_adjoint_identity(adjoint, solution)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
println("Double basin: Ek = $Ek, δ_E = $(round(δ_E; sigdigits=3)), shelf depth $shelf_depth = $(round(shelf_depth / δ_E; sigdigits=2)) δ_E, ",
    "$(Threads.nthreads()) threads, $(solution.timings.surface_pressure.forcing_points) forcing quadrature points")
println("Wall-clock time by stage:")
for (label, t) in timings
    @printf("  %-52s %7.2f s\n", label, t)
end
@printf("Northward strait transport %.4e (x in %s, y = %.1f)\n", strait_transport, section_x, section_y)
println("Balance closure (relative L2 residual on the 65 × 65 grid):")
@printf("  DIM x %.1e, DIM y %.1e, BV %.2e, depth-integrated vorticity %.2e, GCVB %.2e, TDVB %.2e\n",
    dim.timings.x_relative_l2_residual, dim.timings.y_relative_l2_residual,
    bv.timings.relative_l2_residual, vorticity.timings.relative_l2_residual,
    gcvb.timings.relative_l2_residual, tdvb.timings.relative_l2_residual)
m = nost_isachsen.metrics
println("Nøst–Isachsen bottom velocity on closed f/H contours ($(m.comparison_points) points), relative L2 error against PG:")
@printf("  calibrated linear drag %.4g: %.3f (speed correlation %.3f)\n", m.calibrated_linear_drag,
    m.linear_relative_l2_velocity_error, m.linear_speed_correlation)
@printf("  calibrated quadratic drag %.4g: %.3f (speed correlation %.3f)\n", m.calibrated_quadratic_drag,
    m.quadratic_relative_l2_velocity_error, m.quadratic_speed_correlation)
@printf("  derived PG drag (no closure): %.4f (speed correlation %.4f)\n",
    m.exact_relative_l2_velocity_error, m.exact_speed_correlation)
@printf("  stress–velocity alignment of the PG bottom stress %.3f; cross-contour velocity %.2e of the speed\n",
    m.stress_velocity_alignment, m.relative_l2_cross_q_velocity)
@printf("Adjoint identity: forward %.6e, adjoint %.6e, relative error %.1e\n",
    identity.forward, identity.adjoint, identity.relative_error)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
style = (; titlefontsize=10, guidefontsize=9, framestyle=:box, legendfontsize=7)
masked(field, wet) = [wet[i, j] ? field[i, j] : NaN for i in axes(field, 1), j in axes(field, 2)]
H = [problem.inside(xx, yy) ? depth(xx, yy) : NaN for xx in dx, yy in dy]
mapstyle = (; aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", xlims=(-ax, ax), ylims=(-ay, ay))
coast = [(ax * cos(t), ay * sin(t)) for t in range(0, 2pi; length=200)]
function decorate!(p)
    contour!(p, dx, dy, H'; levels=[0.4, 0.8, 1.2], color=:gray, lw=0.7, colorbar=false)
    plot!(p, first.(coast), last.(coast); color=:black, lw=1, label="")
    plot!(p, collect(section_x), [section_y, section_y]; color=:gold, lw=3, label="")
    return p
end

Ψ = masked(diagnostics.streamfunction, diagnostics.wet)
Ψmax = maximum(abs, filter(isfinite, Ψ))
panel_a = contourf(dx, dy, Ψ'; levels=range(-Ψmax, Ψmax; length=17), color=:balance,
    title="(a) \$\\Psi\$, isobaths 0.4, 0.8, 1.2, strait", colorbar=false, mapstyle..., style...)
decorate!(panel_a)

j0 = 2
v_section = [section.wet[i, j0, k] ? section.v[i, j0, k] : NaN for i in eachindex(x3), k in eachindex(z3)]
b_section = [section.wet[i, j0, k] ? section.b[i, j0, k] : NaN for i in eachindex(x3), k in eachindex(z3)]
vmax = maximum(abs, filter(isfinite, v_section))
panel_b = heatmap(x3, z3, v_section'; color=:balance, clims=(-vmax, vmax), xlabel="\$x\$", ylabel="\$z\$",
    title="(b) \$v\$ and \$b\$ at \$y = 0\$", colorbar=false, style...)
contour!(panel_b, x3, z3, b_section'; levels=12, color=:black, lw=0.6, colorbar=false)
plot!(panel_b, x3, [problem.inside(xx, 0.0) ? -depth(xx, 0.0) : NaN for xx in x3]; color=:black, lw=1.5, label="")
vspan!(panel_b, collect(section_x); color=:gold, alpha=0.2, label="")

r = nost_isachsen
speed_max = maximum(filter(isfinite, r.pg_speed[r.theory_wet]))
panel_c = scatter(r.pg_speed[r.theory_wet], r.linear_speed[r.theory_wet]; ms=1.5, msw=0, alpha=0.4,
    color=:gray, label=@sprintf("assumed linear drag (%.0f%%)", 100m.linear_relative_l2_velocity_error),
    xlabel="PG bottom speed", ylabel="Nøst–Isachsen bottom speed",
    title="(c) closed-contour bottom velocity", legend=:topleft, style...)
scatter!(panel_c, r.pg_speed[r.theory_wet], r.exact_speed[r.theory_wet]; ms=1.5, msw=0, alpha=0.5,
    color=:steelblue, label=@sprintf("derived PG drag (%.0f%%)", 100m.exact_relative_l2_velocity_error))
plot!(panel_c, [0, speed_max], [0, speed_max]; color=:black, ls=:dash, label="")

function kernel_panel(field, title)
    values = masked(field, adjoint.wet)
    # Clipped, so that the local response at the section does not hide the
    # basin-scale structure.
    scale = quantile(abs.(filter(isfinite, values)), 0.97)
    p = heatmap(dx, dy, values'; color=:balance, clims=(-scale, scale), title, colorbar=false,
        mapstyle..., style...)
    return decorate!(p)
end
panel_d = kernel_panel(adjoint.wind_stress_x, "(d) strait transport: sensitivity to \$\\tau^x\$")
panel_e = kernel_panel(adjoint.wind_stress_y, "(e) sensitivity to \$\\tau^y\$")
panel_f = kernel_panel(adjoint.restoring_buoyancy[:, :, 1], "(f) sensitivity to \$B\$ at \$z = -0.5\$")

figure = plot(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f; layout=(2, 3), size=(1300, 900),
    left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "double_basin_showcase.$extension"))
end
