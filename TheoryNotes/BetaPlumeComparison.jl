# Stommel (1982) beta plume recovered by the PG solution, with the classical
# drag derived rather than fitted (manuscript Section 6c).
#
# A flat rectangular basin on a beta plane is forced by an isolated Gaussian
# patch of wind-stress curl east of the basin centre, with no buoyancy forcing
# (as PG_Beta_Plume_Example.ipynb). The classical model is the frictional
# barotropic vorticity equation of Stommel (1948, 1982),
#     ∇·(r ∇Ψ) + β ∂x Ψ = Ek curl τ,   Ψ = 0 on the coast,
# solved here by second-order finite differences, so that the walls and a
# spatially varying drag are treated exactly. Two drags are used, both from the
# deep limit of the PG bottom stress (Section 6a):
#   classical: r = f0 δ_E / (2H), constant;
#   corrected: r(y) = f δ_E / (2 (H - δ_E/2)), with δ_E = sqrt(2 ν Ek / f),
#              i.e. the local Ekman drag acting over the displacement thickness.
# West of the source the plume spreads like a tracer diffusing westward with
# diffusivity ℓ = r/β, until it reaches the zonal walls. The PG streamfunction
# is integrated from the PG transport, Ψ(x) = -∫_x^{east} V dx'.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/BetaPlumeComparison.jl
# Writes TheoryNotes/beta_plume_comparison.{pdf,png}.

ENV["GKSwstype"] = "100"
using Logging
using PlanetaryGeostrophy
using Plots
using Printf
using SparseArrays
include("ComparisonTools.jl")
include("FigureStyle.jl")

global_logger(ConsoleLogger(stderr, Logging.Warn))

const Lx, Ly = 4.0, 2.0
const H0 = 1.0
const f0, beta = 1.0, 0.3
const Γ = 0.2
const centre = (0.8, 0.0)
const radius = 0.2
const curl_amplitude = 8e-4

# Azimuthal stress whose curl is a Gaussian.
const tau, wind_curl = gaussian_curl_stress(centre, radius, curl_amplitude)

coriolis(y) = f0 + beta * y
ekman_depth(Ek, f) = sqrt(2Ek / f)
classical_drag(Ek) = (x, y) -> f0 * ekman_depth(Ek, f0) / (2H0)
corrected_drag(Ek) = (x, y) -> coriolis(y) * ekman_depth(Ek, coriolis(y)) /
    (2 * (H0 - ekman_depth(Ek, coriolis(y)) / 2))

problem(Ek) = PGProblem(
    parameters=PGParameters(Ek, Γ),
    depth=(x, y) -> H0,
    coriolis=(x, y) -> coriolis(y),
    wind_stress=tau,
    wind_stress_curl=wind_curl,
    restoring_buoyancy=(x, y, z) -> 0.0,
    surface_buoyancy=(x, y) -> 0.0,
    boundary=rectangle_boundary(Lx, Ly),
)

# ∇·(r∇Ψ) + β ∂xΨ = Ek curl τ on the cell-centred interior of a uniform grid,
# with Ψ = 0 on the walls. Returns Ψ on the grid nodes, walls included.
function stommel_fd(Ek, drag; Δ=0.005)
    nx, ny = round(Int, Lx / Δ), round(Int, Ly / Δ)
    x = range(-Lx / 2, Lx / 2; length=nx + 1)
    y = range(-Ly / 2, Ly / 2; length=ny + 1)
    index(i, j) = (j - 2) * (nx - 1) + (i - 1)          # interior nodes 2:nx, 2:ny
    n = (nx - 1) * (ny - 1)
    rows, cols, vals = Int[], Int[], Float64[]
    rhs = zeros(n)
    for j in 2:ny, i in 2:nx
        k = index(i, j)
        re, rw = drag(x[i] + Δ / 2, y[j]), drag(x[i] - Δ / 2, y[j])
        rn, rs = drag(x[i], y[j] + Δ / 2), drag(x[i], y[j] - Δ / 2)
        push!(rows, k); push!(cols, k); push!(vals, -(re + rw + rn + rs) / Δ^2)
        for (p, q, c) in ((i + 1, j, re / Δ^2 + beta / 2Δ), (i - 1, j, rw / Δ^2 - beta / 2Δ),
                          (i, j + 1, rn / Δ^2), (i, j - 1, rs / Δ^2))
            (2 <= p <= nx && 2 <= q <= ny) || continue
            push!(rows, k); push!(cols, index(p, q)); push!(vals, c)
        end
        rhs[k] = Ek * wind_curl(x[i], y[j])
    end
    Ψ = zeros(nx + 1, ny + 1)
    Ψ[2:nx, 2:ny] = reshape(sparse(rows, cols, vals, n, n) \ rhs, nx - 1, ny - 1)
    return x, y, Ψ
end

function compare(Ek; mesh_resolution=min(0.04, classical_drag(Ek)(0.0, 0.0) / beta / 3))
    ℓ = classical_drag(Ek)(0.0, 0.0) / beta
    seconds = @elapsed solution = solve_pg(problem(Ek);
        mesh_file=joinpath(tempdir(), "pg_beta_plume.msh"), mesh_resolution, overwrite_mesh=true)
    x = clamp.(collect(range(-Lx / 2, Lx / 2; length=801)), -Lx / 2 + 1e-6, Lx / 2 - 1e-6)
    y = clamp.(collect(range(-Ly / 2, Ly / 2; length=201)), -Ly / 2 + 1e-6, Ly / 2 - 1e-6)
    Ψ_pg = pg_streamfunction(solution, x, y)
    classical = stommel_fd(Ek, classical_drag(Ek))
    corrected = stommel_fd(Ek, corrected_drag(Ek))
    on_grid(model) = [interpolate_grid(model..., xx, yy) for xx in x, yy in y]
    Ψ_classical, Ψ_corrected = on_grid(classical), on_grid(corrected)
    # The plume: west of the source, away from the western boundary layer.
    plume = (x .> -Lx / 2 + 6ℓ) .& (x .< centre[1] - 2radius)
    return (; Ek, s=H0 / ekman_depth(Ek, f0), ℓ, mesh_resolution, seconds, x, y,
        Ψ_pg, Ψ_classical, Ψ_corrected, plume,
        classical=relative_l2(Ψ_pg, Ψ_classical), corrected=relative_l2(Ψ_pg, Ψ_corrected),
        # (No plume region once 6ℓ reaches the source.)
        plume_classical=any(plume) ? relative_l2(Ψ_pg[plume, :], Ψ_classical[plume, :]) : NaN,
        plume_corrected=any(plume) ? relative_l2(Ψ_pg[plume, :], Ψ_corrected[plume, :]) : NaN,
    )
end

# Finite-difference convergence of the classical model.
let Ek = 3e-3
    coarse, fine = stommel_fd(Ek, classical_drag(Ek); Δ=0.01), stommel_fd(Ek, classical_drag(Ek); Δ=0.005)
    @printf("FD check (Ek %.1e): relative change from Δ = 0.01 to 0.005: %.1e\n", Ek,
        relative_l2(coarse[3], fine[3][1:2:end, 1:2:end]))
end

cases = map((7.5e-4, 3e-3, 1.2e-2, 4.8e-2)) do Ek
    c = compare(Ek)
    @printf("Ek %.2e  s %5.1f  ℓ = r/β %.4f  mesh %.4f | Ψ relative L2, basin: classical %.2e corrected %.2e | plume: classical %.2e corrected %.2e (PG solve %.1f s)\n",
        c.Ek, c.s, c.ℓ, c.mesh_resolution, c.classical, c.corrected, c.plume_classical, c.plume_corrected, c.seconds)
    c
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
reference = cases[2]
const figure_size = (1100, 720)
style = manuscript_style(figure_size[1], 14)
levels = range(minimum(reference.Ψ_pg), 0; length=13)
mapstyle = (; aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", colorbar=false,
    levels, color=:viridis, xlims=(-Lx / 2, Lx / 2), ylims=(-Ly / 2, Ly / 2))
source = [(centre[1] + radius * cos(t), centre[2] + radius * sin(t)) for t in range(0, 2pi; length=60)]
panel_a = contourf(reference.x, reference.y, reference.Ψ_pg';
    title=@sprintf("(a) PG \$\\Psi\$, \$s = %.0f\$", reference.s), mapstyle..., style...)
plot!(panel_a, first.(source), last.(source); color=:white, ls=:dash, label="")
panel_b = contourf(reference.x, reference.y, reference.Ψ_classical';
    title="(b) Stommel, \$r = f_0\\delta_E/2H\$", mapstyle..., style...)
plot!(panel_b, first.(source), last.(source); color=:white, ls=:dash, label="")

# (c) Meridional sections across the plume, west of the source.
section_x = -0.6
panel_c = plot(; xlabel="\$y\$", ylabel="\$\\Psi / \\min \\Psi_{\\mathrm{Stommel}}\$",
    title="(c) section across the plume, \$x = $(section_x)\$", legend=:bottom, style...)
for (k, c) in enumerate(cases)
    i = argmin(abs.(c.x .- section_x))
    scale = minimum(c.Ψ_classical[i, :])
    plot!(panel_c, c.y, c.Ψ_pg[i, :] ./ scale; color=k, lw=1.5, label=@sprintf("\$s = %.1f\$", c.s))
    plot!(panel_c, c.y, c.Ψ_classical[i, :] ./ scale; color=k, ls=:dash, lw=1, label="")
end
plot!(panel_c, [NaN], [NaN]; color=:black, ls=:dash, label="Stommel")

s_values = [c.s for c in cases]
panel_d = plot(s_values, [c.classical for c in cases]; marker=:circle, lw=1.5, xscale=:log10,
    yscale=:log10, xlabel="\$s = H/\\delta_E\$", ylabel="relative \$L_2\$ difference in \$\\Psi\$",
    title="(d) PG against Stommel", label="constant \$r\$",
    xticks=(s_values, [@sprintf("%.1f", v) for v in s_values]),
    yticks=([0.005, 0.01, 0.02, 0.05, 0.1], ["0.005", "0.01", "0.02", "0.05", "0.1"]),
    ylims=(0.005, 0.12), legend=:bottomleft, style...)
plot!(panel_d, s_values, [c.corrected for c in cases]; marker=:square, lw=1.5,
    label="corrected \$r(y)\$")

colorbar_ab = slim_colorbar((first(levels), last(levels)), style; color=:viridis)
figure = plot(panel_a, panel_b, colorbar_ab, panel_c, panel_d, blank_panel();
    layout=grid(2, 3; widths=[0.48, 0.48, 0.04]), size=figure_size,
    left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "beta_plume_comparison.$extension"))
end
