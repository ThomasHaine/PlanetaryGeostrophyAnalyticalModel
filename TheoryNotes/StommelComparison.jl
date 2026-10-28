# Sverdrup interior and Stommel western boundary layer recovered by the PG
# solution, with the classical drag coefficient derived rather than fitted
# (manuscript Section 6c).
#
# A flat rectangular basin on a beta plane is forced by a zonal wind with one
# sign of curl, and has no buoyancy forcing (as PG_Stommel_Gyre_Example.ipynb).
# The classical model is Stommel (1948):
#     r ∇²Ψ + β ∂x Ψ = Ek curl τ,   Ψ = 0 on the coast,
# with the linear drag r = f δ_E / (2H) from the deep limit of the PG bottom
# stress, and δ_E = sqrt(2 ν Ek / f) at f = f0. For τx = -τ0 cos(π η) the
# solution separates exactly as Ψ = X(x) sin(π η). The PG streamfunction is
# integrated from the PG transport, Ψ(x) = -∫_x^east V dx', independent of the
# finite-element streamfunction solve. Ek is swept so that s = H/δ_E runs from
# deep (thin Ekman layers) to s ≈ 3.5 (Ekman layers fill much of the column).
#
# Two corrections to the classical constant drag follow from the PG bottom
# stress, and are applied row by row (Ψ ≈ X_{r(y)}(x) sin(π η), which neglects
# ∂y r): the Ekman drag varies with latitude, r(y) = sqrt(f(y) ν Ek / 2) / H,
# and the bottom Ekman layer is a displacement thickness, so the geostrophic
# velocity it acts on is U/(H - δ_E/2).
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/StommelComparison.jl
# Writes TheoryNotes/stommel_comparison.{pdf,png}.

ENV["GKSwstype"] = "100"
using Logging
using PlanetaryGeostrophy
using Plots
using Printf
include("ComparisonTools.jl")
include("FigureStyle.jl")

global_logger(ConsoleLogger(stderr, Logging.Warn))

const Lx, Ly = 2.0, 1.0
const H0 = 1.0
const f0, beta = 1.0, 0.6
const Γ = 0.2
const tau0 = 5e-4

eta(y) = (y + Ly / 2) / Ly
tau_x(x, y) = -tau0 * cos(pi * eta(y))
wind_curl(x, y) = -tau0 * pi / Ly * sin(pi * eta(y))

ekman_depth(Ek) = sqrt(2 * Ek / f0)
stommel_drag(Ek) = f0 * ekman_depth(Ek) / (2 * H0)

problem(Ek) = PGProblem(
    parameters=PGParameters(Ek, Γ),
    depth=(x, y) -> H0,
    coriolis=(x, y) -> f0 + beta * y,
    wind_stress=(x, y) -> tau_x(x, y) + 0im,
    wind_stress_curl=wind_curl,
    restoring_buoyancy=(x, y, z) -> 0.0,
    surface_buoyancy=(x, y) -> 0.0,
    boundary=rectangle_boundary(Lx, Ly),
)

# Stommel (1948): Ψ = X(x) sin(πη) with r (X'' - k² X) + β X' = -Ek τ0 k,
# k = π/Ly, X(±Lx/2) = 0.
function stommel_X(Ek, r)
    k = pi / Ly
    Xp = Ek * tau0 * k / (r * k^2)
    m1 = (-beta + sqrt(beta^2 + 4r^2 * k^2)) / (2r)   # western boundary layer
    m2 = (-beta - sqrt(beta^2 + 4r^2 * k^2)) / (2r)   # slow eastern decay
    # Xp + A e^{m1 (x + Lx/2)} + B e^{m2 (x - Lx/2)} = 0 at both walls.
    M = [1.0 exp(-m2 * Lx); exp(m1 * Lx) 1.0]
    A, B = M \ [-Xp, -Xp]
    return x -> Xp + A * exp(m1 * (x + Lx / 2)) + B * exp(m2 * (x - Lx / 2))
end

function compare(Ek; mesh_resolution)
    seconds = @elapsed solution = solve_pg(problem(Ek);
        mesh_file=joinpath(tempdir(), "pg_stommel.msh"), mesh_resolution, overwrite_mesh=true)
    r = stommel_drag(Ek)
    width = r / beta
    # Points cluster in the western boundary layer.
    xw = range(-Lx / 2, -Lx / 2 + 8width; length=161)
    x = sort(unique(vcat(collect(xw), collect(range(-Lx / 2, Lx / 2; length=401)))))
    x = clamp.(x, -Lx / 2 + 1e-6, Lx / 2 - 1e-6)
    y = range(-Ly / 2 + 1e-6, Ly / 2 - 1e-6; length=41)
    Ψ_pg = pg_streamfunction(solution, x, y)
    separable(drag) = [stommel_X(Ek, drag(yy))(xx) * sin(pi * eta(yy)) for xx in x, yy in y]
    f(y) = f0 + beta * y
    Ψ_classical = separable(_ -> r)
    Ψ_local = separable(y -> sqrt(f(y) * Ek / 2) / H0)
    Ψ_thickness = separable(y -> sqrt(f(y) * Ek / 2) / (H0 - sqrt(2Ek / f(y)) / 2))
    return (; Ek, s=H0 / ekman_depth(Ek), r, width, mesh_resolution, x, y, Ψ_pg,
        Ψ_classical, seconds,
        classical=relative_l2(Ψ_pg, Ψ_classical),
        local_drag=relative_l2(Ψ_pg, Ψ_local),
        thickness=relative_l2(Ψ_pg, Ψ_thickness))
end

cases = map((6.25e-4, 2.5e-3, 1e-2, 4e-2)) do Ek
    c = compare(Ek; mesh_resolution=min(0.02, stommel_drag(Ek) / beta / 3))
    @printf("Ek %.2e  s = H/δ_E %5.1f  r %.4f  r/β %.4f  mesh %.4f | Ψ relative L2: classical %.2e, local r(y) %.2e, + thickness %.2e  (PG solve %.1f s)\n",
        c.Ek, c.s, c.r, c.width, c.mesh_resolution, c.classical, c.local_drag, c.thickness, c.seconds)
    c
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
reference = cases[2]
const figure_size = (1000, 720)
style = manuscript_style(figure_size[1], 14)
limits = (0.0, maximum(reference.Ψ_pg))
mapticks = (; xlims=(-Lx / 2, Lx / 2), ylims=(-Ly / 2, Ly / 2), xticks=-1:0.5:1, yticks=-0.5:0.5:0.5)
panel_a = contourf(reference.x, reference.y, reference.Ψ_pg'; levels=12, clims=limits,
    color=:viridis, aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", mapticks...,
    colorbar=false, title=@sprintf("(a) PG \$\\Psi\$, \$s = %.0f\$", reference.s), style...)
panel_b = contourf(reference.x, reference.y, reference.Ψ_classical'; levels=12, clims=limits,
    color=:viridis, aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", colorbar=false, mapticks...,
    title="(b) Stommel (1948), \$r = f_0\\delta_E/2H\$", style...)

panel_c = plot(; xlabel="\$(x + L_x/2)\\,\\beta/r\$", ylabel="\$\\Psi / \\max \\Psi_{\\mathrm{Stommel}}\$",
    title="(c) western boundary layer, \$y = 0\$", xlims=(0, 12), legend=:bottomright, style...)
for (k, c) in enumerate(cases)
    j = (length(c.y) + 1) ÷ 2
    scale = maximum(c.Ψ_classical[:, j])
    ξ = (c.x .+ Lx / 2) ./ c.width
    plot!(panel_c, ξ, c.Ψ_pg[:, j] ./ scale; color=k, lw=1.5, label=@sprintf("\$s = %.1f\$", c.s))
    plot!(panel_c, ξ, c.Ψ_classical[:, j] ./ scale; color=k, ls=:dash, lw=1, label="")
end
plot!(panel_c, [NaN], [NaN]; color=:black, ls=:dash, label="Stommel")

s_values = [c.s for c in cases]
panel_d = plot(s_values, [c.classical for c in cases]; marker=:circle, lw=1.5, xscale=:log10,
    yscale=:log10, xlabel="\$s = H/\\delta_E\$", ylabel="relative \$L_2\$ difference in \$\\Psi\$",
    title="(d) PG against Stommel", label="constant \$r\$",
    xticks=(s_values, [@sprintf("%.1f", v) for v in s_values]),
    yticks=([0.01, 0.02, 0.05, 0.1], ["0.01", "0.02", "0.05", "0.1"]), ylims=(0.007, 0.2),
    legend=:topright, style...)
plot!(panel_d, s_values, [c.local_drag for c in cases]; marker=:square, lw=1.5,
    label="local \$r(y)\$")
plot!(panel_d, s_values, [c.thickness for c in cases]; marker=:diamond, lw=1.5,
    label="local \$r(y)\$, \$H - \\delta_E/2\$")

colorbar_ab = slim_colorbar(limits, style; color=:viridis)
figure = plot(panel_a, panel_b, colorbar_ab, panel_c, panel_d, blank_panel();
    layout=grid(2, 3; widths=[0.48, 0.48, 0.04]), size=figure_size,
    left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "stommel_comparison.$extension"))
end
