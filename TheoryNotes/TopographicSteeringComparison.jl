# f/H steering over a continental slope recovered by the PG solution, and where
# it fails on a shallow shelf (manuscript Section 6c).
#
# On an f-plane a flat-topped shelf (H ≈ 0.1) in the north falls across a
# slope to the deep basin (H = 1) in the south. An isolated Gaussian patch of
# wind-stress curl on the slope drives a topographic β plume that runs west
# along the isobaths, shallow water to the right. There is no buoyancy forcing.
#
# The classical layer model is the deep limit of the PG transport (Section 6a):
#     f (U - U_E) = -a ∇p_s + b ẑ×∇p_s,   U_E = Ek (τ^y, -τ^x)/f,
# geostrophic transport over the depth b plus the down-gradient bottom Ekman
# transport, with a = δ_E/2 from the derived bottom stress. With U = ẑ×∇Ψ and
# G = f (U - U_E), ∇p_s = -(a G + b ẑ×G)/(a² + b²), and curl ∇p_s = 0 gives
#     ∇·(K ∇Ψ) + Q_y ∂xΨ - Q_x ∂yΨ = curl(c1 G_E) + ∇·(c2 G_E),
#     c1 = a/(a² + b²), c2 = b/(a² + b²), K = c1 f, Q = c2 f, G_E = f U_E,
# with Ψ = 0 on the coast. For a ≪ b this is steering along f/b with drag
# f a / b². It is solved by second-order finite differences, with
#   classical: b = H, steering along f/H;
#   corrected: b = H - δ_E/2, the displacement thickness (Section 6a).
# Ek is swept so that the shelf has s = H/δ_E = 1, 2, and 5, while the deep
# basin has s = 10 to 50. The PG streamfunction is integrated from the PG
# transport, Ψ(x) = -∫_x^{east} V dx'.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/TopographicSteeringComparison.jl
# Writes TheoryNotes/topographic_steering_comparison.{pdf,png}.

ENV["GKSwstype"] = "100"
using Logging
using PlanetaryGeostrophy
using Plots
using Printf
using SparseArrays
include("ComparisonTools.jl")

global_logger(ConsoleLogger(stderr, Logging.Warn))

const Lx, Ly = 4.0, 2.0
const f0 = 1.0
const Γ = 0.2
const H_shelf, H_deep = 0.1, 1.0
const y_break, w_slope = 0.3, 0.25
const centre = (1.0, y_break)
const radius = 0.15
const curl_amplitude = 8e-4

depth(x, y) = H_shelf + (H_deep - H_shelf) * (1 - tanh((y - y_break) / w_slope)) / 2

# Azimuthal stress whose curl is a Gaussian.
const tau, wind_curl = gaussian_curl_stress(centre, radius, curl_amplitude)

problem(Ek) = PGProblem(
    parameters=PGParameters(Ek, Γ),
    depth=depth,
    coriolis=(x, y) -> f0,
    wind_stress=tau,
    wind_stress_curl=wind_curl,
    restoring_buoyancy=(x, y, z) -> 0.0,
    surface_buoyancy=(x, y) -> 0.0,
    boundary=rectangle_boundary(Lx, Ly),
)

# Layer model on a uniform grid; returns Ψ at the nodes, walls included.
function layer_model(Ek, effective_depth; Δ=0.005)
    δ = sqrt(2Ek / f0)
    a = δ / 2
    b(x, y) = effective_depth(depth(x, y), δ)
    c1(x, y) = a / (a^2 + b(x, y)^2)
    c2(x, y) = b(x, y) / (a^2 + b(x, y)^2)
    K(x, y) = c1(x, y) * f0
    Q(x, y) = c2(x, y) * f0
    GE(x, y) = (Ek * imag(tau(x, y)), -Ek * real(tau(x, y)))     # f U_E
    nx, ny = round(Int, Lx / Δ), round(Int, Ly / Δ)
    x = range(-Lx / 2, Lx / 2; length=nx + 1)
    y = range(-Ly / 2, Ly / 2; length=ny + 1)
    index(i, j) = (j - 2) * (nx - 1) + (i - 1)
    n = (nx - 1) * (ny - 1)
    rows, cols, vals = Int[], Int[], Float64[]
    rhs = zeros(n)
    for j in 2:ny, i in 2:nx
        k = index(i, j)
        xx, yy = x[i], y[j]
        Ke, Kw = K(xx + Δ / 2, yy), K(xx - Δ / 2, yy)
        Kn, Ks = K(xx, yy + Δ / 2), K(xx, yy - Δ / 2)
        Qx = (Q(xx + Δ, yy) - Q(xx - Δ, yy)) / 2Δ
        Qy = (Q(xx, yy + Δ) - Q(xx, yy - Δ)) / 2Δ
        push!(rows, k); push!(cols, k); push!(vals, -(Ke + Kw + Kn + Ks) / Δ^2)
        for (p, q, c) in ((i + 1, j, Ke / Δ^2 + Qy / 2Δ), (i - 1, j, Kw / Δ^2 - Qy / 2Δ),
                          (i, j + 1, Kn / Δ^2 - Qx / 2Δ), (i, j - 1, Ks / Δ^2 + Qx / 2Δ))
            (2 <= p <= nx && 2 <= q <= ny) || continue
            push!(rows, k); push!(cols, index(p, q)); push!(vals, c)
        end
        # curl(c1 G_E) + ∇·(c2 G_E), centred differences.
        F1(x, y) = c1(x, y) .* GE(x, y)
        F2(x, y) = c2(x, y) .* GE(x, y)
        curl = (F1(xx + Δ, yy)[2] - F1(xx - Δ, yy)[2] - F1(xx, yy + Δ)[1] + F1(xx, yy - Δ)[1]) / 2Δ
        divergence = (F2(xx + Δ, yy)[1] - F2(xx - Δ, yy)[1] + F2(xx, yy + Δ)[2] - F2(xx, yy - Δ)[2]) / 2Δ
        rhs[k] = curl + divergence
    end
    Ψ = zeros(nx + 1, ny + 1)
    Ψ[2:nx, 2:ny] = reshape(sparse(rows, cols, vals, n, n) \ rhs, nx - 1, ny - 1)
    return x, y, Ψ
end

classical_depth(H, δ) = H
corrected_depth(H, δ) = H - δ / 2

function compare(Ek; mesh_resolution=0.015)
    δ = sqrt(2Ek / f0)
    seconds = @elapsed solution = solve_pg(problem(Ek);
        mesh_file=joinpath(tempdir(), "pg_topographic_steering.msh"), mesh_resolution, overwrite_mesh=true)
    x = clamp.(collect(range(-Lx / 2, Lx / 2; length=801)), -Lx / 2 + 1e-6, Lx / 2 - 1e-6)
    y = clamp.(collect(range(-Ly / 2, Ly / 2; length=401)), -Ly / 2 + 1e-6, Ly / 2 - 1e-6)
    Ψ_pg = pg_streamfunction(solution, x, y)
    on_grid(model) = [interpolate_grid(model..., xx, yy) for xx in x, yy in y]
    Ψ_classical = on_grid(layer_model(Ek, classical_depth))
    Ψ_corrected = on_grid(layer_model(Ek, corrected_depth))
    H = [depth(xx, yy) for xx in x, yy in y]
    shelf, deep = H .< 2H_shelf, H .> 0.6H_deep
    region(F, mask) = F[mask]
    errors(model) = (; all=relative_l2(Ψ_pg, model),
        deep=relative_l2(region(Ψ_pg, deep), region(model, deep)),
        shelf=relative_l2(region(Ψ_pg, shelf), region(model, shelf)))
    return (; Ek, δ, s_shelf=H_shelf / δ, s_deep=H_deep / δ, mesh_resolution, seconds, x, y,
        Ψ_pg, Ψ_classical, Ψ_corrected, classical=errors(Ψ_classical), corrected=errors(Ψ_corrected))
end

let Ek = 1.25e-3
    coarse, fine = layer_model(Ek, corrected_depth; Δ=0.01), layer_model(Ek, corrected_depth; Δ=0.005)
    @printf("FD check (Ek %.2e): relative change from Δ = 0.01 to 0.005: %.1e\n", Ek,
        relative_l2(coarse[3], fine[3][1:2:end, 1:2:end]))
    a, b = compare(Ek; mesh_resolution=0.02), compare(Ek; mesh_resolution=0.01)
    @printf("PG mesh check (Ek %.2e): relative change from mesh 0.02 to 0.01: %.1e\n", Ek,
        relative_l2(a.Ψ_pg, b.Ψ_pg))
end

cases = map([5e-3, 1.25e-3, 2e-4]) do Ek
    c = compare(Ek)
    @printf("Ek %.2e  s shelf %.1f deep %.0f | Ψ relative L2 (all / deep / shelf): f/H %.2e %.2e %.2e | f/(H-δ/2) %.2e %.2e %.2e (PG solve %.1f s)\n",
        c.Ek, c.s_shelf, c.s_deep, c.classical.all, c.classical.deep, c.classical.shelf,
        c.corrected.all, c.corrected.deep, c.corrected.shelf, c.seconds)
    c
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
reference = cases[2]
style = (; titlefontsize=10, guidefontsize=9, framestyle=:box, legendfontsize=7)
extremum = max(maximum(abs, reference.Ψ_pg), maximum(abs, reference.Ψ_classical))
levels = range(-extremum, extremum; length=17)
mapstyle = (; aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", colorbar=false, levels,
    color=:balance, xlims=(-Lx / 2, Lx / 2), ylims=(-Ly / 2, Ly / 2))
isobaths = [0.2, 0.4, 0.6, 0.8]
source = [(centre[1] + radius * cos(t), centre[2] + radius * sin(t)) for t in range(0, 2pi; length=60)]
function map_panel(Ψ, title)
    p = contourf(reference.x, reference.y, Ψ'; title, mapstyle..., style...)
    contour!(p, reference.x, reference.y, [depth(xx, yy) for xx in reference.x, yy in reference.y]';
        levels=isobaths, color=:gray, ls=:dash, lw=0.8, colorbar=false)
    plot!(p, first.(source), last.(source); color=:black, ls=:dot, label="")
    # Ψ = 0 separates the plume from the weak recirculation east of the source.
    # It is drawn in the interior only: on the walls Ψ = 0 to round-off, and the
    # sign of that round-off would trace the walls.
    interior = 3:length(reference.x)-2, 3:length(reference.y)-2
    contour!(p, reference.x[interior[1]], reference.y[interior[2]], Ψ[interior...]';
        levels=[0.0], color=:black, ls=:dashdot, lw=1.8, colorbar=false)
    return p
end
panel_a = map_panel(reference.Ψ_pg, @sprintf("(a) PG \$\\Psi\$, shelf \$s = %.0f\$", reference.s_shelf))
panel_b = map_panel(reference.Ψ_classical, "(b) layer model, steering along \$f/H\$")

section_x = -0.5
panel_c = plot(; xlabel="\$y\$", ylabel="\$\\Psi / \\max|\\Psi_{\\mathrm{PG}}|\$",
    title="(c) section across the plume, \$x = $(section_x)\$", legend=:bottomleft, style...)
for (k, c) in enumerate(cases)
    i = argmin(abs.(c.x .- section_x))
    scale = maximum(abs, c.Ψ_pg[i, :])
    plot!(panel_c, c.y, c.Ψ_pg[i, :] ./ scale; color=k, lw=1.5,
        label=@sprintf("PG, shelf \$s = %.0f\$", c.s_shelf))
    plot!(panel_c, c.y, c.Ψ_classical[i, :] ./ scale; color=k, ls=:dash, lw=1, label="")
end
plot!(panel_c, [NaN], [NaN]; color=:black, ls=:dash, label="layer model, \$f/H\$")
vspan!(panel_c, [y_break + 2w_slope, Ly / 2]; color=:gray, alpha=0.15, label="shelf")

s_values = [c.s_shelf for c in cases]
panel_d = plot(; xscale=:log10, yscale=:log10, xlabel="shelf \$s = H_{\\mathrm{shelf}}/\\delta_E\$",
    ylabel="relative \$L_2\$ difference in \$\\Psi\$", title="(d) PG against the layer model",
    xticks=(s_values, [@sprintf("%.0f", v) for v in s_values]), legend=:bottomleft, style...)
for (label, field, marker) in (("deep, \$f/H\$", c -> c.classical.deep, :circle),
                               ("deep, \$f/(H - \\delta_E/2)\$", c -> c.corrected.deep, :square),
                               ("shelf, \$f/H\$", c -> c.classical.shelf, :utriangle),
                               ("shelf, \$f/(H - \\delta_E/2)\$", c -> c.corrected.shelf, :diamond))
    plot!(panel_d, s_values, field.(cases); marker, lw=1.5, label)
end

figure = plot(panel_a, panel_b, panel_c, panel_d; layout=(2, 2), size=(1100, 720),
    left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "topographic_steering_comparison.$extension"))
end
