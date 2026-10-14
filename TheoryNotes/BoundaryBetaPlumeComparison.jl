# Boundary beta plume of Haine and Fuller (2016) recovered by the PG solution
# (manuscript Section 6c).
#
# A flat basin on a beta plane is forced by an isolated Gaussian patch of
# wind-stress curl at the origin. A straight coast C, inclined at angle θ to
# east–west and a perpendicular distance d_s from the source, intersects the
# westward plume. With Rayleigh drag λ the classical streamfunction solves
#     β ∂xΨ + λ ∇²Ψ + Q = 0,   Q = -Ek curl τ,   Ψ = 0 on C,
# and Haine and Fuller (2016) give the exact solution as a source plus a
# weighted image source,
#     G_b(x; xs) = G(x; xs) - G(x; xi) exp(d sinθ / δ_S),
#     G(x; xs) = K0(|x - xs| / 2δ_S) exp((xs - x) / 2δ_S) / (2π λ),
# with δ_S = λ/β, d the distance from xs to C, and xi the reflection of xs in
# C. Far down-plume, along the line through the source parallel to C, it decays
# as (-s)^{-3/2} exp(-|s| (1 - cos θ) / 2δ_S), with s the distance along C. The classical Ψ is the convolution of G_b
# with Q over the source patch. The drag is λ = f0 δ_E / (2H), the deep limit
# of the PG bottom stress, so there is no fitted parameter. β is small, so f,
# and with it the Ekman drag, varies by only about ±7% over the domain.
#
# The PG basin is a box cut by C; the plume region is far from the other walls.
# The PG streamfunction is integrated from the PG transport westward from the
# eastern wall, Ψ(x) = -∫_x^{east} V dx'.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/BoundaryBetaPlumeComparison.jl
# SpecialFunctions is loaded from the AnalyticalModel environment, stacked on
# the package environment. Writes TheoryNotes/boundary_beta_plume_comparison.{pdf,png}.

push!(LOAD_PATH, joinpath(@__DIR__, "..", "AnalyticalModel"))
ENV["GKSwstype"] = "100"
using Logging
using PlanetaryGeostrophy
using Plots
using Printf
using SpecialFunctions

global_logger(ConsoleLogger(stderr, Logging.Warn))

const H0 = 1.0
const f0, beta = 1.0, 0.1
const Γ = 0.2
const Ek = 1.28e-4
const δ_E = sqrt(2Ek / f0)
const λ = f0 * δ_E / (2H0)          # derived Rayleigh drag
const δ_S = λ / beta                 # Stommel length
const σ = 0.1                        # source radius
const curl_amplitude = 8e-4
const d_s = 5δ_S                     # source distance from the coast
const box = (-3.5, 1.0, -2.0, 1.5)   # xmin, xmax, ymin, ymax
const pg_mesh_resolution = 0.02

# Azimuthal stress whose curl is exactly the Gaussian below.
function stress_factor(x, y)
    r2 = x^2 + y^2
    r2 < 1e-14 && return curl_amplitude / 2
    return curl_amplitude * σ^2 * (-expm1(-r2 / (2σ^2))) / r2
end
tau(x, y) = complex(-stress_factor(x, y) * y, stress_factor(x, y) * x)
wind_curl(x, y) = curl_amplitude * exp(-(x^2 + y^2) / (2σ^2))

# The box clipped to the half plane n·x ≤ d_s, n = (-sinθ, cosθ), counterclockwise.
function basin(θ)
    n = (-sin(θ), cos(θ))
    xmin, xmax, ymin, ymax = box
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    side(p) = d_s - (n[1] * p[1] + n[2] * p[2])
    polygon = Tuple{Float64,Float64}[]
    for k in eachindex(corners)
        p, q = corners[k], corners[mod1(k + 1, length(corners))]
        side(p) >= 0 && push!(polygon, p)
        if side(p) * side(q) < 0
            t = side(p) / (side(p) - side(q))
            push!(polygon, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
        end
    end
    # Points along C, so that the mesh resolves the coast where the plume meets it.
    refined = Tuple{Float64,Float64}[]
    for k in eachindex(polygon)
        p, q = polygon[k], polygon[mod1(k + 1, length(polygon))]
        m = max(1, ceil(Int, hypot(q[1] - p[1], q[2] - p[2]) / pg_mesh_resolution))
        append!(refined, [(p[1] + (q[1] - p[1]) * j / m, p[2] + (q[2] - p[2]) * j / m) for j in 0:m-1])
    end
    return refined
end

problem(θ) = PGProblem(
    parameters=PGParameters(Ek, Γ),
    depth=(x, y) -> H0,
    coriolis=(x, y) -> f0 + beta * y,
    wind_stress=tau,
    wind_stress_curl=wind_curl,
    restoring_buoyancy=(x, y, z) -> 0.0,
    surface_buoyancy=(x, y) -> 0.0,
    boundary=basin(θ),
)

# Free-space plume; besselkx(0, ρ) = K0(ρ) e^ρ avoids overflow far down-plume.
function plume(x, y, xs, ys)
    ρ = hypot(x - xs, y - ys) / (2δ_S)
    return besselkx(0, ρ) * exp((xs - x) / (2δ_S) - ρ) / (2pi * λ)
end

# Haine and Fuller (2016) boundary plume, convolved with Q = -Ek curl τ.
function classical_streamfunction(θ, x, y; spacing=σ / 6)
    n = (-sin(θ), cos(θ))
    offsets = (-3.5σ + spacing / 2):spacing:(3.5σ)
    sources = [(xs, ys, -Ek * wind_curl(xs, ys) * spacing^2)
               for xs in offsets, ys in offsets if hypot(xs, ys) <= 3.5σ]
    Ψ = zeros(length(x), length(y))
    Threads.@threads for j in eachindex(y)
        for i in eachindex(x), (xs, ys, weight) in sources
            d = d_s - (n[1] * xs + n[2] * ys)
            xi, yi = xs + 2d * n[1], ys + 2d * n[2]
            Ψ[i, j] += weight * (plume(x[i], y[j], xs, ys) -
                plume(x[i], y[j], xi, yi) * exp(d * sin(θ) / δ_S))
        end
    end
    return Ψ
end

function pg_streamfunction(solution, inside, x, y)
    Ψ = fill(NaN, length(x), length(y))
    Threads.@threads for j in eachindex(y)
        inside(x[end], y[j]) || continue
        Ψ[end, j] = 0.0
        V_east = depth_integrated_current(solution, x[end], y[j]).V
        for i in length(x)-1:-1:1
            inside(x[i], y[j]) || break
            V = depth_integrated_current(solution, x[i], y[j]).V
            Ψ[i, j] = Ψ[i+1, j] - 0.5 * (V + V_east) * (x[i+1] - x[i])
            V_east = V
        end
    end
    return Ψ
end

relative_l2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

function compare(θ)
    seconds = @elapsed solution = solve_pg(problem(θ);
        mesh_file=joinpath(tempdir(), "pg_boundary_plume.msh"),
        mesh_resolution=pg_mesh_resolution, overwrite_mesh=true)
    inside = solution.problem.inside
    xmin, xmax, ymin, ymax = box
    x = collect(range(xmin + 1e-6, xmax - 1e-6; length=451))
    y = collect(range(ymin + 1e-6, ymax - 1e-6; length=351))
    Ψ_pg = pg_streamfunction(solution, inside, x, y)
    Ψ_classical = classical_streamfunction(θ, x, y)
    wet = isfinite.(Ψ_pg)
    Ψ_classical[.!wet] .= NaN
    # Along the line through the source parallel to C (d = 0), down-plume.
    along = collect(range(-0.2, -3.5; step=-0.01))
    points = [(s * cos(θ), s * sin(θ)) for s in along]
    keep = [inside(p...) && p[2] > ymin + 0.3 && p[1] > xmin + 0.3 for p in points]
    along, points = along[keep], points[keep]
    Ψ_line_pg = [interpolate_grid(x, y, Ψ_pg, p...) for p in points]
    Ψ_line_classical = [interpolate_grid(x, y, Ψ_classical, p...) for p in points]
    # e-folding length down-plume, removing the (-s)^{-3/2} prefactor of the
    # Haine and Fuller asymptote, where |Ψ| is above 1e-3 of its peak.
    fit = (along .< -1.5δ_S / (1 - cos(θ))) .& (abs.(Ψ_line_pg) .> 1e-3 * maximum(abs, Ψ_line_pg))
    efold(line) = sum(fit) > 5 ?
        1 / linear_slope(along[fit], log.(abs.(line[fit])) .+ 1.5 .* log.(-along[fit])) : NaN
    return (; θ, seconds, x, y, Ψ_pg, Ψ_classical, wet, along, Ψ_line_pg, Ψ_line_classical,
        error=relative_l2(Ψ_pg[wet], Ψ_classical[wet]),
        efold_pg=efold(Ψ_line_pg), efold_classical=efold(Ψ_line_classical),
        efold_theory=2δ_S / (1 - cos(θ)))
end

function interpolate_grid(x, y, F, px, py)
    i = clamp(searchsortedlast(x, px), 1, length(x) - 1)
    j = clamp(searchsortedlast(y, py), 1, length(y) - 1)
    a, b = (px - x[i]) / (x[i+1] - x[i]), (py - y[j]) / (y[j+1] - y[j])
    return (1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i+1, j] +
        (1 - a) * b * F[i, j+1] + a * b * F[i+1, j+1]
end

function linear_slope(s, v)
    s̄, v̄ = sum(s) / length(s), sum(v) / length(v)
    return sum((s .- s̄) .* (v .- v̄)) / sum(abs2, s .- s̄)
end

@printf("Ek %.2e, s = H/δ_E %.0f, λ = f0 δ_E/2H %.4f, δ_S = λ/β %.4f, d_s %.3f, source σ %.2f\n",
    Ek, H0 / δ_E, λ, δ_S, d_s, σ)
let
    θ = deg2rad(30)
    x, y = collect(range(-1, 0.5; length=31)), collect(range(-1, 0.3; length=27))
    coarse, fine = classical_streamfunction(θ, x, y), classical_streamfunction(θ, x, y; spacing=σ / 12)
    @printf("Source quadrature: relative change from spacing σ/6 to σ/12: %.1e\n", relative_l2(coarse, fine))
end

cases = map((30, 60)) do degrees
    c = compare(deg2rad(degrees))
    @printf("θ = %2d°: Ψ relative L2 %.2e | down-plume e-folding: PG %.3f, Haine & Fuller solution %.3f, asymptote 2δ_S/(1 - cos θ) = %.3f (PG solve %.1f s)\n",
        degrees, c.error, c.efold_pg, c.efold_classical, c.efold_theory, c.seconds)
    c
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
style = (; titlefontsize=10, guidefontsize=9, framestyle=:box, legendfontsize=7)
panels = []
for c in cases
    levels = range(minimum(filter(isfinite, c.Ψ_pg)), 0; length=13)
    mapstyle = (; aspect_ratio=:equal, xlabel="\$x\$", ylabel="\$y\$", colorbar=false, levels,
        color=:viridis, xlims=(box[1], box[2]), ylims=(box[3], box[4]))
    degrees = round(Int, rad2deg(c.θ))
    line = [(s * cos(c.θ), s * sin(c.θ)) for s in (0.0, -3.5)]
    a = contourf(c.x, c.y, c.Ψ_pg'; title="PG \$\\Psi\$, \$\\theta = $(degrees)^\\circ\$", mapstyle..., style...)
    plot!(a, first.(line), last.(line); color=:white, ls=:dot, label="")
    b = contourf(c.x, c.y, c.Ψ_classical'; title="Haine & Fuller (2016), \$\\theta = $(degrees)^\\circ\$", mapstyle..., style...)
    plot!(b, first.(line), last.(line); color=:white, ls=:dot, label="")
    scale = maximum(abs, c.Ψ_line_classical)
    profile = plot(-c.along ./ δ_S, abs.(c.Ψ_line_pg) ./ scale; yscale=:log10, lw=2, color=:black,
        label="PG", xlabel="distance down-plume, \$-s/\\delta_S\$", ylabel="\$|\\Psi| / \\max|\\Psi_{\\mathrm{HF}}|\$",
        title="along the line through the source, \$\\theta = $(degrees)^\\circ\$", legend=:bottomleft, style...)
    plot!(profile, -c.along ./ δ_S, abs.(c.Ψ_line_classical) ./ scale; lw=1.5, ls=:dash, color=1, label="Haine & Fuller")
    s_ref = -c.along[end÷2]
    ref = abs(c.Ψ_line_classical[end÷2]) / scale
    plot!(profile, -c.along ./ δ_S,
        ref .* (-c.along ./ s_ref) .^ -1.5 .* exp.(-(-c.along .- s_ref) ./ c.efold_theory); color=:gray,
        lw=1, label="far-field asymptote")
    append!(panels, [a, b, profile])
end
figure = plot(panels...; layout=(2, 3), size=(1300, 720), left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "boundary_beta_plume_comparison.$extension"))
end
