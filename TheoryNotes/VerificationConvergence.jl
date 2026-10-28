# Mesh-convergence verification of the PG solution (manuscript Section 6b).
#
# A sloping beta-plane basin with a seamount, a vertical coastal wall
# (H > 0 at the coast), surface stress, and depth-dependent horizontal
# buoyancy forcing is solved on successively finer surface-pressure meshes.
# Errors are measured against the finest mesh. The script also reports the
# mapped barotropic and geostrophic-contour vorticity budget residuals with the
# cell-average and centered-difference methods at a fixed diagnostic grid.
#
# Run from the repository root with threads:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/VerificationConvergence.jl
# Writes TheoryNotes/verification_convergence.{pdf,png} and prints a table.

ENV["GKSwstype"] = "100"
using Logging
using PlanetaryGeostrophy
using Plots
using Printf

global_logger(ConsoleLogger(stderr, Logging.Warn))

include("VerificationBasin.jl")
include("ComparisonTools.jl")
include("FigureStyle.jl")
problem = verification_problem()

resolutions = [0.16, 0.08, 0.04, 0.02]
reference_resolution = 0.01

# Comparison points: a regular grid inside 97% of the coastal radius.
sample_x = range(-0.97ax, 0.97ax; length=49)
sample_y = range(-0.97ay, 0.97ay; length=37)
points = [(x, y) for x in sample_x, y in sample_y if (x / ax)^2 + (y / ay)^2 < 0.97^2]

# Fixed diagnostic grid for streamfunction and budget maps.
diagnostic_x = range(-0.9ax, 0.9ax; length=19)
diagnostic_y = range(-0.9ay, 0.9ay; length=15)

function sample_fields(solution)
    surface = [surface_pressure(solution, x, y) for (x, y) in points]
    current = [depth_integrated_current(solution, x, y) for (x, y) in points]
    stress = [bottom_stress(solution, x, y) for (x, y) in points]
    middepth = [
        horizontal_velocity(solution, x, y, -0.5problem.depth(x, y))
        for (x, y) in points
    ]
    diagnostics = compute_diagnostics(solution, diagnostic_x, diagnostic_y)
    budgets = Dict(
        method => (
            bv=compute_barotropic_vorticity_diagnostics(solution, diagnostics; method),
            gcvb=compute_geostrophic_contour_vorticity_diagnostics(
                solution, diagnostics; method,
            ),
        ) for method in (:cell_average, :centered)
    )
    return (;
        p_s=surface,
        U=[complex(c.U, c.V) for c in current],
        τ_b=[complex(s.x, s.y) for s in stress],
        u_mid=[complex(u.u, u.v) for u in middepth],
        Ψ=diagnostics.streamfunction[diagnostics.wet],
        wet=diagnostics.wet,
        budgets,
        timings=solution.timings,
    )
end


# Orders are computed from the realized mesh size, the mean triangle edge
# length reported by `solve_pg`, which is close to `mesh_resolution`.

solutions = Dict{Float64,Any}()
for h in vcat(resolutions, reference_resolution)
    seconds = @elapsed begin
        solution = solve_pg(
            problem;
            mesh_file=joinpath(tempdir(), "verification_$(h).msh"),
            mesh_resolution=h,
            overwrite_mesh=true,
        )
        solutions[h] = (;
            sample_fields(solution)...,
            edge=solution.timings.surface_pressure.mean_edge_length,
        )
    end
    @printf("mesh_resolution = %.3f (mean edge %.4f) solved and sampled in %.1f s\n",
        h, solutions[h].edge, seconds)
end

reference = solutions[reference_resolution]
fields = (:p_s, :U, :τ_b, :u_mid)
errors = Dict(
    field => [relative_l2(getproperty(solutions[h], field), getproperty(reference, field))
              for h in resolutions]
    for field in fields
)
errors[:Ψ] = [
    begin
        wet = solutions[h].wet .& reference.wet
        relative_l2(
            solutions[h].Ψ[wet[solutions[h].wet]], reference.Ψ[wet[reference.wet]],
        )
    end for h in resolutions
]
residual(h, method, budget) =
    getproperty(solutions[h].budgets[method], budget).timings.relative_l2_residual
all_h = vcat(resolutions, reference_resolution)
edge = [solutions[h].edge for h in resolutions]
all_edge = [solutions[h].edge for h in all_h]

println()
println("Relative L2 error against the finest mesh (mean edge $(round(solutions[reference_resolution].edge; digits=4))):")
@printf("%8s %8s %10s %10s %10s %10s %10s\n", "nominal", "edge", "p_s", "U", "τ_b", "u(-H/2)", "Ψ")
for (k, h) in enumerate(resolutions)
    @printf("%8.3f %8.4f %10.2e %10.2e %10.2e %10.2e %10.2e\n", h, edge[k],
        errors[:p_s][k], errors[:U][k], errors[:τ_b][k], errors[:u_mid][k], errors[:Ψ][k])
end
println("Observed orders between successive meshes, log(e1/e2)/log(h1/h2) with mean edge h:")
for field in (:p_s, :U, :τ_b, :u_mid, :Ψ)
    e = errors[field]
    orders = [log(e[k] / e[k + 1]) / log(edge[k] / edge[k + 1]) for k in 1:length(e) - 1]
    @printf("  %-6s %s\n", field, join((@sprintf("%.2f", o) for o in orders), ", "))
end
println("Mapped budget relative residuals (cell average / centered):")
@printf("%8s %12s %12s %12s %12s\n", "edge", "BV cell", "BV centered", "GCVB cell", "GCVB centered")
for h in all_h
    @printf("%8.4f %12.2e %12.2e %12.2e %12.2e\n", solutions[h].edge,
        residual(h, :cell_average, :bv), residual(h, :centered, :bv),
        residual(h, :cell_average, :gcvb), residual(h, :centered, :gcvb))
end

const figure_size = (1000, 420)
common = (; xscale=:log10, yscale=:log10, xlabel="mean triangle edge length \$h\$",
    legend=:bottomright, minorgrid=false, xflip=false, manuscript_style(figure_size[1], 16)...)
labels = Dict(:p_s => "\$p_s\$", :U => "\$\\mathbf{U}\$", :τ_b => "\$\\tau_b\$",
    :u_mid => "\$\\mathbf{u}(-H/2)\$", :Ψ => "\$\\Psi\$")
tick_labels(values) = (values, [@sprintf("%.3f", v) for v in values])
convergence = plot(; title="(a) error against the finest mesh",
    ylabel="relative \$L_2\$ error", xticks=tick_labels(edge), common...)
for (field, marker) in zip((:p_s, :U, :τ_b, :u_mid, :Ψ), (:circle, :square, :diamond, :utriangle, :star5))
    plot!(convergence, edge, errors[field]; marker, lw=1.5, label=labels[field])
end
plot!(convergence, edge, errors[:U][end] .* (edge ./ edge[end]) .^ 2;
    color=:gray, ls=:dash, label="\$\\propto h^2\$")
plot!(convergence, edge, errors[:p_s][end] .* (edge ./ edge[end]) .^ 3;
    color=:gray, ls=:dot, label="\$\\propto h^3\$")

budgets = plot(; title="(b) mapped vorticity-budget residuals",
    ylabel="relative \$L_2\$ residual", xticks=tick_labels(all_edge),
    common..., legend=:bottomright)
plot!(budgets, all_edge, [residual(h, :cell_average, :bv) for h in all_h];
    marker=:circle, lw=1.5, color=:black, label="BV, cell average")
plot!(budgets, all_edge, [residual(h, :cell_average, :gcvb) for h in all_h];
    marker=:square, lw=1.5, color=:steelblue, label="GCVB, cell average")
plot!(budgets, all_edge, [residual(h, :centered, :bv) for h in all_h];
    marker=:circle, lw=1.5, ls=:dash, color=:black, label="BV, centered")
plot!(budgets, all_edge, [residual(h, :centered, :gcvb) for h in all_h];
    marker=:square, lw=1.5, ls=:dash, color=:steelblue, label="GCVB, centered")

figure = plot(convergence, budgets; layout=(1, 2), size=figure_size,
    left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "verification_convergence.$extension"))
end
