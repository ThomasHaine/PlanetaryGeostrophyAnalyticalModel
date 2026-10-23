# Budget closure in the double basin (manuscript Section 6b, "Dynamical
# balances"), with the geometry and forcing of TheoryNotes/DoubleBasinShowcase.jl:
#   - mapped BV and GCVB residuals with centered differences and with cell
#     averages, on several diagnostic grids;
#   - the GCVB integrated over the regions enclosed by closed f/H contours;
#   - the stress circulation around closed isobaths;
#   - the GCVB integrated over the northern and southern halves of the basin.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy -t auto TheoryNotes/DoubleBasinBudgets.jl

source = read(joinpath(@__DIR__, "DoubleBasinShowcase.jl"), String)
include_string(Main, source[1:findfirst("# Calculation, timed", source).start-1],
    joinpath(@__DIR__, "DoubleBasinShowcase.jl"))
for name in ("GCVBDiagnostics", "IsobathCirculationDiagnostics", "IslandRuleDiagnostics")
    include(joinpath(@__DIR__, "..", "AnalyticalModel", "$name.jl"))
end

solution = solve_pg(problem; mesh_file=joinpath(tempdir(), "double_basin_budgets.msh"),
    mesh_resolution, overwrite_mesh=true)

println("Mapped residuals (relative L2) on n × n diagnostic grids inside 0.94 of the coastal radius:")
@printf("  %5s  %22s  %22s\n", "n", "BV centered / cell", "GCVB centered / cell")
reference = nothing
for n in (17, 33, 65, 129)
    d = compute_diagnostics(solution, range(-0.94ax, 0.94ax; length=n), range(-0.94ay, 0.94ay; length=n))
    r(f, method) = f(solution, d; method).timings.relative_l2_residual
    @printf("  %5d  %10.2e / %9.2e  %10.2e / %9.2e\n", n,
        r(compute_barotropic_vorticity_diagnostics, :centered), r(compute_barotropic_vorticity_diagnostics, :cell_average),
        r(compute_geostrophic_contour_vorticity_diagnostics, :centered), r(compute_geostrophic_contour_vorticity_diagnostics, :cell_average))
    n == 65 && (global reference = d)
end

gcvb = compute_geostrophic_contour_vorticity_diagnostics(solution, reference)
area = GCVBDiagnostics.compute_gcvb_area_consistency(solution, gcvb)
isobath = IsobathCirculationDiagnostics.compute_isobath_circulation_consistency(solution, reference)
island = IslandRuleDiagnostics.compute_wall_cut_island_rule(solution, reference, gcvb; cut_y=0.0)
println("Closed-contour integrals (65 × 65 grid):")
@printf("  GCVB over closed f/H contours: max relative residual %.1e (advection), %.1e (forcing)\n",
    area.metrics.max_relative_advection_residual, area.metrics.max_relative_forcing_residual)
@printf("  stress circulation around closed isobaths: max relative residual %.1e\n",
    isobath.metrics.max_relative_residual)
@printf("  GCVB over the northern and southern halves: relative residual %.1e and %.1e\n",
    island.metrics.north_local_balance_relative_residual, island.metrics.south_local_balance_relative_residual)
