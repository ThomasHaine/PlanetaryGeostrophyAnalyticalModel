PlanetaryGeostrophyAnalyticalModel

* LaTeX notes on planetary geostrophy with analytic solutions for certain cases.
* Julia code to support the theory derivation using `Symbolics.jl`
* Julia code to compute solutions using analytic results and `Gridap.jl` solutions of the surface pressure and depth-integrated streamfunction PDEs

See also: [this archived repo](https://github.com/ThomasHaine/ArcticOceanDynamicsWorkshop25), on which this one is based.

  Questions/comments welcome.

## Non-dimensional PG API

The reusable Julia API is a standalone package in the root-level
`PlanetaryGeostrophy/` directory. Its project contains only the
analytical solver dependencies; Oceananigans remains confined to the separate
validation environment.
It packages the B23 Green-function velocity and buoyancy kernels used by
`PG_Solution_v0.1.ipynb` with the Gridap surface-pressure solve used by
`Oceananigans_Variable_Basin_Validation.ipynb`.

From the repository root, activate the standalone API environment:

```julia
using Pkg
Pkg.activate("PlanetaryGeostrophy")
Pkg.instantiate()
using PlanetaryGeostrophy
```

A problem is specified entirely with non-dimensional parameters and functions:

```julia
parameters = PGParameters(0.02, 0.2)
boundary = rectangle_boundary(2.0, 1.5)

problem = PGProblem(
    parameters=parameters,
    depth=(x, y) -> 1.0,
    coriolis=(x, y) -> 1.0 + 0.15y,
    wind_stress=(x, y) -> 1e-3 * cos(pi * y / 1.5) + 0im,
    restoring_buoyancy=(x, y, z) -> 1e-2z,
    surface_buoyancy=(x, y) -> 0.0,
    boundary=boundary,
)

solution = solve_pg(problem; mesh_file="sample_pressure.msh")
fields = sample_solution(
    solution,
    range(-0.9, 0.9; length=9),
    range(-0.65, 0.65; length=7),
    range(-1.0, 0.0; length=9),
)
write_solution(fields, "sample_pg")
plot_solution(fields; z_index=5, output="sample_pg.png")
```

`wind_stress` uses the complex convention `τx + im * τy`. Custom analytic
derivatives can be passed as `depth_x`, `depth_y`, `coriolis_x`, and
`coriolis_y`; centered derivatives are generated otherwise. The sampled
solution contains `u`, `v`, `w`, and `b` on the 3D grid plus surface pressure
and depth. Vertical velocity is reconstructed from discrete horizontal
divergence with no-normal-flow conditions at the bottom and rigid lid.

The solver logs each mesh, finite-element, field-reconstruction, output, and
plotting stage with elapsed time, allocation, and garbage-collection metrics.
The solve and sampling metrics are also available programmatically:

```julia
performance_summary(solution)
performance_summary(fields)
files.timings
```

Bottom stress, depth-integrated current, and its streamfunction are available
as point queries or sampled maps. These use equations (62)--(69) and the
streamfunction weak form (C2) in the model derivation:

```julia
stress = bottom_stress(solution, x, y)
current = depth_integrated_current(solution, x, y)
ψ = solve_streamfunction(solution)
ψxy = streamfunction(ψ, x, y)
current_from_ψ = streamfunction_current(ψ, x, y)

diagnostics = compute_diagnostics(solution, xs, ys)
plot_diagnostics(
    diagnostics;
    arrow_scale=0.35,
    output="sample_pg_diagnostics.png",
)
plot_bottom_stress(diagnostics)
plot_depth_integrated_current(diagnostics)
plot_streamfunction(diagnostics)
```

`depth_integrated_current` returns the direct analytical expression (66).
Because Gridap enforces its nondivergence only weakly, `compute_diagnostics`
uses the discretely nondivergent current
`(U, V) = (-∂yΨ, ∂xΨ)` for `transport_U` and `transport_V`, so plotted arrows
are tangent to the streamfunction contours, which are overlaid on transport
plots. The direct pointwise values remain
available as `analytical_transport_U` and `analytical_transport_V`; their
relative projection difference is reported in the performance summary.

The depth-integrated momentum terms corresponding to equations (32)--(33),
bottom pressure, and baroclinic potential energy are available through:

```julia
dim = compute_dim_diagnostics(solution, diagnostics)
plot_dim_balance(dim; output="sample_pg_dim_balance.png")
plot_baroclinic_potential_energy(dim; output="sample_pg_chi.png")
plot_bottom_pressure(dim; output="sample_pg_bottom_pressure.png")
```

The DIM balance plot shows the signed Coriolis, bottom-pressure-gradient,
baroclinic-gradient, bottom-stress, and surface-stress terms plus the closure
residual for each horizontal component. Terms within each component share
symmetric color limits.

The API uses conventional oceanographic buoyancy and the fixed upper boundary:
`∂z p = b`, `p = p_s - ∫_z^0 b dz`,
`p_b = p_s - ∫_{-H}^0 b dz`, and
`τ_b = -ν ∂z u|bottom`. Its DIM balance is
`-fV = -H p_b,x - χ_x + Ek(τ_b,x + τ_x)`, with the analogous y equation.
These are the conventions used by the Green-function velocity, pressure,
diagnostic, and plotting paths.

The vector plots normalize the longest arrow to `arrow_scale=0.35` times the
sampled grid spacing. Arrowheads are drawn explicitly in data coordinates
with `arrowhead_size=0.08` times that spacing, so their apparent size remains
consistent in standalone and composite figures. Reduce either value for
shorter shafts or smaller heads; the background color retains the physical
vector magnitude.

`Analytical Model/PG_Stommel_Gyre_Example.ipynb` provides a separate
flat-bottom, beta-plane subtropical gyre example with zero buoyancy forcing.
It compares the computed transport streamfunction with the Sverdrup interior
and diagnoses the frictional western boundary current.
