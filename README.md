# Planetary Geostrophy Analytical Model

* LaTeX notes on planetary geostrophy with analytic solutions for certain cases.
* Julia code to support the theory derivation using `Symbolics.jl`
* Julia code to compute solutions using analytic results and `Gridap.jl` solutions of the surface pressure and depth-integrated streamfunction PDEs

See also: [this archived repo](https://github.com/ThomasHaine/ArcticOceanDynamicsWorkshop25), on which this one is based.

Questions/comments welcome.

## Contents

- [Non-dimensional PG API](#non-dimensional-pg-api)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Problem specification and field output](#problem-specification-and-field-output)
  - [Three-dimensional Lagrangian trajectories](#three-dimensional-lagrangian-trajectories)
  - [Performance metrics](#performance-metrics)
  - [Solution method and threading](#solution-method-and-threading)
- [Transport diagnostics](#transport-diagnostics)
- [Forcing sensitivity](#forcing-sensitivity)
- [Dynamical-balance diagnostics](#dynamical-balance-diagnostics)
  - [Depth-integrated momentum](#depth-integrated-momentum)
  - [Barotropic vorticity](#barotropic-vorticity)
  - [Depth-integrated vorticity](#depth-integrated-vorticity)
  - [Geostrophic-contour vorticity](#geostrophic-contour-vorticity)
  - [Transport-divergence vorticity](#transport-divergence-vorticity)
  - [Surface-pressure equation](#surface-pressure-equation)
  - [Mapped-derivative conventions](#mapped-derivative-conventions)
- [Model and plotting conventions](#model-and-plotting-conventions)
- [Example notebooks](#example-notebooks)
  - [General variable-topography example](#general-variable-topography-example)
  - [Stommel gyre](#stommel-gyre)
  - [Isolated beta plume](#isolated-beta-plume)
  - [Schott--Stommel beta spiral](#schott--stommel-beta-spiral)
  - [Validation and derivation notebooks](#validation-and-derivation-notebooks)

## Non-dimensional PG API

The reusable Julia API is a standalone package in the root-level
`PlanetaryGeostrophy/` directory. Its project contains only the
analytical solver dependencies; Oceananigans remains confined to the separate
validation environment.
It packages the B23 Green-function velocity and buoyancy kernels used by
the original derivation notebook with the Gridap surface-pressure solve used by
`Oceananigans_Variable_Basin_Validation.ipynb`.

### Installation

From the repository root, activate the standalone API environment:

```julia
using Pkg
Pkg.activate("PlanetaryGeostrophy")
Pkg.instantiate()
using PlanetaryGeostrophy
```

### Quick start

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
plot_solution(fields; sigma_level=0.5, output="sample_pg.png")
plot_solution(fields; z_index=length(fields.z), output="sample_pg_surface.png")
```

### Problem specification and field output

`wind_stress` uses the complex convention `τx + im * τy`. Custom analytic
derivatives can be passed as `depth_x`, `depth_y`, `coriolis_x`, and
`coriolis_y`; centered derivatives are generated otherwise. The sampled
solution contains `u`, `v`, `w`, and `b` on the 3D grid plus surface pressure
and depth. Vertical velocity is reconstructed from discrete horizontal
divergence with no-normal-flow conditions at the bottom and rigid lid.
For variable topography, `sigma_level=0.5` plots a terrain-following slice at
`z/H=-0.5`; a fixed `z_index` intentionally leaves cells below the local
bottom blank.

### Three-dimensional Lagrangian trajectories

The steady three-dimensional velocity can be sampled in the terrain-following
coordinate `sigma=-z/H(x,y)` and used to integrate particle paths:

```julia
lagrangian_field = sample_lagrangian_velocity(
    solution,
    range(-0.9, 0.9; length=33),
    range(-0.65, 0.65; length=33),
    range(0.0, 1.0; length=33),
)
initial_positions = [
    (-0.4, -0.2, -0.25),
    (-0.4, -0.2, -0.50),
    ( 0.3,  0.2, -0.75),
]
trajectories = compute_trajectories(
    lagrangian_field,
    initial_positions,
    (0.0, 4000.0);
    saveat=range(0.0, 4000.0; length=401),
)
plot_trajectories(trajectories; output="sample_pg_trajectories.png")
animate_trajectories(
    trajectories;
    output="sample_pg_trajectories.gif",
    frame_stride=4,
    fps=15,
)
```

The transformed vertical velocity is reconstructed from
`∂x(Hu)+∂y(Hv)+∂sigma(H*dsigma/dt)=0` and corrected to vanish at both the
rigid lid and seabed. Multilinear interpolation and adaptive `Tsit5`
integration then evolve `(x,y,sigma)`. Seeds must lie strictly inside the
sampled all-water patch; trajectories terminate cleanly if they reach a
horizontal edge, the surface, or the seabed. Coordinates, velocity, and time
remain non-dimensional. The animation shows the accumulated 3D and horizontal
paths together with each particle's current position.

### Performance metrics

The solver logs each mesh, finite-element, field-reconstruction, output, and
plotting stage with elapsed time, allocation, and garbage-collection metrics.
The solve and sampling metrics are also available programmatically:

```julia
performance_summary(solution)
performance_summary(fields)
files.timings
```

### Solution method and threading

By default (`method=:fast`), `solve_pg` evaluates every vertical integral in
single-integral form:

- the transport forcing 𝔅 from (69),
- the bottom-stress forcing T from (65),
- the velocity from (B5) with the closed-form G_𝔲 (60),
- the buoyancy from (51).

Each of these needs only the baroclinic pressure
`Π(ξ) = ∫_ξ^0 b dz`. Π comes from the closed-form kernels (A6)–(A7) with one
quadrature. Vertical integrals use a fixed composite Gauss–Legendre rule
(`ColumnQuadrature`). Its panels are scaled to the smaller of the Ekman depth
and the buoyancy-diffusion depth, and it is split at kernel kinks.

𝔅 is evaluated in parallel before assembly. With
`forcing_assembly=:quadrature` (the default) it is computed at every
finite-element quadrature point, which gives the same load vector as direct
evaluation. With `:nodal` it is computed once per node and interpolated; that
uses about a third of the evaluations but adds an O(h³) error.

Finite-element point queries keep one cell-search cache per thread. Sampling,
transport diagnostics and DIM diagnostics are threaded, so start Julia with
`julia -t auto`, or use a threaded IJulia kernel:

```julia
using IJulia
IJulia.installkernel("Julia (auto threads)", env=Dict("JULIA_NUM_THREADS" => "auto"))
```

`method=:b23` restores the original B23 symbolic-kernel quadratures. The test
suite checks that the two methods agree to about 1e-9.

Timings for the `PG_Double_Basin_Prototype.ipynb` high-resolution setting
(mesh 0.02, 128×129×98 samples, 65² diagnostic grid) on an 18-thread Apple
M5 Max:

| Stage | `:b23` | `:fast` |
|---|---|---|
| `solve_pg` | 1037 s | 3.5 s |
| `sample_solution` | 207 s | 1.05 s |
| `compute_diagnostics` | 780 s | 2.8 s |
| DIM and vorticity budgets | ~70 s | 2.1 s |

## Transport diagnostics

Bottom stress, depth-integrated current, and its streamfunction are available
as point queries or sampled maps. These use equations (63)--(70) and the
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

`depth_integrated_current` returns the direct analytical expression (67).
Because Gridap enforces its nondivergence only weakly, `compute_diagnostics`
uses the discretely nondivergent current
`(U, V) = (-∂yΨ, ∂xΨ)` for `transport_U` and `transport_V`, so plotted arrows
are tangent to the streamfunction contours, which are overlaid on transport
plots. The direct pointwise values remain
available as `analytical_transport_U` and `analytical_transport_V`; their
relative projection difference is reported in the performance summary.

## Forcing sensitivity

The steady PG problem is linear in the restoring buoyancy $B$, surface
buoyancy $b_s$, and surface-stress components $(\tau^x,\tau^y)$. The adjoint
diagnostic computes the spatial functional derivatives of one scalar
cost built from the depth-integrated current:

```math
J=\iint \rho(x,y)\,c(U,V)\,dx\,dy,
```

and

```math
\delta J
=\iint K_{\tau^x}\delta\tau^x\,dx\,dy
+\iint K_{\tau^y}\delta\tau^y\,dx\,dy
+\iint K_{b_s}\delta b_s\,dx\,dy
+\iiint K_B\delta B\,dx\,dy\,dz.
```

It solves the transpose of the surface-pressure system by conjugating its
complex transport coefficient, which reverses the pressure operator's
advecting velocity. One adjoint solve therefore provides the sensitivity to
forcing at every sampled location. The nominal point observation is a
normalized Gaussian average with radius `target_radius`; this avoids a
mesh-dependent point delta and converges toward a point query as the pressure
mesh and radius are refined together.

The user supplies `cost(U, V)`, which must return one real scalar. The API
uses automatic differentiation to evaluate `∂cost/∂U` and `∂cost/∂V` at the
solved transport. Linear costs such as `(U, V) -> V` recover component
sensitivities; nonlinear costs such as transport speed produce the
first-order sensitivity about the base solution.

```julia
adjoint_sensitivity = compute_transport_adjoint_sensitivity(
    solution,
    range(-0.9, 0.9; length=25),
    range(-0.65, 0.65; length=25),
    [-0.75, -0.5, -0.25];
    target=(0.4, -0.15),
    cost=(U, V) -> hypot(U, V),
    cost_label="transport speed",
    target_radius=0.06,
)

plot_transport_adjoint_sensitivity(
    adjoint_sensitivity;
    output="transport_adjoint_sensitivity.png",
)
```

The result contains 2D `wind_stress_x`, `wind_stress_y`, and
`surface_buoyancy` kernels, plus the 3D `restoring_buoyancy` kernel sampled at
the requested depths. Variable topography introduces an additional
distributional contribution where the restoring field meets the seabed; this
is retained separately as `bottom_restoring_buoyancy`.
Every plot panel marks the target center with a cross and the Gaussian target
radius with a dashed circle; the same coordinates and radius appear in the
figure title.

The earlier `solve_forcing_sensitivities` API remains available for
directional derivatives with respect to the amplitude of a complete,
user-supplied forcing pattern. Those scalar-amplitude responses are distinct
from the spatial functional derivatives returned by the adjoint API.

## Dynamical-balance diagnostics

### Diagnostic API

The depth-integrated momentum terms corresponding to equations (32)--(33),
bottom pressure, and baroclinic potential energy are available through:

```julia
dim = compute_dim_diagnostics(solution, diagnostics)
plot_dim_balance(dim; output="sample_pg_dim_balance.png")
plot_baroclinic_potential_energy(dim; output="sample_pg_chi.png")
plot_bottom_pressure(dim; output="sample_pg_bottom_pressure.png")

barotropic_vorticity =
    compute_barotropic_vorticity_diagnostics(solution, diagnostics)
plot_barotropic_vorticity_balance(
    barotropic_vorticity;
    output="sample_pg_barotropic_vorticity_balance.png",
)

vorticity = compute_vorticity_diagnostics(solution, diagnostics)
plot_vorticity_balance(
    vorticity;
    output="sample_pg_vorticity_balance.png",
)

gcvb = compute_geostrophic_contour_vorticity_diagnostics(
    solution, diagnostics,
)
plot_geostrophic_contour_vorticity_balance(
    gcvb;
    output="sample_pg_geostrophic_contour_vorticity_balance.png",
)

tdvb = compute_transport_divergence_vorticity_diagnostics(
    solution, diagnostics,
)
plot_transport_divergence_vorticity_balance(
    tdvb;
    output="sample_pg_transport_divergence_vorticity_balance.png",
)

pressure_equation = compute_surface_pressure_equation_diagnostics(
    solution, diagnostics,
)
plot_surface_pressure_equation_diagnostics(
    pressure_equation;
    output="sample_pg_surface_pressure_equation.png",
)
```

### Depth-integrated momentum

The DIM balance plot shows the signed Coriolis, bottom-pressure-gradient,
baroclinic-gradient, bottom-stress, and surface-stress terms plus the closure
residual for each horizontal component. Terms within each component share
symmetric color limits.

### Barotropic vorticity

The barotropic vorticity diagnostic evaluates equation (36):

```math
f(\partial_x U+\partial_y V)+U f_x+V f_y
=J(p_b,H)
+\mathrm{Ek}\,[\nabla\times(\boldsymbol{\tau}_b)]_z
+\mathrm{Ek}\,[\nabla\times\boldsymbol{\tau}]_z,
```

where $J(p_b,H)=p_{b,x}H_y-p_{b,y}H_x$. It retains the transport-divergence
term explicitly and evaluates it as exactly zero from the steady fixed-lid,
no-normal-flow mass balance, rather than differentiating sampled transport.
The residual is then the cell-averaged $-f\nabla\cdot\mathbf U$, which is the
finite-element mass-conservation error (see
[Mapped-derivative conventions](#mapped-derivative-conventions)).

### Depth-integrated vorticity

The depth-integrated vorticity diagnostic evaluates equation (41), generalized
from the manuscript's beta plane to flexible $f(x,y)$:

```math
U f_x + V f_y + f w_b
=
\mathrm{Ek}\,[\nabla\times(\boldsymbol{\tau}_b)]_z
+ \mathrm{Ek}\,[\nabla\times\boldsymbol{\tau}]_z,
```

where $w_b=-(u_bH_x+v_bH_y)$ follows from the bottom kinematic boundary
condition. On a beta plane, $U f_x+V f_y=\beta V$. The plot shows these
signed terms and the closure residual with one shared symmetric color scale.
For sloping bathymetry, equation (41) requires the bottom vertical-vorticity
flux with horizontal derivatives taken at fixed $z$. The diagnostic therefore
corrects the curl of stress sampled along $z=-H(x,y)$ by
$f w_b+J(p_b,H)$. Omitting this chain-rule correction gives a spuriously large
DIV residual over topography, although the two curls coincide for flat depth.

### Geostrophic-contour vorticity

The geostrophic-contour vorticity diagnostic evaluates equation (43):

```math
J\left(\frac{f}{H},\Psi\right)
=J\left(\frac{1}{H},\chi\right)
-\mathrm{Ek}\left[\nabla\times
\left(\frac{\boldsymbol{\tau}+\boldsymbol{\tau}_b}{H}\right)\right]_z,
```

where $(U,V)=(-\Psi_y,\Psi_x)$ and
$J(a,b)=a_xb_y-a_yb_x$. The JEBAR term
$J(1/H,\chi)$ and the two stress contributions diagnose transport across
$f/H$ contours without bottom pressure.

### Transport-divergence vorticity

The transport-divergence vorticity diagnostic evaluates equation (47):

```math
0
=J\left(p_b,\frac{H}{f}\right)
+J\left(\chi,\frac{1}{f}\right)
+\mathrm{Ek}\left[\nabla\times
\left(\frac{\boldsymbol{\tau}+\boldsymbol{\tau}_b}{f}\right)\right]_z.
```

This balance contains no circulation variable because the depth-integrated
transport divergence vanishes in the steady fixed-lid problem. It instead
diagnoses cancellation among bottom pressure, baroclinic potential energy,
and the surface and bottom stresses.

### Surface-pressure equation

The surface-pressure coefficient diagnostic evaluates the fields in equation
(75):

```math
\nabla\mathbin{\cdot}(\mathfrak A_r\nabla p_s)
-J(\mathfrak A_i,p_s)
=-\partial_x\mathfrak B_r-\partial_y\mathfrak B_i.
```

It reports the advection velocity
$(\partial_y\mathfrak A_i,-\partial_x\mathfrak A_i)$ as inviscid,
frictional, and total fields on a common plot scale. The coefficient split is
$\mathfrak A_i^{(0)}=H/f$ and

```math
\mathfrak A_i^{(\mathrm{fr})}
=-\frac{\sinh(2s)+\sin(2s)}
{2\alpha f[\cosh(2s)+\cos(2s)]},
\qquad
\alpha=\sqrt{\frac{|f|}{2\nu\mathrm{Ek}}},\quad s=\alpha H.
```

The diagnostic also reports diffusivity $\mathfrak A_r$ and source
$-\partial_x\mathfrak B_r-\partial_y\mathfrak B_i$. Because the advecting
velocity depends only on $\mathfrak A(H,f,\mathrm{Ek},\nu)$, its split is
distinct from the buoyancy- and stress-driven source split. The decomposition
$\mathfrak B=\mathfrak B_b+\mathfrak B_\tau$ gives separate baroclinic and
surface-stress contributions, which share a color scale in the plot.

The complex coefficients $\mathfrak A$ and $\mathfrak B$ are exactly those
used to assemble the Gridap surface-pressure problem. The mapped diagnostic
recovers $\mathfrak B$ from the previously sampled analytical transport rather
than repeating its costly column quadrature. Its derivatives use centered
diagnostic-grid differences, so unsupported edge and near-land cells are left
blank. The reported residual is the sampled transport divergence, equivalent
to the left-hand side minus the source in equation (75). Pointwise values are
available from
`surface_pressure_equation_coefficients(solution, x, y)`.

### Mapped-derivative conventions

By default (`method=:cell_average`), the four mapped vorticity diagnostics
report each term as its average over the diagnostic cell around each grid
point. The cell edges are the midpoints between neighbouring grid points.
Curl and Jacobian terms are converted to edge integrals with Stokes' theorem,

```math
\iint \hat{\mathbf z}\cdot\nabla\times\mathbf F\,dA=\oint \mathbf F\cdot\hat{\mathbf t}\,ds,
\qquad
\iint J(a,b)\,dA=\oint a\,\nabla b\cdot\hat{\mathbf t}\,ds .
```

The resulting edge integrals are:

- bottom-pressure torque $\oint p_b\,dH$ (BV, DIV) and $\oint p_b\,d(H/f)$ (TDVB),
- JEBAR $\oint \chi\,\nabla H\cdot\hat{\mathbf t}/H^2\,ds$,
- baroclinic torque $\oint \chi\,d(1/f)$,
- stress curls $\mathrm{Ek}\oint \boldsymbol\tau\cdot\hat{\mathbf t}\,ds$, divided by $H$ or $f$ where the balance requires.

The advection terms $U f_x+V f_y$ and $J(f/H,\Psi)=-\mathbf U\cdot\nabla(f/H)$
use area quadrature. The bottom stretching $f w_b$ uses its cell-centre value;
it vanishes under the no-slip condition (18). No finite-element
field is differentiated. This matters because the surface-pressure gradient,
which enters the bottom stress and transport, is discontinuous across element
edges. Centered differences of those fields do not converge as the grid is
refined, and at small Ekman number they dominate the residual. In the
double-basin example at $\mathrm{Ek}=0.002$ the centered residuals are 0.4--1.

Because the depth-integrated momentum balance holds pointwise and the
transport divergence is set to zero, every cell-averaged residual equals the
cell-averaged finite-element mass-conservation error, up to quadrature error:

- $-f\nabla\cdot\mathbf U$ for BV and DIV,
- $(f/H)\nabla\cdot\mathbf U$ for GCVB,
- $-\nabla\cdot\mathbf U$ for TDVB.

Each interior edge is integrated once and shared by its two cells, and the
work is threaded. A cell is used when its centered stencil is wet and every
edge and area quadrature point lies in water. `CellBudgetQuadrature` sets the
edge panels and points and the area rule. `method=:centered` restores the
earlier maps. Those use centered diagnostic-grid differences of the sampled
bottom pressure, $\chi$, and the bottom stress interpolated into a continuous
finite-element space. The prescribed surface-stress curl there comes from
`wind_stress_curl(x, y)` or automatic differentiation.

## Model and plotting conventions

### Oceanographic sign conventions

The API uses conventional oceanographic buoyancy and the fixed upper boundary:
`∂z p = b`, `p = p_s - ∫_z^0 b dz`,
`p_b = p_s - ∫_{-H}^0 b dz`, and
`τ_b = -ν ∂z u|bottom`. Its DIM balance is
`-fV = -H p_b,x - χ_x + Ek(τ_b,x + τ_x)`, with the analogous y equation.
These are the conventions used by the Green-function velocity, pressure,
diagnostic, and plotting paths.

### Vector-plot scaling

The vector plots normalize the longest arrow to `arrow_scale=0.35` times the
sampled grid spacing. Arrowheads are drawn explicitly in data coordinates
with `arrowhead_size=0.08` times that spacing, so their apparent size remains
consistent in standalone and composite figures. Reduce either value for
shorter shafts or smaller heads; the background color retains the physical
vector magnitude.

## Example notebooks

### General variable-topography example

`AnalyticalModel/PG_API_Example.ipynb` is the main API walkthrough. It uses a
smooth zonal slope and Gaussian seamount to demonstrate flexible
two-dimensional bathymetry, solves and plots the full three-dimensional
fields at the surface, fixed `z=-0.5`, and terrain-following `z/H=-0.5`
levels. It also computes and animates 3D Lagrangian trajectories and runs every
mapped dynamical-balance and surface-pressure diagnostic. Its additional
forward/adjoint consistency solves are disabled by default; set
`RUN_ADJOINT_CHECKS = true` in the setup cell to run them.

### Stommel gyre

`AnalyticalModel/PG_Stommel_Gyre_Example.ipynb` provides a separate
flat-bottom, beta-plane subtropical gyre example with zero buoyancy forcing.
It compares the computed transport streamfunction with the Sverdrup interior
and diagnoses the frictional western boundary current. It uses
$\mathrm{Ek}=0.0025$, $\beta=0.60$, a pressure-mesh resolution of $0.03$,
and a $101\times51$ diagnostic grid to resolve a clearer western
intensification.
Its Sverdrup comparison uses the nondimensional balance
$\beta V=\mathrm{Ek}\,[\nabla\times\boldsymbol{\tau}]_z$ and plots the
interior balance separately from its bottom-stress boundary closure.

### Isolated beta plume

`AnalyticalModel/PG_Beta_Plume_Example.ipynb` provides a flat-bottom,
unstratified beta-plume example inspired by Stommel (1982). A smooth
azimuthal surface stress produces an exactly Gaussian, spatially isolated
wind-stress curl east of the basin center. The notebook compares the
westward PG response with the inviscid Sverdrup plume and diagnoses its
bottom-stress closure, following the vorticity-budget perspective of Haine
and Fuller (2016). See
[Stommel (1982)](https://doi.org/10.1016/0012-821X(82)90038-3) and
[Haine and Fuller (2016)](https://doi.org/10.1002/qj.2866).

### Schott--Stommel beta spiral

`AnalyticalModel/PG_Beta_Spiral_Example.ipynb` reproduces the local,
inviscid beta-spiral mechanism of Schott and Stommel (1978). It integrates a
synthetic subtropical PG water column, plots the absolute-velocity hodograph,
verifies the isopycnal-slope constraint, recovers the unknown reference
velocity by overdetermined least squares, and demonstrates the paper's
wrong-latitude residual test. It also compares the inviscid spiral with an
independently solved frictional PG basin at small Ekman number. See
[Schott and Stommel (1978)](https://doi.org/10.1016/0011-7471(78)90001-7).

### Validation and derivation notebooks

- `AnalyticalModel/Oceananigans_PG_Validation.ipynb` validates the analytical
  solution against Oceananigans.
- `AnalyticalModel/Oceananigans_Variable_Basin_Validation.ipynb` covers
  variable-basin validation.
