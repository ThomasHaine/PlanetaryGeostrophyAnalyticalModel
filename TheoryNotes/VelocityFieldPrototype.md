# Velocity-field prototype

`VelocityFieldPrototype.jl` compares the complex velocity \(u + iv\) from
equations (B8) and (B19) of the manuscript.

The implementation uses conventional oceanographic buoyancy,
\(\partial_z p=b\), so the internal pressure is
\(p=p_s-\int_z^0 b\,dz\). The generated symbolic kernels retain the
\(\int_\xi^0\) limits from the derivation; their assembled baroclinic
contribution enters the velocity with an overall minus sign.

```sh
cd TheoryNotes
julia --project=. RunVelocityFieldPrototype.jl
```

It can also be run from a normal Julia REPL without first activating the
project:

```julia
include("RunVelocityFieldPrototype.jl")
```

The runner activates `TheoryNotes/Project.toml` before loading the quadrature
dependencies. The (B19) kernels are checked in as generated plain Julia source,
so the runner does not deserialize version-sensitive `Symbolics` objects from
`symexpr.jld2`.

The script uses the smooth manufactured fields in `run_manufactured_test` and
fails if the relative difference exceeds `2e-5`. It reports both velocity
values, their relative difference, median timings, and the measured speedup.

`velocity_direct` is deliberately the direct nested-quadrature reference:
evaluate it only for validation. `velocity_symbolic` is the production prototype:
it evaluates the generated `Guvb`, `Guvb_xi`, `Guv_y_b`, `Guv_y_b_xi`,
`inner_Gb`, and `inner_Gb_xi` kernels derived by `ComputeQuadratures.ipynb`,
then performs only the remaining single vertical quadratures. The prototype
uses centered horizontal differences for the remaining horizontal derivatives;
replace these with the analytic derivatives of production spline fields to
avoid repeated evaluations. Before a production calculation, regenerate
`VelocitySymbolicKernels.jl` from `ComputeQuadratures.ipynb` whenever its
kernel derivation changes.

The implementation uses the \((\partial_x+i\partial_y)\) convention. Supply
analytic `H_x`, `H_y`, `f_x`, and `f_y` functions for production calculations.
The directional (B19) kernel receives the corresponding derivatives of both
depth and Coriolis parameter, so it supports a sign-definite, nonzero
generalized `f(x, y)` as well as the beta-plane `f(y)` case.
