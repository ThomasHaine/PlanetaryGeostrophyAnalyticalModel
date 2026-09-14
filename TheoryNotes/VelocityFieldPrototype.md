# Velocity-field prototype

`VelocityFieldPrototype.jl` compares the complex velocity \(u + iv\) from
equations (B10) and (B23) in the supplied model notes.

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
dependencies. The B23 kernels are checked in as generated plain Julia source,
so the runner does not deserialize version-sensitive `Symbolics` objects from
`symexpr.jld2`.

The script uses the smooth manufactured fields in `run_manufactured_test` and
fails if the relative difference exceeds `2e-5`. It reports both velocity
values, their relative difference, median timings, and the measured speedup.

`velocity_b10` is deliberately the direct nested-quadrature reference:
evaluate it only for validation. `velocity_b23` is the production prototype:
it evaluates the generated `Guvb`, `Guvb_xi`, `Guv_y_b`, `Guv_y_b_xi`,
`inner_Gb`, and `inner_Gb_xi` kernels derived by `ComputeQuadratures.ipynb`,
then performs only the remaining single vertical quadratures. The prototype
uses centered horizontal differences for the remaining horizontal derivatives;
replace these with the analytic derivatives of production spline fields to
avoid repeated evaluations. Before a production calculation, regenerate
`VelocitySymbolicKernels.jl` from `ComputeQuadratures.ipynb` whenever its
kernel derivation changes.

The implementation uses the \((\partial_x+i\partial_y)\) convention. For
general \(H(x,y)\), it obtains the missing \(x\)-kernel from the
\(H_y\)-dependent part of the generated \(G_{\mathfrak u,y}\) kernel by
evaluating it with \(\beta=0\) and \(H_y=1\). Supply analytic `H_x` and `H_y`
functions for production calculations.
