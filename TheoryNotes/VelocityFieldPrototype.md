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

The runner activates `TheoryNotes/Project.toml` before loading `Symbolics`.
That is required because `symexpr.jld2` contains serialized Symbolics
expressions and must be rebuilt with the package versions recorded in the
repository's `Manifest.toml`.

The script uses the smooth manufactured fields in `run_manufactured_test` and
fails if the relative difference exceeds `2e-5`. It reports both velocity
values, their relative difference, median timings, and the measured speedup.

`velocity_b10` is deliberately the direct nested-quadrature reference:
evaluate it only for validation. `velocity_b23` is the production prototype:
it evaluates the symbolic `Guvb`, `Guvb_xi`, `Guv_y_b`, `Guv_y_b_xi`,
`inner_Gb`, and `inner_Gb_xi` kernels saved by `ComputeQuadratures.ipynb`,
then performs only the remaining single vertical quadratures. The prototype
uses centered horizontal differences for the remaining horizontal derivatives;
replace these with the analytic derivatives of production spline fields to
avoid repeated evaluations. Before a production calculation, regenerate
`symexpr.jld2` by running `ComputeQuadratures.ipynb` whenever its kernel
derivation changes.

The current derivation uses the \((\partial_x+i\partial_y)\) convention and
the \(G_{\mathfrak u,y}\) kernels in (B23); consequently, the prototype's
comparison state has `H = H(y)`. Supply the same geometry assumption, or add
the corresponding \(x\)-derivative kernel before applying (B23) to
two-dimensionally varying bathymetry.
