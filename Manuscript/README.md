# Manuscript source

LaTeX source of *On Robust Diagnostic Planetary Geostrophy* (AMS template),
kept in step with the Overleaf project.

- The first commit adding this folder is the unmodified Overleaf export of
  2026-10-14. Later commits are edits made here, so `git log -p Manuscript/`
  shows every change exactly.
- Build locally with

  ```bash
  pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
  ```

  Build products are git-ignored.

## How changes are marked

All tracked changes have been accepted, and `manuscript.tex` carries no
change markup. To propose new edits for review, load the
[`changes`](https://ctan.org/pkg/changes) package in the preamble:

```latex
\usepackage{changes}
\definechangesauthor[name={Claude review}, color=magenta]{CR}
```

and mark edits with `\added[id=CR]{...}`, `\deleted[id=CR]{...}` and
`\replaced[id=CR]{new}{old}`. To accept them, keep the new text, drop the old,
and remove the package lines again (or load it as
`\usepackage[final]{changes}`, which hides all marking).

## Figures made in this repository

| File | Made by |
|---|---|
| `coefficient_regimes.pdf` | `TheoryNotes/PlotCoefficientRegimes.jl` |
| `verification_convergence.pdf` | `TheoryNotes/VerificationConvergence.jl` |
| `oceananigans_comparison.pdf` | `AnalyticalModel/OceananigansComparison.jl` |
| `stommel_comparison.pdf` | `TheoryNotes/StommelComparison.jl` |
| `beta_plume_comparison.pdf` | `TheoryNotes/BetaPlumeComparison.jl` |
| `boundary_beta_plume_comparison.pdf` | `TheoryNotes/BoundaryBetaPlumeComparison.jl` |
| `topographic_steering_comparison.pdf` | `TheoryNotes/TopographicSteeringComparison.jl` |
| `double_basin_showcase.pdf` | `TheoryNotes/DoubleBasinShowcase.jl` |

Copy regenerated figures here, and upload them to Overleaf.
