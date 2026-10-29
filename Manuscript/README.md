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

Two conventions are in use:

- `\Claude{...}` (red) marks earlier text drafted by Claude and pasted in on
  Overleaf.
- Edits made in this folder use the
  [`changes`](https://ctan.org/pkg/changes) package with author `CL` (Claude):
  insertions in blue, deletions struck through, each tagged CL. All changes
  to date have been accepted. For a summary of their counts after the title,
  add `\listofchanges[style=summary]` after `\maketitle`.
  - To review: compile and read the blue text.
  - To accept every change: change `\usepackage{changes}` to
    `\usepackage[final]{changes}`. That keeps the insertions, drops the
    deletions, and removes all marking and the summary.
  - To accept changes selectively, edit `\added{...}`, `\deleted{...}` and
    `\replaced{new}{old}` by hand.

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
