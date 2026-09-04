# Final analyses

One directory per manuscript figure, each self-contained: scripts, the compact
result tables they produce, and the figure built from those tables. Source data
stays where it belongs and is read from there rather than copied in.

| Directory | Manuscript figure | Source data |
|---|---|---|
| `HumanAnnotation/` | Figure 1, inter-annotator agreement | `EuBIC-annotation/` |
| `Staircase/` | Figure 2, cumulative ablation | `Staircase/benchmark_runs/`, on the cluster |
| `ScientificGeneralizationV2/` | not yet placed | |

Figures 3 and 4 are not here yet. Figure 3, model comparison and entity-level
matching, is waiting on the six-model benchmark re-run. Figure 4, the repository
record against the article, currently lives under `MLMarker/analysis/` and
should move here once it settles.

## Conventions

**Scripts read source data, never copies.** `HumanAnnotation/scripts/common.py`
reads the brat files in `EuBIC-annotation/` directly. If an analysis needs data
that lives somewhere else in the repository, it reads it from there.

**`results/` holds compact tables, not runtime output.** Enough to rebuild the
figure and to check a number, small enough to keep in git. Raw per-run output
stays on the cluster and is gitignored, which is why `Staircase/benchmark_runs/`
and `benchmark_inputs/` are absent here.

**`figures/manuscript_figureN/` is the deliverable.** One figure per directory,
numbered by the current manuscript. Diagnostic per-panel images are gitignored.

## A naming wrinkle worth knowing

`Staircase/figures/figure3/` holds the aggregated benchmark tables. The `3` is
the **old** manuscript numbering; in the current numbering that analysis is
**Figure 2**. The directory keeps its name so it does not conflict with further
pushes, and the manuscript figure is written to
`Staircase/figures/manuscript_figure2/` instead. `build_manuscript_figure2.py`
reads from the former and writes to the latter.

## Figures come in two forms

Each `build_*` script writes the composite that goes in the manuscript **and**
every panel separately under `figures/manuscript_figureN/panels/`. Both are
kept. The composite is the deliverable; the separate panels are what actually
gets used in a talk or sent to a co-author.
