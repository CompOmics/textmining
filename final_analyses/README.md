# Final analyses

One directory per manuscript figure, each self-contained: scripts, the compact
result tables they produce, and the figure built from those tables. Source data
stays where it belongs and is read from there rather than copied in.

| Directory | Manuscript figure | Source data |
|---|---|---|
| `HumanAnnotation/` | Figure 1, inter-annotator agreement | `EuBIC-annotation/` |
| `SapBERTStaircase/` | Figure 2, cumulative ablation (SapBERT-scored, single run; `Staircase/` deprecated) | `SapBERTStaircase/benchmark_runs/` |
| `MatcherCalibration/` | Supplementary, matcher permutation null and Lin IC | `SapBERTStaircase/benchmark_runs/` |
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
stays on the cluster and is gitignored.

**`figures/manuscript_figureN/` is the deliverable.** One figure per directory,
numbered by the current manuscript. Diagnostic per-panel images are gitignored.

## Naming

`SapBERTStaircase/figures/figure3_panel_*.png` carry the **old** manuscript
numbering; in the current numbering that analysis is **Figure 2**. The files
keep their names so they do not conflict with further pushes.
