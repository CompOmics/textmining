# SapBERT Staircase analysis

This directory contains a single-run re-scoring of the frozen primary Staircase
annotations. The semantic tier uses
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`; all other benchmark behavior is
unchanged, including the semantic threshold (`0.70`), historical final metric
threshold (`0.50`), field filtering, SDRF gold files, and match hierarchy.

The original annotations under `../Staircase/benchmark_inputs/` are read-only
inputs and are not copied or modified. Results and figures are written here.

Only the primary inference run is included. Consequently, figures contain no
between-run standard deviations or confidence intervals. Repeat consistency is
not reported because it requires multiple independent inference runs.


## Current-matcher re-score and Figure 2 (2026-09-14)

`benchmark_runs/` and `figures/` were scored before
`framework/benchmark/semantic_matcher.py` compared numbers as numbers
(a different count was accepted as a match) and cover one inference run.
`scripts/rescore_replicates.py` re-scores all three replicates in
`benchmark_annotations/` with the current matcher (tier 4 with CL/UBERON,
SapBERT at 0.70, the six unreported fields dropped) into
`benchmark_runs_current_matcher/` and `manuscript_figure2/benchmark_metrics.csv`
(mean and SD over runs). `scripts/build_figure2.py` adds Lin IC on the eight
ontology-backed fields, output volume and three-run unanimity, and draws
`manuscript_figure2/figure2_performance.png` (a-d) and
`figure2_consistency.png` (e-f). These are the tables the manuscript uses.
