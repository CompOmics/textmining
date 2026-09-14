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

