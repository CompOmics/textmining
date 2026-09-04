# Figure 3: additive staircase

This directory contains the currently supported Figure 3 panels. All
gold-standard performance panels use the SDRF benchmark.

`benchmark_metrics.csv` was re-scored on 2026-09-04 after tier 5 of
`framework/benchmark/semantic_matcher.py` moved from SciBERT to SapBERT. No
inference was re-run; `scripts/rescore_staircase.py` re-scores the preserved
extractions in `benchmark_annotations/`. Full reasoning in `provenance.json`.

## Completed panels

- **a — Overall F1:** weighted F1 by arm and model, mean and sample
  SD across three independent full-inference runs.
- **b — F1 by metadata class:** the same metric for Biological,
  Technical, and Experimental Design agents.
- **c — Output consistency:** non-unknown fields per PXD and exact normalized
  field-value unanimity across all three runs. All-three-unknown units are
  excluded from the agreement denominator. Case and whitespace are normalized,
  and semicolon-delimited value order is ignored.
- **d — Unknown rate:** fraction of the frozen extraction schema that is
  missing, empty, null/none, or the literal value `unknown`. The plotted rate
  is the mean and sample SD across three runs; `unknown_rate_by_field.csv`
  retains the per-field results.
- **f — Crossover:** full-framework Qwen3.8-27B at S5 versus Gemma 4 31B and
  GLM-4.7 at S0 and S1. This tests whether framework structure can outweigh
  model choice; it is not a direct ranking of the models at a common arm.

Every panel is provided as 300-DPI PNG and vector PDF. Source tables and full
provenance are stored alongside the figures.

## Not yet generated

Panel **e**, S5 leave-one-out deltas, requires inference for minus-validation,
minus-gleaning, minus-evidence-pairing, and minus-normalization arms. Those runs
do not currently exist. No placeholder values were fabricated.
