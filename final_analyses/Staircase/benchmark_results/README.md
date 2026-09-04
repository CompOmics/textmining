# Staircase benchmark results

These are the presentation-ready SDRF benchmark results used for the reported
headline metrics.

- Evaluation set: 30 held-out PXDs.
- Matcher: `framework/benchmark/semantic_matcher.py`, tier 5 embedding SapBERT
  (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`). Superseded SciBERT on
  2026-09-04; see `figures/figure3/provenance.json`.
- Semantic classification threshold: 0.70.
- Final metric threshold: 0.50.
- Summary statistic: macro-average across Biological, Technical, and
  Experimental Design agents.
- Variability: mean and sample standard deviation across three independent
  full-inference runs.

| Model | Arm | Weighted F1, mean ± SD |
|---|---:|---:|
| GLM-4.7 | S0 | 0.308 ± 0.043 |
| GLM-4.7 | S1 | 0.749 ± 0.012 |
| GLM-4.7 | S2 | 0.860 ± 0.002 |
| GLM-4.7 | S3 | 0.861 ± 0.002 |
| GLM-4.7 | S4 | 0.864 ± 0.002 |
| GLM-4.7 | S5 | 0.847 ± 0.007 |
| Gemma 4 31B | S0 | 0.647 ± 0.011 |
| Gemma 4 31B | S1 | 0.761 ± 0.005 |
| Gemma 4 31B | S2 | 0.881 ± 0.002 |
| Gemma 4 31B | S3 | 0.877 ± 0.005 |
| Gemma 4 31B | S4 | 0.878 ± 0.004 |
| Gemma 4 31B | S5 | 0.864 ± 0.008 |
| Qwen3.8-27B | S0 | 0.492 ± 0.026 |
| Qwen3.8-27B | S1 | 0.769 ± 0.017 |
| Qwen3.8-27B | S2 | 0.879 ± 0.004 |
| Qwen3.8-27B | S3 | 0.863 ± 0.013 |
| Qwen3.8-27B | S4 | 0.876 ± 0.006 |
| Qwen3.8-27B | S5 | 0.855 ± 0.024 |

`staircase_f1.png` and `staircase_f1.pdf` plot these means with
sample-standard-deviation error bars. `summary.csv` and `summary.json` contain
the replicate values, means, SDs, and Student-t 95% confidence intervals.
