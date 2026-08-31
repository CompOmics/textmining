# Staircase benchmark results

These are the presentation-ready SDRF benchmark results used for the reported
headline metrics.

- Evaluation set: 30 held-out PXDs.
- Matcher: `agentic-metadata/benchmark/semantic_matcher.py`, unchanged.
- Semantic classification threshold: 0.70.
- Final metric threshold: 0.50.
- Summary statistic: macro-average across Biological, Technical, and
  Experimental Design agents.
- Variability: mean and sample standard deviation across three independent
  full-inference runs.

| Model | Arm | Weighted F1, mean ± SD |
|---|---:|---:|
| Qwen3.8-27B | S0 | 0.543 ± 0.034 |
| Qwen3.8-27B | S1 | 0.856 ± 0.007 |
| Qwen3.8-27B | S2 | 0.972 ± 0.005 |
| Qwen3.8-27B | S3 | 0.969 ± 0.004 |
| Qwen3.8-27B | S4 | 0.978 ± 0.004 |
| Qwen3.8-27B | S5 | 0.961 ± 0.006 |
| Gemma 4 31B | S0 | 0.700 ± 0.005 |
| Gemma 4 31B | S1 | 0.898 ± 0.001 |
| Gemma 4 31B | S2 | 0.966 ± 0.005 |
| Gemma 4 31B | S3 | 0.968 ± 0.001 |
| Gemma 4 31B | S4 | 0.968 ± 0.001 |
| Gemma 4 31B | S5 | 0.949 ± 0.002 |
| GLM-4.7 | S0 | 0.288 ± 0.060 |
| GLM-4.7 | S1 | 0.806 ± 0.007 |
| GLM-4.7 | S2 | 0.960 ± 0.002 |
| GLM-4.7 | S3 | 0.962 ± 0.001 |
| GLM-4.7 | S4 | 0.962 ± 0.001 |
| GLM-4.7 | S5 | 0.943 ± 0.007 |

`staircase_f1.png` and `staircase_f1.pdf` plot these means with
sample-standard-deviation error bars. `summary.csv` and `summary.json` contain
the replicate values, means, SDs, and Student-t 95% confidence intervals.
