# PRIDE contribution by methods availability

Matched production prompts, paired datasets; all PRIDE sections removed from
the manuscript-only control. Methods availability refers to extracted input,
not necessarily the full published article.

| Measure | Without methods (1,136 datasets) | With methods (601 datasets) |
| --- | ---: | ---: |
| Populated fields, manuscript input | 8,469 | 13,528 |
| Populated fields, manuscript + PRIDE | 22,745 | 14,281 |
| Fields per dataset, manuscript input | 7.46 | 22.51 |
| Fields per dataset, manuscript + PRIDE | 20.02 | 23.76 |
| Newly populated fields | 15,033 | 1,824 |
| Fields becoming unknown | 757 | 1,071 |
| Net fields added | 14,276 | 753 |
| Net fields added per dataset | 12.57 | 1.25 |
| Relative increase in populated fields | 168.6% | 5.6% |

The mean net increase is approximately ten times larger without methods. Of
the overall net increase of 15,029 fields, 95.0% occurs in datasets without methods.
Technical fields account for 10,333 of the 14,276 net additions without methods;
biological fields add 1,706 and experimental-design fields add 2,237. With methods,
technical fields add 863, biological fields lose 188, and experimental design adds 78.

## Agreement among shared populated fields

| Value-set agreement | Without methods (7,712 shared fields) | With methods (12,457 shared fields) |
| --- | ---: | ---: |
| Equivalent | 51.5% | 57.4% |
| Hierarchy-related | 1.0% | 0.5% |
| Semantic candidate | 11.2% | 7.1% |
| Partial overlap | 15.9% | 21.0% |
| No accepted overlap | 20.4% | 14.1% |

These percentages are pooled over shared dataset–field pairs, not averages of
per-dataset agreement. Rounding may prevent totals from summing to exactly 100%.
Semantic candidates are not confirmed equivalence; no overlap does not establish
contradiction. Additional fields appear only in one run and are excluded here.

## Interpretation

In this paired, matched-prompt comparison, adding PRIDE context substantially
increased metadata coverage for datasets lacking extracted methods (+12.57 fields
per dataset), with a smaller net increase where methods were available (+1.25).
This supports the conclusion that PRIDE primarily fills missing methodological
context, especially technical details. Increased coverage is not itself evidence
of improved extraction accuracy. Group differences are descriptive, not randomised
effects of methods availability; the cohorts can differ in other ways. Separate
LLM generations retain stochastic variation, and schema aliases are counted.

## Figure legend

**A:** mean populated fields per dataset for manuscript-only versus manuscript
plus PRIDE inputs, stratified by methods availability. Without-methods manuscript
inputs contain abstracts only. **B:** paired net gain in populated fields per
dataset, accounting for both additions and losses. Means describe all datasets
in each group; no significance tests or uncertainty intervals are implied.
Prompts are matched; PRIDE inputs include properties and available protocols.

Outputs: `job_966224/by_methods/summary.json` and
`job_966224/by_methods/pride_gain_by_methods.{png,pdf,svg}`.
