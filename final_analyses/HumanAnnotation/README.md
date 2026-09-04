# Human annotation analysis

Inter-annotator agreement over the EuBIC-MS annotation effort, and the source of
manuscript **Figure 1**.

Reduced on 2026-09-04 to the five panels the manuscript actually uses. The
per-field agreement bar chart, the per-label movement plot, the bootstrap-CI
heatmap, the outlier screen, the threshold-sensitivity sweep, the
harmonised-consensus verification, the Gwet AC1 alternative and the
manual-review audit sample were all removed, along with the tables and scripts
that fed only them.

**`git log` does not have them.** Those files were staged but never committed,
so the removal leaves no history to recover from. A full copy of the directory
as it stood before the reduction is at `~/HumanAnnotation_removed_20260904/`.
Delete that once you are satisfied nothing is needed from it.

## Where the data comes from

Every number derives from `EuBIC-annotation/` at the repository root and from
nothing else. `scripts/common.py` reads the `.ann` files directly.

| Source | Contents |
|---|---|
| `EuBIC-annotation/SingleHuman/` | one expert, all 27 manuscripts |
| `EuBIC-annotation/MultiHuman/` | eight annotators in overlapping batches |
| `EuBIC-annotation/HarmonizedHuman/` | reconciled consensus |
| `EuBIC-annotation/s0_model_iaa_27/final_annotations/` | Qwen3.8-27B, Gemma 4 31B, GLM-4.7 |
| `EuBIC-annotation/pxd_pmid_mapping.csv` | PXD to PMID, needed because the model sets are keyed by accession |

Note what the annotators saw: PMC abstract and methods prose only. The PRIDE
metadata in that corpus is attached as structured CvParam entities and was never
placed in the annotatable text, so it carries no character offsets and no
annotations. The extraction pipeline, by contrast, is given PRIDE project
properties as text. That asymmetry does not affect anything in this directory,
since every comparison here is between annotators of the same corpus, but it
matters to any model-versus-human claim built on top of it.

## Layout

```
scripts/     common.py          loading, kappa, SapBERT, decomposition
             build_tables.py    .ann files    -> results/*.csv + figure1_stats.json
             build_figure1.py   results/*.csv -> figures/manuscript_figure1/
results/     the seven tables behind the figure, and the stats it annotates itself with
figures/manuscript_figure1/figure1.png     the composite, the manuscript figure
figures/manuscript_figure1/panels/*.png    each panel alone, for talks
```

Run in order. `build_tables.py` needs SapBERT and takes a few minutes;
`build_figure1.py` recomputes nothing and is instant.

```
python final_analyses/HumanAnnotation/scripts/build_tables.py
python final_analyses/HumanAnnotation/scripts/build_figure1.py
```

Every table feeds a panel. Nothing else is computed.

| Table | Panel |
|---|---|
| `pair_agreement_value_string.csv` | a, lower triangle |
| `pair_agreement_value_sapbert.csv` | a, upper triangle |
| `agreement_decomposition.csv` | b |
| `label_agreement_presence.csv`, `label_mention_rate.csv` | c |
| `value_match_by_label.csv` | d |
| `value_movement_by_category.csv` | e |
| `figure1_stats.json` | the numbers every panel prints on itself |

## What Figure 1 shows

**a. Pairwise value-level agreement, all 13 identities.** One matrix, both
triangles value-level kappa over the same items with the same pooling. The only
difference is what counts as the same value: exact string identity below the
diagonal, SapBERT-cluster identity above it, where two values within cosine 0.70
share a category. So the two triangles are a genuine before and after, and the
upper one is higher for 77 of 77 pairs, mean +0.11.

Two earlier versions of this panel were wrong in ways worth recording. The first
put presence kappa in the lower triangle. That is a different and easier
question, so the upper triangle sat *below* it and read as SapBERT destroying
agreement. The second used Greens against Purples, which place equal kappas at
visibly unequal lightness and manufactured a drop that was not in the numbers.
Greens and Blues are both ColorBrewer single-hue sequential ramps with matched
lightness profiles, which is why they are used here.

**b. Model agreement is not shared silence.** The decomposition that answers the
first objection a statistically-minded reviewer will raise about the model-model
result: presence agreement is scored over every document-field slot, most of them
empty, so two sparse annotators would agree trivially often.

The rows do not share a denominator and the panel says so. Raw agreement, chance
agreement and both-absent share are shares of all slots. Positive specific
agreement is a share of only the slots at least one rater marked. Cohen's kappa
is not a share at all: it is the fraction of the room above chance that was
used, `(raw - chance) / (1 - chance)`. A kappa of 0.43 sitting numerically below
a chance agreement of 0.56 therefore means nothing; raw agreement was 0.75, the
chance floor 0.56, leaving 0.44 of headroom of which 0.19 was used.

Drawn as boxes with every pair as a point on top, not as bars. The groups
have 35, 27 and 3 members, and a bar showing only the mean hides that the
model-model claim rests on three pairs.

| | human-human (n=35) | model-human (n=27) | model-model (n=3) | model-model vs human-human |
|---|---:|---:|---:|---|
| raw agreement | 0.75 | 0.72 | 0.85 | +0.10, p = 0.018 |
| chance agreement | 0.56 | 0.53 | 0.51 | -0.04, n.s. (p = 0.26) |
| both-absent share | 0.54 | 0.48 | 0.50 | -0.04, n.s. (p = 0.55) |
| positive specific agreement | 0.60 | 0.61 | **0.82** | +0.22, p = 0.023 |
| Cohen's kappa | 0.43 | 0.40 | **0.69** | +0.26, p = 0.018 |

Three tests, because three different questions:

- **Is agreement above chance?** Paired within each pair, Wilcoxon signed-rank
  on `raw - chance`. Human-human p = 2.9e-11, model-human p = 2.8e-06. The
  robust statement is the count, since the difference is positive in 35/35,
  27/27 and 3/3 pairs. With n = 3, model-model cannot reach p < 0.05 by a
  signed-rank test at all, and the panel says so rather than printing a
  p-value that looks like one.
- **Do the three groups differ?** Kruskal-Wallis: kappa p = 0.015, positive
  agreement p = 0.014, chance agreement p = 0.0041, both-absent share p =
  0.00083.
- **Model-model against human-human specifically?** An exact permutation test
  that permutes which **raters** are models, not which pairs are model pairs.
  Each rater sits in up to eleven pairs, so a pair-level rank test would treat
  strongly dependent observations as independent. With 12 raters and 3 models
  there are only C(12,3) = 220 assignments, so the test is enumerated exactly;
  the smallest attainable two-sided p is 1/220 = 0.0045. This is the column in
  the table above, and it is the one to quote.

The result the panel needs is the pattern, not any single p. Kappa and positive
specific agreement are both significantly higher for the model pairs, while
chance agreement and both-absent share are **not** significantly different.
That is exactly what "the model agreement is not shared silence" predicts: the
two quantities that would have to differ for the shared-absence explanation to
work do not, and the two that measure real agreement do.

Chance agreement is *lowest* for the model pairs, which is the opposite of what
shared sparsity predicts. The both-absent share is comparable across groups.
Positive specific agreement discards the both-absent cell entirely, so it cannot
be inflated by shared silence at all, and it widens the gap rather than closing
it. Prevalence shows the mechanism: humans mark 17% to 44% of slots present, a
2.6-fold spread, while the three models span 41% to 43%. Much of the human
disagreement is a difference in threshold, and the models share one.

The harmonised consensus is excluded from this panel. It is derived from the
annotators, so pairing it with them measures reconciliation, not independent
agreement.

**c. Difficulty is not a function of rarity.** Per-label presence kappa over
human pairs against how often the consensus tags the label. Spearman rho = +0.07,
p = 0.65 over 42 labels.

Note a correction here. The old panel scattered human-pair means but annotated a
rho computed over *all* pairs, models included, which gave rho = -0.05, p = 0.73.
Both are null results and the conclusion is unchanged, but the number in the
manuscript should be the one that matches the plotted points.

**d. Value-level agreement: exact string vs conceptual match.** Where both
annotators tagged a field, did they write the same thing. Pooled over 8,286
both-tagged instances: 73% exact, 15% recovered by SapBERT, 12% genuine
disagreement.

**e. Forgiving wording lifts technical fields most.** Panel a's before and after,
pooled per metadata category over human pairs. Technical fields gain most
(+0.13), biological next (+0.08), experimental design least (+0.05). Per label
this would be forty-two small shifts, which is a list rather than a result; three
numbers answer the question of whether the wording problem is concentrated
anywhere.

## Who counts as a human annotator

There are **nine** human annotators, not eight. `SingleHuman` is Ian, who
annotated all 27 manuscripts; his `MultiHuman/Ian/` copy is dropped in
`common.py` as byte-identical, so he appears only under that name. A1 to A8 are
the batch annotators, each covering an overlapping subset. He is easy to lose:
`HUMAN_PAIRS` in both scripts names both `MultiHuman vs. MultiHuman` and
`SingleHuman vs. MultiHuman`, and filtering to the first alone silently drops a
whole rater.

## The model columns

Three model sets from `EuBIC-annotation/s0_model_iaa_27/final_annotations/`.
Integrating them needed four fixes, each of which would have failed silently:

1. **Delimiter.** Model `.ann` files separate T-line fields with `|||`, not tabs.
   The standard brat reader returns *nothing* from them rather than erroring, so
   a naive integration yields empty model columns that look like a plotting bug.
   `pipe_entries()` handles them.
2. **Key.** Model files are named by PXD accession; every human set is named by
   PMID. `pxd_pmid_mapping.csv` was derived by matching manuscript text, verified
   one-to-one across all 27, and is committed rather than recomputed. One pair
   (PXD019394 / 21183079) had weak text overlap and was confirmed by title
   against the PRIDE metadata.
3. **GPT dropped.** The EuBIC-era GPT set is superseded by the three pinned
   models, its version was never recorded, and it depressed the model-model
   figure: including it gave 0.55 where the three current models give 0.69. The
   files remain in `EuBIC-annotation/GPT/`; remove the name from
   `EXCLUDE_IDENTITIES` in `common.py` to bring it back.
4. **Role.** `pair_category()` classified anything unrecognised as a MultiHuman
   annotator, so the models would have been counted as humans and contaminated
   every human-human statistic. Model roles are now explicit, and the human
   numbers were unchanged by the fix, which is the check that it worked.

## Filters applied throughout

A label is kept only if the harmonised consensus mentions it in at least 3 of 27
papers **and** at least 2 unique MultiHuman annotators tagged it somewhere. 68
raw labels reduce to 42. The second filter matters: seven labels appeared in the
consensus despite being tagged by a single annotator, which violated the
documented exclusion rule and inflated the consensus-agreement figure until it
was fixed.

`METHODS.md` carries the methodological detail.
