# Materials and Methods: metadata annotation agreement (Figure 1)

How every number in Figure 1 is computed. Code references point to functions in
`scripts/common.py` unless noted; the tables are written by
`scripts/build_tables.py` and drawn by `scripts/build_figure1.py`, which
recomputes nothing.

## 1. Data source and scope

All analyses read exclusively from `EuBIC-annotation/` at the repository root:
`SingleHuman/` (one expert, independent annotation of all 27 papers),
`MultiHuman/<name>/<batch>/` (eight independent EuBIC-MS annotators, each
covering a subset of the 27 papers), `HarmonizedHuman/` (a curator's consensus
per paper), and `s0_model_iaa_27/final_annotations/<model>/ann_files/` (three
model annotation sets). No file outside `EuBIC-annotation/` is read.

The annotated text is PMC abstract and methods prose with subheadings stripped,
from `EuBIC-annotation/pmc_abstract_methods_nosubheadings/`. The PRIDE metadata
distributed with that corpus is held as structured CvParam entities in
`PRIDE_TextSpace_and_FullMetadata.json` and its slot in the annotatable text
space is empty for all 27 records, so no annotation in this analysis derives
from PRIDE. This differs from the extraction pipeline, which is given PRIDE
project properties rendered as text.

`MultiHuman/Ian/` is excluded during loading (`load_annotators`): its 27 `.ann`
files were confirmed byte-identical to `SingleHuman/`, so including it would
double-count one rater. `GPT/` is excluded via `EXCLUDE_IDENTITIES`: it is the
EuBIC-era model pass, its version was never recorded, and it is superseded by
the three pinned models. This leaves **13 identities**: SingleHuman,
HarmonizedHuman, Annotator1 to Annotator8 (`anonymize`, key written to
`EuBIC-annotation/annotator_name_mapping.csv`), and Qwen, Gemma, GLM.

Three annotation file formats exist and are parsed differently. Getting this
wrong fails silently rather than erroring, in both directions:

- `SingleHuman/` and `MultiHuman/*/*/*.ann` are brat standoff files, one entity
  per line as `T<id>\t<Label> <start> <end>\t<span text>` (`brat_entries`). The
  label is the first whitespace-separated token of field 2; the value is field 3.
- `HarmonizedHuman/` is a flat `Label : value` file, one entity per line
  (`flat_entries`). Parsing these with the brat reader silently zeroes the
  consensus rows.
- The model sets are brat-shaped but `|||`-delimited rather than tab-delimited
  (`pipe_entries`). The brat reader returns nothing at all from them, which
  presents as empty model columns rather than as an error.

The model sets are keyed by PXD accession while every human set is keyed by
PMID. `EuBIC-annotation/pxd_pmid_mapping.csv` was derived by matching manuscript
text, verified one-to-one across all 27, and is committed rather than recomputed
(`pxd_to_pmid`). One pair, PXD019394 to 21183079, had weak text overlap and was
confirmed by title against the PRIDE metadata.

`pair_category` assigns each pair an explicit role. Model roles are named
explicitly because the earlier fallback classified anything unrecognised as a
MultiHuman annotator, which would have counted the three models as human
annotators in every human-human statistic.

## 2. Label vocabulary and filtering

The 68 distinct raw entity labels observed across all sources were enumerated
directly from the files, not taken from an external schema. Each was manually
assigned to `biological`, `technical` or `experimental_design` in
`LABEL_CATEGORY`.

Two filters apply, and a label must pass both (`load_filtered_data`):

1. **Mention count** (`MIN_MENTIONS = 3`): tagged by HarmonizedHuman in at least
   3 of the 27 papers. Below this, per-label statistics rest on one or two
   observations.
2. **Single-annotator exclusion** (`MIN_UNIQUE_ANNOTATORS = 2`): tagged by at
   least 2 distinct MultiHuman identities somewhere in the corpus, not
   necessarily the same paper. This implements the exclusion rule the project's
   own consensus procedure is documented to use. Seven labels passed filter 1
   and failed this one, and had been inflating the consensus-agreement figure.

68 labels reduce to **42**. Both dropped lists are printed by
`build_tables.py`.

## 3. Presence Cohen's kappa

For each kept label and identity, a binary presence indicator per paper: 1 if
the `.ann` file contains any entry for that label on that paper, 0 otherwise
(`build_presence`). This discards the value and asks only whether the label was
tagged.

Cohen's kappa over paired binary items (`cohen_kappa`):

```
p0    = fraction of items where a and b agree
pe    = p(a=1)*p(b=1) + p(a=0)*p(b=0)
kappa = (p0 - pe) / (1 - pe)
```

`per_label_kappa` computes this per (pair, label) using only that label's items
over the papers the pair shares. This feeds **panel c** and is written to
`results/label_agreement_presence.csv`. Presence kappa pooled across all labels
per pair is not plotted; it appears per pair in the decomposition table
(Section 6).

## 4. Text-mention rate (panel c, x axis)

For each kept label, the fraction of the 27 papers where HarmonizedHuman's
presence indicator is 1, computed from the consensus annotation files rather
than any cached percentage. Written to `results/label_mention_rate.csv`.

**Rank correlation.** Spearman's rho between each label's mean pairwise presence
kappa and its mention rate, over **human pairs only**, using
`scipy.stats.spearmanr`: rho = +0.07, p = 0.65 over 42 labels, recorded in
`figure1_stats.json` and printed on the panel.

Human pairs means both `MultiHuman vs. MultiHuman` and `SingleHuman vs.
MultiHuman`. The single expert annotated all 27 papers and is a full rater;
filtering to the first category alone drops him and 336 of the 1,512 human
pairs. An earlier version of this figure scattered human-pair means while
annotating a rho computed over all pairs including the models, which gave
rho = -0.05, p = 0.73. Both are null, but the reported number should be the one
matching the plotted points.

## 5. SapBERT embedding: model choice and threshold

Embeddings use `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `[CLS]` token,
L2-normalised, max length 128 tokens (`SapBertScorer`). This is the model
already used in `framework/normalization/ontology.py`, chosen after comparing it
against two alternatives on this dataset's own values:

- Mean-pooled `allenai/scibert_scivocab_uncased` does not separate known synonym
  pairs from unrelated pairs for these short domain terms; unrelated pairs
  scored up to 0.82, overlapping the range of true synonyms.
- `pritamdeka/S-PubMedBert-MS-MARCO` separates them cleanly but has no precedent
  threshold anywhere in this project.

The 0.70 cutoff is the project's established "accepted as normalised" threshold
in `framework/normalization`, reused rather than tuned separately. On a
negative-control set of value pairs drawn from genuinely unrelated labels, for
instance an Organism value against a CleavageAgent value, SapBERT cosine never
exceeded 0.49 on this dataset.

## 6. Decomposition of presence agreement (panel b)

Presence kappa is scored over every (paper, label) slot and most slots are empty
for most raters, so two sparse annotators agree on many absences. Kappa is meant
to subtract exactly that, but kappa misbehaves with skewed marginals, so the
correction is tested rather than assumed. `decompose_pair` returns the full 2x2
table per pair plus:

| quantity | definition | what it tests |
|---|---|---|
| `raw_agreement` | `(n11 + n00) / n` | p0 |
| `chance_agreement` | `pa*pb + (1-pa)*(1-pb)` | pe. If shared sparsity inflated model kappa, pe would be **higher** for model pairs |
| `both_absent_share` | `n00 / n` | the worry, measured directly |
| `positive_agreement` | `2*n11 / (2*n11 + n10 + n01)` | the Dice/F1 form. n00 does not appear, so shared silence cannot inflate it |
| `prevalence_a`, `prevalence_b` | `(n11 + n10) / n` | how much each rater marks at all |

Written to `results/agreement_decomposition.csv`, one row per pair, grouped by
`identity_group` into human-human, model-human and model-model. `prevalence`
computes the same quantity per rater over all their own slots.

**HarmonizedHuman is excluded from this panel.** It is derived from the
annotators, so pairing it with them measures reconciliation rather than
independent agreement.

**The rows do not share a denominator, and the panel is drawn to say so.** Raw
agreement, chance agreement and both-absent share are shares of all slots and
can be read against each other. Positive specific agreement is a share of a
smaller denominator, the slots at least one rater marked. Cohen's kappa is not a
share: it is the fraction of the available room above chance that was used.
Without the divider, a kappa of 0.43 next to a chance agreement of 0.56 reads as
agreement falling short of chance, which is not what it means. For the
human-human group, p0 = 0.75 against pe = 0.56 leaves 1 - 0.56 = 0.44 of
headroom, of which 0.75 - 0.56 = 0.19 was used, so kappa = 0.19/0.44 = 0.43.

### 6.1 Significance testing

Three questions, three tests, and the pairs are not independent, which
constrains what can be asked.

**Is agreement above chance?** A paired comparison within each pair, so a
Wilcoxon signed-rank test on `raw_agreement - chance_agreement` per group,
one-sided. Human-human p = 2.9e-11 (n = 35), model-human p = 2.8e-06 (n = 27).
The pairs share raters, which makes these p-values anticonservative, so the
robust statement is the count: the difference is positive in 35/35, 27/27 and
3/3 pairs. For model-model, n = 3 cannot reach p < 0.05 under a signed-rank
test regardless of the data, so the exact sign-test floor (0.5^3 = 0.125) is
reported instead of a p-value.

**Do the three groups differ at all?** Kruskal-Wallis across the three groups
per metric: kappa H = 8.46, p = 0.015; positive agreement H = 8.55, p = 0.014;
chance agreement H = 10.98, p = 0.0041; both-absent share H = 14.19,
p = 0.00083. Same independence caveat.

**Model-model against human-human?** `rater_permutation_test` permutes which
**raters** are models rather than which pairs are model pairs. Each rater
appears in up to eleven pairs, so one unusual annotator moves many pairs at
once and a pair-level rank test would count them as independent evidence.
Permuting rater group membership respects that structure. With 12 raters
(SingleHuman, Annotator1-8, and the three models; HarmonizedHuman is excluded
from this panel) and 3 models there are C(12,3) = 220 assignments, so the test
is enumerated exactly rather than sampled. The statistic is the difference
between the model-model mean and the human-human mean, and the test is
two-sided. The smallest attainable p is 1/220 = 0.0045.

| metric | model-model minus human-human | p |
|---|---:|---:|
| Cohen's kappa | +0.263 | 0.018 |
| positive specific agreement | +0.216 | 0.023 |
| raw agreement | +0.102 | 0.018 |
| chance agreement | -0.043 | 0.264 |
| both-absent share | -0.039 | 0.545 |

The pattern is the result, not any single p. Kappa and positive specific
agreement are significantly higher for the model pairs; chance agreement and
both-absent share are not significantly different. The two quantities that
would have to differ for shared absence to explain the model result do not,
and the two that measure real agreement do.

## 7. Value-category Cohen's kappa (panel a)

Section 3's kappa treats "present" as one category regardless of value. Here
each rater's category for a (paper, label) item is the value they extracted, or
the sentinel `ABSENT`, and kappa is computed over those paired categorical
assignments (`multicat_kappa`, the same p0/pe formula generalised to any number
of categories). Two schemes differ only in how values become categories:

- **Exact string identity** (`canonical_string_category`): the category is the
  sorted, normalised set of value strings, joined. Two entries match only if
  their value sets are identical after case and whitespace normalisation
  (`normalize_value`). → **panel a, lower triangle**,
  `results/pair_agreement_value_string.csv`.
- **SapBERT-cluster identity** (`canonical_sapbert_category`): every distinct
  value string used for a label, across all raters and papers, is embedded once
  and clustered by single-link connected components at cosine >= 0.70
  (`cluster_label_values`, union-find). The category is the set of cluster IDs
  the entry's values fall into, so values that differ in wording but embed as
  the same concept share a category. → **panel a, upper triangle**,
  `results/pair_agreement_value_sapbert.csv`.

`pooled_multicat_kappa` pools every (label, paper) item a pair shares into one
kappa per pair, so both triangles are the same items under the same pooling and
differ in exactly one thing. Merging categories can only raise raw agreement, so
the upper triangle should exceed the lower one. It does for **77 of 77** pairs,
mean +0.11, range +0.04 to +0.20.

Two notes on how this panel is drawn, both recording errors that produced a
plausible but wrong reading:

- The lower triangle must be *value* kappa, not presence kappa. Presence is a
  different and easier question, since two raters who both tagged a field but
  wrote unrelated values agree on presence and disagree on value. With presence
  below the diagonal, the SapBERT triangle sits lower and the panel reads as
  SapBERT destroying agreement.
- The two colormaps must have matched lightness ramps. Greens against Purples
  places equal kappas at visibly unequal lightness and manufactures a drop that
  is not in the numbers. Greens and Blues are both ColorBrewer single-hue
  sequential ramps and do not have that problem.

Both triangles use a sequential 0-to-1 scale rather than a diverging -1-to-1
scale, since no pooled kappa observed anywhere in this dataset is negative.

## 8. Value-level match type (panel d)

For every (pair, label, paper) triple where **both** raters tagged the label
with at least one value (`value_agreement`):

- **Exact match**: the two value sets, case-folded and whitespace-normalised,
  share at least one element.
- **Semantic match**, checked only when the exact match fails: the maximum
  pairwise SapBERT cosine between any value from a and any from b is >= 0.70.
- Otherwise **no match**.

`results/value_match_by_label.csv` holds, per label, the percentage of
both-tagged instances in each bucket. Pooled over 8,286 instances: 73.4% exact,
15.0% recovered by SapBERT, 11.6% residual disagreement.

Note that this and Section 7 measure related but different things, which is why
the SapBERT gain looks larger here than in panel a. Recovering 15% of instances
raises p0 substantially, but collapsing distinct value strings into fewer, larger
clusters also raises pe, because fewer and larger categories are easier to land
on by chance. Kappa nets the two against each other.

## 9. Value-kappa movement by category (panel e)

Panel a's before and after, pooled per metadata category rather than per label.
For each category, `pooled_multicat_kappa` is applied to that category's labels
only, per annotator pair, under both schemes; `results/value_movement_by_category.csv`
holds one row per (category, pair) with `string_kappa`, `sapbert_kappa` and
their difference. The panel plots the mean over **human pairs**, with the
individual pairs shown as faint points behind each mean.

Pooling per category is deliberate. Per label this is 42 small shifts, which is
a list rather than a result; three numbers answer whether the wording problem is
concentrated anywhere. It is: technical +0.13, biological +0.08, experimental
design +0.05.

Across all 231 (category, pair) observations, SapBERT raises value kappa in 216
and lowers it in 15, by at most 0.028. The direction is not guaranteed even
though merging categories cannot lower raw agreement, because kappa's chance
correction can move against a merge that changes the marginals.
`figure1_stats.json` records the count rather than the analysis claiming the
gain is universal.
