# Results section draft: retroactive annotation and MLMarker application

Status: placeholder text. Every `[Pn]` is a number that does not exist yet.
The questions at the end say which analysis produces it and who owns it.
Do not quote any number from this file. Prose is provisional and written to
make the shape of the argument visible, not to be final.

---

## Retroactive annotation of 1,737 human PRIDE datasets

We tested the potential of HAMLET in the context of improving training data for
the MLMarker model [REF]. We selected 1,737 potentially interesting human PRIDE
datasets from the PRIDE API. However, due to the lack of open-access papers or a
text-mining-permissive license, we reverted to PRIDE descriptors for `[P1]`
datasets. This is a ceiling on the approach rather than a property of this
selection: across human PRIDE as a whole, `[P2]` per cent of datasets have an
open-access, text-mining-permissive publication at all.

The remaining `[P3]` datasets carry both a deposited record and an accessible
article, which allows a direct question: what does the published article
contribute over the deposited record, field by field?

Because every extracted value carries the sentence it came from, and because
structured PRIDE descriptor lines are separable from manuscript prose, the
source of each value is recoverable after extraction without rerunning
anything. Averaged across all `[P4]` fields, the PRIDE record alone fills
`[P5]` per cent of dataset-field slots. Adding the open-access manuscript
raises this to `[P6]` per cent, a gain of `[P7]` percentage points. The gain is
uneven, and that unevenness is the finding: the repository record covers what a
submission form asks for, while sample preparation and acquisition detail exists
almost exclusively in the Methods section. `[P8]` fields are supplied by the
manuscript in more than `[P9]` per cent of datasets while the record supplies
them in fewer than one per cent.

Coverage is only half the comparison. Where both sources speak, do they say the
same thing? These are two separate questions.

Coverage of entities: do we find the same entities in the manuscript and the
PRIDE summary, independent of their value? Of `[P10]` dataset-field slots,
`[P11]` are populated by both sources, `[P12]` by the manuscript alone and
`[P13]` by the record alone.

Agreement of values: where we find the same entities, in how many cases do they
agree? Scored with the HAMLET matching tiers, concordance rises from `[P14]` per
cent under exact string matching to `[P15]` per cent under normalisation,
`[P16]` per cent under ontology identity, `[P17]` per cent once hierarchical
relations are allowed and `[P18]` per cent under semantic matching. A naive
exact-identifier comparison reports `[P19]` per cent discordance where the full
matcher finds `[P20]` per cent.

Comparing metadata sources by string equality therefore overstates conflict by a
wide margin, and any study that does so will conclude that repositories and
publications contradict each other more often than they do.

The residual `[P20]` per cent is not an error rate. We adjudicated a random
sample of `[P21]` discordant pairs by hand into complementary values, schema
mismatch, genuine contradiction and matcher miss. Genuine contradiction accounts
for `[P22]` per cent of the residual.

---

## Scoring agreement without inflating it

The same matching tiers are used in three distinct comparisons in this work:
extraction accuracy against SDRF ground truth, agreement between the deposited
record and the manuscript, and agreement between text-mined and
expression-derived tissue annotation. In all three, tiered matching raises the
measured score by construction, so the tiers themselves require justification.
Three checks establish that the reported scores are calibrated rather than
generous.

First, the tier boundaries are placed empirically. Against a permutation null in
which annotations are shuffled between datasets, random pairs reach a similarity
of `[P23]` where real pairs reach `[P24]`, and the boundaries fall where the two
distributions separate.

Second, the conclusion does not depend on the scoring function. `[P25]`
alternative ontology similarity measures, spanning path-based, information-
content and set-overlap formulations, rank the same pairs consistently
(Spearman `[P26]`).

Third, the chosen measure was validated against an external reference with no
annotation from this work involved: across all tissue pairs in the Human Protein
Atlas, it reproduces known anatomical and physiological relationships.

Agreement statistics are meaningless without knowing what agreement by chance
looks like. This applies as much to ontology-graded matching as it does to
inter-annotator agreement, where the same correction is conventional.

---

## A corpus of healthy human tissue

Out of the 1,737 annotated datasets, `[P27]` contained healthy human tissue.
Indeed, the majority of public proteomics data does not contain healthy human
primary tissue but often contains cell lines or cultured material: `[P28]`
datasets report an established cell line and `[P29]` report cultured material.
However, due to project-level ambiguity, `[P30]` of those studies mix healthy
with diseased samples, or treated with control, or combine multiple tissues.
Such projects do contain usable runs but lack a label that tells them apart.

Two kinds of ambiguity are involved and they are not equally tractable. `[P31]`
projects span multiple tissues, where the missing distinction is anatomical.
`[P32]` projects mix phenotypes, healthy against diseased or treated against
control, where the missing distinction is not anatomical at all.

This lack of annotation granularity can be partially addressed by applying
classifier tools such as MLMarker [REF], which predicts tissue of origin from
protein abundances at individual run level. Only `[P33]` of the `[P34]` datasets
in the reprocessing database carry SDRF, the sole existing source of true
per-run annotation, so for the remainder there is no per-file label to inherit.

We applied MLMarker to `[P35]` runs across `[P36]` projects for which HAMLET also
produced a tissue annotation. MLMarker resolved a single project-level label
into distinct run-level assignments in `[P37]` projects, in one case into
`[P38]` distinct tissues.

The resolution is partial, and the limit follows from what the classifier
predicts. It separates tissues and not phenotypes, so it addresses the `[P31]`
multi-tissue projects and leaves the `[P32]` phenotypic ones untouched.
Expression-based classification resolves the anatomical ambiguity in
project-level annotation and not the phenotypic ambiguity, and the phenotypic
share is the larger one.

---

## Agreement between text-mined and expression-derived annotation

Where both HAMLET and MLMarker assign a tissue, the two annotations derive from
independent evidence: one from what the authors wrote, one from what the
instrument measured. Agreement between them is therefore a check on both,
subject to three qualifications that determine how the numbers may be read.

**Circularity.** `[P39]` of MLMarker's `[P40]` training datasets fall inside the
annotated corpus. On those datasets MLMarker reproduces labels it was trained
on, so agreement measures memorisation rather than independent confirmation. We
report concordance separately for the `[P41]` projects MLMarker has seen and the
`[P42]` it has not, and only the held-out figure supports a claim about label
quality.

**Vocabulary.** MLMarker's training atlas excludes biofluids and covers only
tissue classes. A HAMLET annotation naming a tissue outside that set cannot
agree with an MLMarker prediction by construction, and counting it as discordant
conflates two different things. We therefore decompose disagreement into cases
where the HAMLET tissue lies outside MLMarker's label set (`[P43]` per cent of
discordant pairs) and cases where both terms are in vocabulary and the two
sources genuinely differ (`[P44]` per cent).

**Material context.** A HAMLET annotation of skin is correct for a study of
patient-derived fibroblasts, but those cells are cultured and differentiated
before measurement, so their proteome no longer resembles skin. Disagreement is
expected in such cases and is not an error in either source. `[P45]` per cent of
discordant pairs involve material that was cultured before measurement.

With these separations in place, held-out concordance is `[P46]` per cent under
exact matching and `[P47]` per cent under the HAMLET matching tiers. Under
ontology-graded matching, mean similarity between disagreeing pairs is `[P48]`,
a median separation of `[P49]` steps in the anatomical hierarchy: where the two
sources disagree, they rarely disagree wildly.

MLMarker's confidence score tracks this agreement. Above a confidence of
`[P50]`, held-out concordance reaches `[P51]` per cent, which supplies a
threshold for accepting a propagated label automatically rather than by
inspection.

---

## Expanding the MLMarker training set

Applying MLMarker's own inclusion criteria to the 1,737 annotated datasets
identifies `[P52]` projects outside the existing training set that qualify as
solid tissue with a healthy baseline, giving `[P53]` biological samples once
fractions of a single sample are collapsed.

Most reinforce classes the model already has. `[P54]` samples fall in `[P55]`
classes the model lacks entirely, not because those tissues are unsuitable but
because there was too little training data when the model was built. For those
classes the question is not whether performance improves but whether they become
predictable at all, which is a different and stronger claim.

Of these, `[P56]` projects qualified directly from the annotation, while `[P57]`
required manual separation of usable from unusable runs, recovering a further
`[P58]` samples. The split matters because it separates what the pipeline
delivers on its own from what still needs a person.

Retraining on the extended atlas changes macro-averaged F1 from `[P59]` to
`[P60]`, with per-class recall for the added classes at `[P61]`. As a control we
repeated the experiment using project-level labels alone, without per-run
resolution, giving `[P62]`; the difference isolates what run-level annotation
contributed.

---

## What text-derived annotation cannot resolve

Some ambiguity is not resolvable from prose at all. Which runs are fractions of
one sample, which files map to which condition, and whether a stated gradient
matches what was acquired are questions the manuscript does not answer.
Resolving that class of question requires the raw files.

Manuscript-derived annotation closes the biological gap that blocks MLMarker
while being weaker on technical fields. Runs carrying a tissue label rise from
`[P63]` to `[P64]` per cent, and all four MLMarker selection fields improve.
Several technical fields lose coverage, by up to `[P65]` percentage points. The
two sources are complementary, and neither is better overall.

Ontology distance is a proxy for biological similarity and an imperfect one.
Comparing similarity derived from ontology structure against similarity derived
from measured proteomes shows where the two diverge, and the divergence bounds
what any ontology-based agreement score can establish.

Expert review remains necessary at the point where datasets are chosen for a
purpose. One dataset in this corpus reads as clean in every extracted field and
is nonetheless unusable, because its multi-enzyme, heavily fractionated design
is not stated in a form an extractor can act on. Conversely, one dataset the
repository record would have discarded is usable, because the record describes
the material as cell culture while the manuscript states plainly that the cells
came from healthy donors. Both are instances of the record-versus-article
disagreement quantified above, appearing as concrete downstream consequences.

---

# Open questions

These are questions, not method specifications. How you answer them is yours to
decide. Notes flag something that would make a result hard to interpret, not a
required approach.

---

## Hari: the record against the article

**Access and ceiling** `[P1-P3]`

- How many of the 1,737 datasets have no open-access, text-mining-permissive
  publication, so that only the PRIDE descriptor is available?
- Beyond this selection, what fraction of human PRIDE datasets could ever be
  annotated from full text? This is the ceiling on the whole approach and is
  worth stating even approximately.

**Coverage of entities** `[P4-P13]`

- Do we find the same entities in the manuscript and the PRIDE summary,
  independent of their value? Per field: populated by both, by the manuscript
  alone, or by the record alone?
- Which fields does the record essentially never supply, and which does the
  manuscript essentially never supply?
- How much does the manuscript add over the record overall, and how uneven is
  that gain?

**Agreement where both sources speak** `[P14-P20]`

- Where we find the same entities, in how many cases do they agree, using the
  HAMLET matching tiers?
- How much of the apparent disagreement is an artefact of how you match? What
  does exact identifier comparison report against the full tiered matcher?

Worth knowing:

- The matcher is at `framework/benchmark/semantic_matcher.py`. Its tiers are
  exact, normalized, ontology, hierarchical (directed, at most 4 hops, siblings
  excluded) and semantic (SciBERT). Note the split: SapBERT does the ontology
  resolution and the hierarchical tier's term lookup, SciBERT only tier 5.
  Reporting resolution per tier as well as
  cumulatively is more informative than a single number.
- It currently imports as `benchmark.semantic_matcher`, which resolves only with
  `framework/` on the path. Worth fixing once rather than twice, since Arnaud
  needs it too.
- Values whose evidence is marked `inferred` are the model reasoning rather than
  either source quoting text, so including them counts something a reuser cannot
  check.

---

## Arnaud: text-mined against expression-derived annotation

Roughly half of this is a rerun or a restriction of analyses already built, not
new work. Tagged accordingly.

**Calibrating the agreement scores** `[P23-P26]` — largely done, needs framing

This now sits in the main text rather than the supplement, because it answers
the obvious objection to every score in the paper that uses the matcher, and
there are three: HAMLET against SDRF ground truth, the PRIDE record against the
manuscript, and HAMLET against MLMarker. Loosening the matching criterion raises
all three by construction, so why are these tiers right?

- Against a permutation null, how far apart are real and shuffled pairs, and do
  the tier boundaries fall where the distributions separate?
- Do the alternative ontology similarity measures rank the same pairs
  consistently, or does the conclusion depend on which one is chosen?
- Does the chosen measure reproduce known relationships on an external
  reference, independent of any annotation produced here?

The eight-metric scorecard and the full HPA pairwise benchmark stay
supplementary, but they are now supporting a claim the reader has already met.
Three numbers reach the main text.

**Circularity** `[P39-P42]` — new

- How much of the compared corpus is in MLMarker's training set?
- Does concordance differ between the datasets MLMarker has seen and those it
  has not? On the seen ones the model reproduces labels it was trained on, so
  only the held-out figure says anything about label quality.

**Where does disagreement come from** `[P43-P44]` — new

- When the two sources disagree, is it because the HAMLET tissue is not in
  MLMarker's label set at all?
- Which tissues does HAMLET report that MLMarker cannot represent, and how many
  datasets do they cover? That list is the candidate set for new classes, so it
  is a result and not only a diagnostic.
- Note: `FILTER_BY_MLM_VOCAB = True` currently restricts the comparison to
  MLMarker's existing classes, which removes exactly these cases. The
  distinction worth keeping is between tissues the atlas deliberately excludes
  (biofluids, cell types) and tissues it simply lacks a class for (breast,
  cervix, skin). The first are not meaningful comparisons; the second are the
  interesting ones.

**Material context** `[P45]` — new

- HAMLET can say skin and be right while the proteome looks like something else,
  because patient fibroblasts are sampled from skin and then cultured and
  differentiated. How much of the disagreement is this?
- If datasets are classified by material context before scoring, does
  disagreement concentrate in the cultured categories? Asked beforehand this is
  a prediction that can fail, which is more useful than the same point raised
  afterwards as a caveat.

**Concordance** `[P46-P49]` — rerun on a partition

- On held-out data, how often do the two sources agree under exact matching,
  under the HAMLET tiers, and under ontology-graded matching?
- Where they disagree, how far apart are they? The graded score is well suited
  to this and the tiered one is not.

**Confidence** `[P50-P51]` — rerun on a partition

- Does MLMarker's confidence track agreement on held-out data?
- Is there a threshold above which a propagated label could be accepted
  automatically rather than by inspection? That threshold is the practical
  output.

**Granularity** `[P35-P38]` — done, needs redoing per sample

- In how many projects does MLMarker resolve a single project-level tissue into
  distinct run-level assignments?
- Note: fractions of one sample are not independent observations. Counting per
  run inflates both the subdivision count and the apparent agreement, since
  fractions of the same sample agree trivially.
  `MLMarker/Reprocessing_database/parition_combiner/fraction_groups.tsv` maps
  runs to samples.

**Which metric** `[methods]` — extend the existing benchmark

- How does the HAMLET tiered matcher compare against a continuous ontology score
  on the same pairs?
- How does HAMLET's 4-hop limit compare with hub blocking as a way of stopping
  everything collapsing into `organ`? Two solutions to the same problem, and it
  is not obvious which is better.
- Does the continuous score add anything over the tiers here, or do they rank
  the same pairs the same way?
- Note on embeddings: for scoring whether two ontology terms denote the same
  entity, SapBERT is the right model, since it is trained on UMLS synonymy.
  That is what the framework already uses for ontology resolution and for the
  hierarchical tier. Tier 5 of the matcher uses SciBERT by design, so if the
  aim is to compare against the matcher as it runs, SciBERT is the faithful
  choice there; if the aim is the best available tissue-term similarity, use
  SapBERT. State which question is being answered.

**Feeds the discussion**

- Comparing ontology-structural similarity against proteome-derived similarity
  says where the ontology's notion of relatedness and the measured biology
  diverge. That bounds what any ontology-based agreement score can establish,
  including ours, and is better stated by us than by a reviewer.
- The three-way comparison in `Merge_annotation.ipynb` (HAMLET raw, HAMLET
  normalized, the earlier agent output) bears on an open question elsewhere:
  whether the two agent sources should be combined rather than one replacing the
  other. Worth flagging if it points either way.

One practical thing: the notebooks carry local absolute paths and read at least
one input that is not in the repository, so nobody else can reproduce them.
Worth fixing before the results are quoted.

---

## Tine

**The corpus** `[P27-P34]`

- How many of the 1,737 datasets contain healthy human tissue?
- How many report cell lines or cultured material, and does that account for the
  bulk of the exclusions?
- How many contain multiple tissues and would benefit from MLMarker resolution?
- How many contain a phenotypic mixture, healthy against diseased or treated
  against control, which MLMarker cannot resolve because it predicts anatomy and
  not phenotype?
- How many datasets carry true per-run annotation already, via SDRF?

**The residual** `[P21-P22]`

- Of the pairs the matcher leaves discordant, how many are genuine
  contradictions rather than complementary values, schema mismatches or matcher
  misses? Until this is answered the residual cannot be called an error rate.

**The extended training set** `[P52-P58]`

- Are there datasets outside the training data that can be added, in existing
  and in new classes, meeting solid tissue and healthy baseline requirements?
- How many qualify directly from the annotation, and how many need manual
  separation of usable from unusable runs?
- How many samples does the manual work recover, and is that proportionate to
  the effort?
- Note: two eligibility implementations currently disagree. One has to win.

**Coverage gain** `[P63-P65]`

- How much does run-level annotation rise, and which fields lose coverage in the
  process? The technical regressions belong in the result, not only the gains.

---

## Blocked

`[P59-P62]`, the retrained model and its project-level control, need
`pride_quant.parquet` and the `mlmarker` package. Neither is in this repository.

Two decisions will need making when it unblocks:

- Fraction aggregation. Collapsing fractions is currently an identifier
  operation and touches no quantitative data. When the abundance matrix is
  joined, several fraction rows must become one feature vector. Neither the mean
  nor the median of per-run NSAF is right, because each run's vector is
  normalised within that run, so a protein confined to one fraction is diluted
  by the fraction count. Summing counts or intensities first and normalising
  once avoids this.
- Consistency with the existing atlas. If the current training set treats each
  fraction as an independent row, collapsing only the new samples biases the
  comparison either way.

---

## Open in the text itself

- Whether the record-versus-article comparison stands before the MLMarker
  application, as drafted, or folds into it.
- Whether the retraining is in scope. If it stays blocked, the section ends at
  the extended training set, and manual curation of the remaining backlog is
  speculative effort.
- Naming. "Hybrid" in HAMLET referred to the text-plus-raw-data combination and
  no longer applies in this scope.
