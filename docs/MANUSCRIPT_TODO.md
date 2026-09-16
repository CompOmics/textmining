# HAMLET manuscript: completion to-do

Started 2026-09-14. Target document: `docs/HAMLET_draft_092026.docx`.
Tick items as they land; append to the session log at the bottom. `[Pn]`
tokens refer to `docs/section_mlmarker_draft.md`, which defines every
placeholder number and who owns it.

Who does what:

- **Hari (external, own machine)**: staircase inference, six-model benchmark,
  record-vs-article comparison. All three are done and already in the draft
  (text, Figure 2, the six-model panel, and the Methods "Comparison of input
  sources" subsection). Nothing is re-run here; what remains is checking the
  numbers, missing figure files, and prose.
- **Arnaud**: matcher calibration, HAMLET-vs-MLMarker concordance.
- **Tine (this machine)**: corpus counts, residual adjudication, extended
  training set, all writing, integration of everyone's numbers into the docx.

Size tags: S = under half a day, M = 1-3 days, L = more than 3 days.

---

## 0. Decisions (settled 2026-09-14 unless marked open)

- [x] **Matcher**: SapBERT everywhere. Every SciBERT-scored number is replaced
  by its SapBERT counterpart; analyses still on SciBERT are rerun (see A and
  B3). Hari's staircase and six-model tables must be SapBERT-scored; ask him
  to confirm or re-score with `framework/benchmark/semantic_matcher.py`.
- [x] **Retraining MLMarker** is in scope, as the last part of the paper.
  Outline and feasibility in `docs/extended_training_set_outline.md`.
- [x] **Record vs article** stays a standalone section before the MLMarker
  section, as drafted.
- [ ] **Six-model model list**: one set of version strings and inference
  settings in Methods, from Hari.
- [ ] **Name**: what the H in HAMLET stands for now that "hybrid" no longer
  applies.
- [ ] **Eligibility criteria** for the training-set extension: recommended
  in the outline (existing-atlas rule: human, healthy, solid tissue or
  purified cell type, no biofluid, no cell line) and atlas design 2 (rebuild
  from the reprocessing database, HAMLET supplies labels). Confirm with
  Sander and Arnaud.

## A. Numbers to replace in the draft (Tine, S each)

All from committed result files; SapBERT scoring throughout.

- [ ] **Figure 1 text** (`final_analyses/HumanAnnotation/results/figure1_stats.json`
  and a pooled-kappa recomputation with `scripts/common.py`):
  - "SciBERT pass recovers 15.0%" -> SapBERT; value 15.0% is right
    (73.4% exact, 15.0% recovered, 11.6% residual over 8,286 instances).
  - model-model value kappa "0.45 to 0.58" -> 0.45 to 0.57.
  - human-human value kappa 0.34 to 0.44: right.
  - category gains "+0.13 technical, +0.09 biological, +0.05 experimental
    design" -> **+0.11, +0.08, +0.05** (`movement` block). README and
    METHODS.md say +0.13 for technical; the stats file says 0.112. Fix the
    docs to match the stats file.
  - single expert vs annotators "0.45 (0.27 to 0.67)" -> 0.44 (0.27 to 0.67);
    consensus 0.53 (0.30 to 0.74) is right.
  - positive specific agreement "0.82 vs 0.61" -> 0.82 vs 0.60 (human-human
    0.60; 0.61 is model-human).
  - all other Figure 1 numbers checked and correct (0.43, 0.17-0.71, rho
    +0.07 p 0.65, 0.40, 0.69, 0.68-0.72, p 0.018, 0.51 vs 0.56, p 0.26 and
    0.55, p 0.023, 17-44%, 2.6-fold, 41-43%, 77/77 pairs, mean +0.10).
- [ ] **Staircase text** (`final_analyses/SapBERTStaircase/manuscript_figure2/benchmark_metrics.csv`;
  three inference runs re-scored 2026-09-14 with the current matcher
  (numeric equality, tier 4 on, 16 fields); mean +/- SD):

  | Arm | GLM-4.7 | Gemma 4 31B | Qwen3.8-27B |
  |---|---|---|---|
  | S0 | 0.285 +/- 0.050 | 0.655 +/- 0.007 | 0.500 +/- 0.036 |
  | S1 | 0.728 +/- 0.013 | 0.728 +/- 0.008 | 0.717 +/- 0.015 |
  | S2 | 0.772 +/- 0.002 | 0.793 +/- 0.002 | 0.795 +/- 0.004 |
  | S3 | 0.773 +/- 0.009 | 0.792 +/- 0.002 | 0.786 +/- 0.003 |
  | S4 | 0.783 +/- 0.002 | 0.793 +/- 0.002 | 0.804 +/- 0.017 |
  | S5 | 0.763 +/- 0.010 | 0.776 +/- 0.003 | 0.770 +/- 0.018 |

  - "S0 ... ranging from 0.29 to 0.70" -> 0.29 to 0.66.
  - "by S2 ... approximately 0.96 which stays flat through S5" -> ~0.77-0.80
    at S2-S4; S5 costs 1-3 points in every model, from the biological agent
    (0.86-0.91 -> 0.81-0.86). Technical plateaus at 0.93 from S2;
    experimental design at 0.53-0.56 (it does NOT catch up: counts are now
    scored by equality). Lin IC on the ontology-backed fields: 0.80-0.84
    from S2, flat through S5 (`manuscript_figure2/lin_by_cell.csv`).
    (Goldens fixed 2026-09-14: taxon-id prefix `9606 (Homo sapiens)` stripped
    in 11 annotation-sourced goldens; species acceptance 0.76 -> 0.98.)
  - Acceptance decomposition at S5 (`acceptance_decomposition_by_agent.csv`,
    Fig 2e): biological 53 % exact, 2 % ontology/hierarchy, 19 % SapBERT
    tier, 9 % below the tier at cosine >= 0.5, 16 % rejected; technical
    54 / 0 / 37 / 1 / 4 / 4 % missing; experimental design 49 % exact, 4 %
    below tier, 44 % rejected.
  - Output consistency (`manuscript_figure2/consistency_by_cell.csv`):
    non-unknown fields per dataset 7-9 at S0 -> 20-21 from S2; three-run
    unanimity 0.21-0.51 at S0 -> 0.68-0.95 from S1 (Qwen 0.68, Gemma 0.87,
    GLM 0.93). Replaces "4-10 -> 23-25" and "0.05-0.32 -> 0.55-0.90".
  - Figure: `manuscript_figure2/figure2.png` (a-g) and the entity-level
    supplementary `figureS2_entities.png` (heatmap, outcomes with the human
    ceiling from `HumanAnnotation/results/value_match_by_label.csv`, Lin
    severity), drawn by `scripts/draw_figure2.py` from the tables of
    `rescore_replicates.py`, `build_figure2.py`, `build_error_analysis.py`.
  - experimental design S0 spread "0.19 to 0.52" -> recompute from
    `figures/figure3/benchmark_metrics.csv` (ExperimentalDesignAgent_f1_mean).
  - crossover: "Qwen S5 0.961 > Gemma S1 0.898 > Gemma S0 0.700, GLM S0
    0.288" -> Qwen S5 0.855 > Gemma S1 0.761 > Gemma S0 0.647, GLM S0 0.308.
  - consistency and unknown-rate sentences (4-10 -> 23-25 fields; 0.05-0.32
    -> 0.55-0.90 agreement; unknown 0.73-0.91 -> 0.34-0.39) are correct.
- [ ] **Six-model panel**: values in the draft image are 0.90-0.96 and are
  SciBERT-era. Replace once Hari re-scores.
- [ ] Model naming: "GLM-4" -> "GLM-4.7"; "Qwen3.8-27B" everywhere.
- [ ] Ontology count: text says 19; `framework/ontologies/` holds 11 files.
  State the number actually indexed for the production run.
- [ ] Remove the inline note to Arnaud in the framework section once B3 lands.
- [ ] Add the caveat that human annotators saw PMC prose only while the
  pipeline also sees PRIDE properties as text.
- [ ] Panel f of Figure 2 is a structure-vs-model test, not a ranking.

## B. Analyses still needed

### B1. Staircase, Figure 2 (done by Hari; check only)

- [ ] Make the text agree with `final_analyses/SapBERTStaircase/figures/benchmark_metrics.csv`
  (item A). `final_analyses/Staircase/` is deprecated.
- [ ] Figure 2 panels are `final_analyses/SapBERTStaircase/figures/figure3_panel_*.png`
  (a overall F1, b by class, c output volume, d unknown rate, f crossover);
  replace the docx figure with these, or a composite built from them.
- [ ] Panel e (leave-one-out at S5) was never run; leave out unless a
  reviewer asks.

### B2. Six-model benchmark, Figure 3 (done by Hari; check only)

- [ ] The draft shows it as panel D of an older four-panel composite that
  also holds a METI panel (out of scope) and an old master-vs-agents panel.
  Replace with a standalone figure of the six-model panel; ask Hari for the
  source table and script, commit under `final_analyses/ModelComparison/`.
- [ ] Results text and figure must use the same matcher as Figure 2 (decision 0).
- [ ] Update the figure ledger in `final_analyses/README.md` (it still says
  "waiting on the six-model benchmark re-run").

### B3. Matcher calibration (Arnaud, M) `[P23-P26]`

Answers the objection that tiered matching inflates every score in the paper.
Three numbers reach the main text; the scorecard and HPA benchmark stay
supplementary.

- [x] Permutation null and Lin-IC comparison, done 2026-09-14 on the
  `SapBERTStaircase` pairs (3 models x S0-S5, 3,960 pairs):
  `final_analyses/MatcherCalibration/results/summary.md`. Headlines: null F1
  is 0.54-0.58 at S2-S5 against observed 0.85-0.90; 38 % of shuffled pairs
  clear the 0.70 tier-5 threshold and 55 % the 0.50 F1 acceptance; all 208
  numerically different values are counted as TP (experimental-design F1
  0.90 -> 0.58 with numeric equality, overall 0.89 -> 0.78); Lin rises at S5
  where the matcher score falls. Changes to the matcher/F1 rule listed at
  the end of that file; decide which go into the paper before the
  staircase tables are final.
- [x] **Numeric equality in the matcher, applied 2026-09-14** (Tine: essential).
  `framework/benchmark/semantic_matcher.py` now compares numbers as numbers.
- [x] Staircase re-scored with the patched matcher, all three replicates:
  `final_analyses/SapBERTStaircase/scripts/rescore_replicates.py` ->
  `benchmark_runs_current_matcher/`, `manuscript_figure2/`. Hari's
  `benchmark_runs/` and `figures/benchmark_metrics.csv` predate the fix
  (15:50 vs 17:40 on 2026-09-14) and are superseded.
- [ ] **Hari: six-model benchmark with the patched matcher** (Figure 3); the
  experimental-design class will drop by ~0.35 there too.
- [ ] `SDRFS_github/PXD032098_TechnicalAgent_golden.json` has
  `instrument = "1; 2; 3; 4"` (SDRF column parsed wrong); fix by hand or in
  `sdrf_to_golden.py`.
- [ ] Rank concordance against Resnik, Jaccard, GraphIC on Arnaud's tissue
  pairs (`mlmarker_hamlet/output/Try_distance/all_metrics_comprehensive_comparison.csv`)
  if still wanted; Lin and Wu-Palmer are covered above.
- [ ] External check on Human Protein Atlas tissue pairs:
  `mlmarker_hamlet/output/HPA_Benchmark/`. Note
  `merged_mlmarker_qwen3_runs.tsv` there is 77 bytes, an empty write.
- [ ] 4-hop limit vs hub blocking: report which is used and why.
- [ ] Rerun `mlmarker_hamlet/script/SciBERT_tissue_similarity.ipynb` with
  SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, [CLS], L2-norm,
  as in `framework/benchmark/semantic_matcher.py`) and replace
  `mlmarker_hamlet/output/SciBERT/` with a SapBERT output directory. Any
  tier-5 or embedding number in B6 comes from that rerun.
- [ ] Retire the SciBERT wording in `docs/section_mlmarker_draft.md:261-263`.
- [ ] Fix once: `benchmark.semantic_matcher` only imports with `framework/`
  on `sys.path` (`framework/benchmark/__init__.py`).

### B4. Record vs article, Figure 4 (done by Hari; integrate) `[P1-P20]`

- [ ] The Methods subsection "Comparison of input sources" is written, and
  the two figures at the end of Methods are **broken links** in the docx
  (`/media/image4.png`, `/media/image5.png` point outside the file). Get the
  PNGs and source tables from Hari and embed them.
- [ ] The Results section still has two sentences and an `[XXXX]` for the
  PRIDE-descriptor-only count. Write it from Hari's tables: per-field
  coverage (both / manuscript only / record only), agreement per tier and
  cumulative, exact-identifier vs full-matcher discordance, `[P1]` and the
  open-access ceiling `[P2]`.
- [ ] Check `inferred:` values were excluded from agreement counts.
- [ ] Commit tables and figure under `final_analyses/RecordVsArticle/`; the
  `MLMarker/analysis/` path named in `final_analyses/README.md` does not exist.
- [ ] **Residual adjudication (Tine, M)** `[P21-P22]`: hand-classify a random
  sample of about 100 discordant pairs from Hari's output into complementary,
  schema mismatch, genuine contradiction, matcher miss. Needs Hari's
  discordant-pair table with both values and evidence sentences.

### B6b. MLMarker granularity and disease (done 2026-09-14)

- [x] **File-level disambiguation, validated on 5 projects / 287 files**
  (`MLMarkerPenalty/results/concordance_figure/summary.md`): choice from
  HAMLET's list correct on 84 % of held-out files (72, chance 0.21) and 86 %
  overall; MLMarker alone extends a too-short HAMLET list in 54 % of cases;
  22 % of files have a tissue outside MLMarker's 34 classes. PXD010296
  dropped (rat spleen).
- [ ] `build_run_labels.py`: an AMBIGUOUS organism ("rattus norvegicus;
  homo sapiens") is currently counted as human; treat as not human.
- [ ] Optional truth for PXD010271, PXD005693, PXD048647 (PRIDE SDRF?).
- [x] **HAMLET proposes, MLMarker chooses**
  (`final_analyses/MLMarkerPenalty/results/propose_choose/summary.md`, figure
  there): held-out runs with SDRF / run-name truth, 4 projects / 123 runs:
  MLMarker alone 0.58, restricted to HAMLET's list 0.93 (multi-tissue
  projects only: 0.78 vs chance 0.50, 41 runs); in-training projects 0.97
  either way. Run-level atlas: 3,500 labelled usable runs, 2,136 outside the
  training projects, 67 of them healthy native (`atlas_runs.tsv`).

- [x] `final_analyses/MLMarkerPenalty/results/granularity/summary.md`:
  multi-tissue projects, MLMarker top-1 within HAMLET's list 63 % of usable
  runs (chance 26 %), restricted-to-list accuracy 78 % vs run-level truth
  (2 projects, 41 runs; unrestricted 34 %). Healthy vs diseased: no
  separation (within-project AUC 0.48 on the single labelled project;
  across projects diseased are not predicted worse). Write as "HAMLET
  proposes, MLMarker chooses", validated on two projects; disease stays a
  text label.

### B5. Healthy human tissue corpus (Tine, M) `[P27-P34]`

Input: `mlmarker_hamlet/output/HAMLET_normalized.tsv` (1,737 rows, one per
PXD; `tissue`, `disease_state`, `material_type`, `cell_line`, `sample_source`
as value/ontology dicts). Write a script under
`final_analyses/HealthyCorpus/scripts/` that reports:

- [ ] datasets with healthy human primary tissue (`disease_state` normal or
  disease free, `material_type` tissue, no cell line);
- [ ] datasets reporting a cell line or cultured material (rough counts seen:
  cell culture 420, cell line 411, tissue 246, biofluid 164);
- [ ] multi-tissue datasets (anatomical ambiguity, MLMarker can resolve);
- [ ] phenotype-mixed datasets, healthy vs diseased or treated vs control
  (MLMarker cannot resolve);
- [ ] datasets with per-run SDRF already: 57 PXDs in
  `MLMarker/Reprocessing_database/notebooks/run_metadata_sdrf.tsv`.
- [ ] Do the same counts for the 233 projects MLMarker was applied to.

### B6. HAMLET vs MLMarker concordance (Arnaud, M) `[P35-P51]`

Arnaud's commit 8ba4885f (2026-09-14) adds `mlmarker_hamlet/script/MLMarker_v_HAMLET.ipynb`
with outputs under `mlmarker_hamlet/output/HAMLETv3/MLMarker_v_HAMLET/`: a
new MLMarker run over all 62,329 runs with top-5 classes
(`output/mlmarker/run_meta_mlmarker_all.tsv`, input `sample_nsaf_matrix.parquet`,
provenance to be documented), the 183 training PXDs excluded, cohorts
healthy-native / skin / cancer, Lin-IC tiers, the cultured-fibroblast case
study, and an AI-generated decomposition report (`HAMLETv3/HAMLET_summary_AI.md`).
`HAMLETv3/HAMLET_normalized.tsv` is byte-identical to `output/HAMLET_normalized.tsv`.
The old prediction file kept only confidence >= 0.3 (5,212 runs); the new
one keeps all runs (median confidence 0.14, Bone marrow top-1 for 52%), which
is why exact concordance reads 23% now against 94% before. Report both with
the threshold stated; the threshold is the `[P50-P51]` result.
- [ ] Ask Arnaud for the script and export behind `sample_nsaf_matrix.parquet`.
  Established 2026-09-14: fraction-collapsed, human projects only, from a
  newer database state, and with values that differ from `nsaf_diann.parquet`
  on the same runs (top-1 agreement 37%). See `final_analyses/MLMarkerPenalty/README.md`.
- [ ] Reconcile his project-level cohort counts (424 human primary, 95 healthy,
  45 mixed, 134 MLMarker-compatible, filtered on `sample_source`) with the
  B5 counts (filtered on `material_type`); one rule for the paper.
- [ ] His commit also deleted the six root preprocessing scripts; restored
  from c9558661 and staged on 2026-09-14. Check with him that this was accidental.

Existing: `mlmarker_hamlet/script/Concordance_metrics_normalized.ipynb`,
executed. Current numbers (per run, vocabulary-filtered): 4,107 overlapping
runs, 93.6% exact, kappa 0.94, mean Lin 0.97, 28 of 233 projects
heterogeneous, confidence Mann-Whitney p = 2.9e-16. Its markdown synthesis
cell (cell 45) is stale; quote executed cells only.

- [ ] **Circularity** `[P39-P42]`: 139 of the 183 MLMarker training PXDs
  (`MLMarker/Training_PXDs/SupplementaryTableS1.tsv`) are in the 1,737
  corpus, 44 of the 233 predicted projects. Report every concordance figure
  separately for seen and held-out projects.
- [ ] **Vocabulary** `[P43-P44]`: rerun with `FILTER_BY_MLM_VOCAB = False`
  (926 runs currently dropped). Split discordant pairs into out-of-vocabulary
  vs in-vocabulary; within out-of-vocabulary separate deliberately excluded
  (biofluids, cell types) from simply missing classes (breast, cervix, skin).
  The missing-class list is a result: candidate new classes.
- [ ] **Material context** `[P45]`: stratify by `material_type` before
  scoring; predict disagreement concentrates in cultured material.
- [ ] **Per-sample** `[P35-P38]`: collapse fractions with
  `MLMarker/Reprocessing_database/parition_combiner/fraction_groups.tsv`
  (8,709 runs to 831 samples, 157 PXDs) and recompute granularity and
  concordance. Every current number is per run.
- [ ] **Confidence** `[P50-P51]`: held-out only; find the threshold above
  which a propagated label can be accepted. Calibration bins already computed
  (0.60-0.70 bin 98.9% exact, top bin non-monotone at 94.1%).
- [ ] Move the notebooks into `final_analyses/MLMarkerConcordance/` with
  repository-relative paths; all five currently hard-code
  `C:\Users\jung.arnaud\...` and one reads a file outside the repo.

### B7. Extended training set (Tine, M-L) `[P52-P58]`, `[P63-P65]`

Outline, first-pass counts (38 direct, 17 mixed, 84 diseased-only, 15
unknown; 2 realistic new classes: skin, breast) and the eligibility rule are
in `docs/extended_training_set_outline.md`.

- [ ] Apply the winning eligibility criteria (decision 0) to the 1,737
  datasets minus the 183 training PXDs; count qualifying projects and
  samples after fraction collapse; split by existing vs new classes.
- [ ] Count projects qualifying directly vs needing manual run separation;
  do the manual separation for the tractable ones and record samples
  recovered.
- [ ] Coverage gain vs record: runs carrying a tissue label before and after
  manuscript-derived annotation, the four MLMarker selection fields, and the
  technical fields that lose coverage.

### B8. Retraining (in scope, last; Tine with Sander) `[P59-P62]`

Feasibility in `docs/extended_training_set_outline.md` section 3. Software
is present (`../MLMarker` package, atlas, training notebook, RF trains in
minutes here). Blocking input: the full reprocessing quant matrix
(`nsaf_diann.parquet`, 89,626 runs) is on Sander's machine or the cluster;
the local `~/git/Tissue_prediction_dev/repro_db/` view covers only 336
projects.

- [ ] Get the quant matrix and a matching `run_metadata_combined.tsv`.
- [ ] Settle design: rebuild the atlas from the reprocessing database with
  HAMLET labels (recommended) vs append to the spectral-count atlas.
- [ ] Fix fraction aggregation: sum intensities across fractions, then
  normalise once.
- [ ] Train baseline, extended, and project-level-label control; report
  macro-F1, per-class recall for new classes, holdout confusion.

---

## C. Writing (Tine)

### C1. Results

- [ ] R3 staircase: rewrite with SapBERT numbers and the S5 cost (A above).
- [ ] R4 six-model: one paragraph exists; add per-class detail and the
  Llama-4 Scout experimental-design gap from the panel values.
- [ ] R5 record vs article: fill from B4; keep the three-question structure
  (ceiling, coverage, agreement, residual).
- [ ] R6 healthy corpus and MLMarker: fill `[XXX]` from B5-B7, using the
  prose skeleton in `docs/section_mlmarker_draft.md`.
- [ ] "What text-derived annotation cannot resolve" paragraph (two case
  datasets) from `docs/section_mlmarker_draft.md`.

### C2. Materials and Methods, missing subsections

- [ ] Human annotation corpus and inter-annotator analysis: lift from
  `final_analyses/HumanAnnotation/METHODS.md` (13 identities, 42-label
  filter, SapBERT clustering, decomposition, rater-level permutation test).
- [ ] Development and test sets: 137 SDRF-annotated PXDs, 107/30 split,
  ground truth in `agentic-benchmark/benchmark_data/SDRFS_github/`.
- [ ] Staircase design: arms S0-S5, frozen prompts with SHA-256 manifest,
  Title+Abstract+Methods restriction, temperature 0, one inference run
  scored. Source: `final_analyses/SapBERTStaircase/README.md`,
  `frozen_prompts/`, `arms/`.
- [ ] SDRF benchmark and matcher: five tiers and scores, semantic threshold
  0.70, metric threshold 0.50, macro-average over agents, SapBERT replacing
  SciBERT with the 88,226-pair negative control (18.2% vs 0.00% above 0.70).
  Source: `final_analyses/SapBERTStaircase/README.md`, `figures/provenance.json`,
  `framework/benchmark/semantic_matcher.py` docstring.
- [ ] Six-model benchmark settings (from Hari) if kept.
- [ ] Model runtime for Gemma and GLM: `framework/docs/gemma4_runtime.md`,
  `framework/configs/`. The current Methods only describe Qwen.
- [ ] Confidence score and re-extraction loop (0.3 format, 0.5 evidence, 0.2
  completeness; retrigger below 0.6): `framework/docs/pipeline_architecture.md`.
  Check this matches what the production run actually used.
- [ ] MLMarker application: reprocessing database, run-level prediction,
  fraction grouping, training-set provenance, concordance scoring, metric
  calibration. Sources: `MLMarker/Reprocessing_database/README.md`, B3, B6.
- [ ] Statistics paragraph: Wilcoxon, Kruskal-Wallis, exact rater permutation
  (220 assignments), Spearman, Mann-Whitney, sample SD with ddof=1, Student-t
  CIs.

### C3. Front and back matter

- [ ] Abstract.
- [ ] Discussion: M1 difficulty floor and reproducibility vs accuracy; M2
  structure over model, with the S5 normalization cost and what
  normalization buys downstream; M3 grounding and QC; record-vs-article
  implications for repository practice; MLMarker as expression-based
  resolution of anatomical but not phenotypic ambiguity; ontology vs
  proteome divergence bounding agreement scores; open-access ceiling; limits
  (per-file mapping needs raw data, 30-PXD test set size, three model pairs).
- [ ] Figure legends 1-4; supplementary figures and tables (per-field
  tables, six-model table, eight-metric scorecard, HPA benchmark, unknown
  rate by field).
- [ ] References: replace every `[ref]`, `[FAIR]`, `[PX]`, `[SDRF]`,
  `[lesSDRF]`, `[SciBite]`, `[SPIRES]`, `[MLMarker]`, `[REF]`.
- [ ] Title, author list, contributions, data and code availability (Zenodo
  for the 1,737 annotations and the EuBIC corpus), funding.

---

## D. Repository hygiene (Tine, S-M)

- [ ] Root `README.md` describes `corpus/`, `benchmark/`, `outline.txt`,
  `docs/Metadata_paper_v2.1-2.pdf`; none exist. Describe `final_analyses/`,
  `EuBIC-annotation/`, `agentic-benchmark/`, `MLMarker/`, `mlmarker_hamlet/`.
- [ ] `final_analyses/README.md`: Figure 3 and 4 rows once B2/B4 land.
- [ ] `docs/section_mlmarker_draft.md` lines 261-263 still say tier 5 is
  SciBERT; correct or retire the file once its content is in the docx.
- [ ] Large duplicates: `final_annotations.zip` (11 MB, same content as
  `mlmarker_hamlet/input/...1737_validated/`), `mlmarker_hamlet/input/data/hpa_cache/normal_tissue.tsv`
  (83 MB) plus its zip, duplicate `uberon-basic.obo` / `uberon_basic.obo`.
- [ ] Two divergent copies each of `agent_metadata.tsv` and
  `run_metadata_combined.tsv` (`MLMarker/Reprocessing_database/results/` vs
  `mlmarker_hamlet/input/agentic-metadata/`); record which is canonical.
- [ ] `plot_style.py` references `PLOT_STYLE_GUIDE.md`, which lives in
  `../HAMLET`, not here.

---

## Time estimate

Focused working days. Hari's three blocks are counted as calendar waiting
time plus integration, not as work on this side.

| Block | Owner | Days |
|---|---|---|
| A. Fix stale numbers, rewrite R3 | Tine | 1 |
| B1/B2/B4 checks, missing figure files, R5 text from Hari's tables | Tine | 2-3 |
| B3 matcher calibration | Arnaud | 2-3 |
| B4 residual adjudication (~100 pairs) | Tine | 1-2 |
| B5 corpus counts | Tine | 2 |
| B6 concordance reruns | Arnaud | 3-4 |
| B7 extended training set incl. manual separation | Tine | 3-5 |
| B8 retraining (after quant matrix arrives) | Tine + Sander | 6-9 |
| C2 Methods gaps | Tine | 2-3 |
| C1 Results rewrite | Tine | 3-4 |
| C3 Abstract, Discussion, legends, refs, supplement | Tine | 4-5 |
| D hygiene | Tine | 1 |
| **Tine total** | | **25-35 days, about 6-7 weeks** |
| **Arnaud total** | | 5-7 days |
| **Calendar** | | 3 months: SapBERT re-score and Figure 4 files from Hari gate R3-R5; quant matrix from Sander gates B8; one co-author review round after the full draft |

Critical path: matcher decision -> A -> R3/R4; Figure 4 files and discordant
pairs from Hari -> residual adjudication -> R5 -> Discussion. B5-B7 and C2 can
proceed now.

---

## Session log

- 2026-09-14: repo read, draft audited, this file created. Staircase,
  six-model benchmark and record-vs-article are Hari's and already in the
  draft; their source tables are not in this repo except the staircase. Draft
  F1 numbers are SciBERT-era while the committed staircase re-score is SapBERT
  (decision 0). Figures 4/5 in the docx are broken links. MLMarker concordance
  exists but needs the circularity, vocabulary, material and per-sample splits.
  Docx comments (5, all Tine): check whether prompt optimization included
  validation; "lost in the middle" citation; move a passage to supplement;
  one "not needed"; "this is only on the manuscript data right?".
- 2026-09-14 (2): decisions taken: SapBERT everywhere, retraining in scope
  as the last part, record-vs-article standalone. Section A now lists every
  number to replace, taken from result files. Extended training set outlined
  in `docs/extended_training_set_outline.md`; retraining is feasible here
  once the full quant matrix arrives (local repro_db view has 336 of the
  1,719 projects). Figure 1 corrections: technical gain +0.11 not +0.13,
  model-model value kappa 0.57 not 0.58, single expert 0.44, PSA 0.60.
- 2026-09-14 (3): quant matrix received and unzipped (NSAF and FlashLFQ,
  89,626 runs). `final_analyses/ExtendedTrainingSet/` built: run labels
  without MLMarker, atlas rebuilt with class-aware project split (2,870
  samples, 24 classes, 64 projects; new evaluable classes skin, metencephalon,
  PBMCs, breast). Arnaud's pull reviewed (see B6). Cleanup script for
  `MLMarker/Reprocessing_database/` written, not run.
- 2026-09-14 (4): first retraining results in
  `docs/extended_training_set_outline.md` section 5. Published MLMarker on
  unseen projects: macro-F1 0.33; rebuilt atlas LOPO 0.48 (run-level labels)
  vs 0.37 (project-level control). Failures trace to material context
  (cultured cells, perfusates, sperm) rather than the classifier.
- 2026-09-14 (5): three retraining rounds done, table in the outline
  section 5. Honest result: 21 healthy solid-tissue projects after material
  filtering, LOPO macro-F1 0.54 (run-level) vs 0.38 (project-level) at >= 2
  projects per class. PXD071075 (1,001 "brain" runs, source "cell culture")
  needs a manual look. Ranked mixed-phenotype project list written for the
  manual separation step.
- 2026-09-14 (6): penalty_factor test. Arnaud's run used raw `predict_proba`
  (no penalty; SHAP path with penalty 0 reproduces it 100%). With penalty 1
  on a 360-run pilot oversampling Bone-marrow calls: Bone marrow top-1 drops
  from 42% to 0.3%, but exact agreement with HAMLET rises only 27% -> 30%;
  of 152 Bone-marrow calls 13% become correct, the rest scatter to Ovary,
  Kidney, Testis, Prostate, and true bone marrow recall falls 28% -> 0%.
  The penalty removes the default class, not the uncertainty; keep the
  confidence threshold as the acceptance rule. Files:
  `final_analyses/ExtendedTrainingSet/scripts/mlmarker_penalty_pilot.py`,
  `results/mlmarker_penalty_pilot_*`.
- 2026-09-14 (7): full MLMarker rerun with coverage rule (penalty 1 below 10%
  feature coverage) and Arnaud's notebook re-executed:
  `final_analyses/MLMarkerPenalty/`. Penalty removes Bone marrow (52% -> 11%
  of top-1 overall) but shifts accuracy by ~2 points. Healthy-native exact
  15% vs Arnaud's 23%: the difference is the input matrix (62% of runs under
  10% coverage in the run-level file), and his healthy-vs-cancer Lin
  separation disappears on it. Decision needed: one agreed quant matrix
  (fraction-collapsed) before any MLMarker number goes in the paper.
- 2026-09-14 (8): Arnaud's `sample_nsaf_matrix.parquet` downloaded from the
  CNRS share (reproduces his predictions 100%). Coverage-rule rerun and his
  notebook re-executed on it: `final_analyses/MLMarkerPenalty/results/`.
  Exact agreement unchanged (23.3% -> 23.9%); "exact + near" 59% -> 39% and
  healthy-vs-cancer Lin gap 0.63/0.43 -> 0.53/0.43, because Bone marrow
  default calls on blood samples had counted as near-agreement. Below 10%
  coverage the model is uninformative either way (4-5% exact). Proposal for
  the paper: report the coverage >= 0.10 group (56.6% exact, 87% in vocab,
  93.5% at confidence >= 0.3) and the share of unclassifiable samples.
- 2026-09-14 (9): biofluid rule added to Arnaud's notebook (biofluid-only
  samples not assessable, excluded from denominators). Healthy native on his
  matrix: 1,729 assessable runs, 43.4% exact, 79.7% at coverage >= 0.10,
  96.8% in vocabulary at confidence >= 0.3. Table in
  `final_analyses/MLMarkerPenalty/README.md`. These are the numbers to carry
  into `[P46-P51]`, with the biofluid and low-coverage shares stated.
- 2026-09-14 (10): `final_analyses/MLMarkerPenalty/` collapsed to one final
  result set (Arnaud's matrix, penalty 1 below 10% coverage, biofluids not
  assessable). Intermediate reruns moved to the session scratchpad, not kept.
