# Extending the MLMarker training set from HAMLET annotations: outline

Written 2026-09-14 from a read-only pass over the repo and its sibling
repositories. Numbers below are first-pass counts from
`mlmarker_hamlet/output/HAMLET_normalized.tsv` and are meant to size the
work, not to be quoted. Fills `[P52-P62]` of `docs/section_mlmarker_draft.md`.

## 1. What MLMarker is trained on today

| Item | Value | Source |
|---|---|---|
| Training projects | 183 PXDs | `MLMarker/Training_PXDs/SupplementaryTableS1.tsv` |
| Training atlas | 1,394 samples x 5,979 protein features, columns `cell_type, tissue_name, disease_status` + UniProt NSAF | `../MLMarker/data/atlas_TP_20250113.csv` |
| Classes (quant model) | 34 tissues | `../MLMarker/models/TP_RF_quant_20260418_label_encoder.joblib` |
| Quantity | NSAF from ionbot spectral counts, healthy only, biofluids excluded | `../Tissue_prediction_manuscript/Build_atlases.ipynb`, `NSAF.py` |
| Training recipe | Random Forest (also XGBoost and a small NN), per-class 80/20 split plus 10% holdout, missingness filter at mean +/- 1 SD | `../MLMarker/tissue_predictor_training.ipynb` |
| Package | `mlmarker` 0.1.5 installed for miniconda python 3.12 (import currently fails, needs a reinstall from `../MLMarker`); xgboost 3.2, sklearn 1.7.1, torch 2.10 present | pip |

Inclusion rule of the existing atlas: human, physiological (healthy) sample,
solid tissue or purified cell type, no biofluid, no cell line. This is the
rule the extension must apply to be consistent.

## 2. Candidate datasets in the 1,737 corpus (first pass)

Filter chain on HAMLET annotations, project level, human only, excluding the
139 training PXDs that sit inside the corpus:

| Step | Datasets |
|---|---|
| Human, not in training set | 1,286 |
| Tissue annotated | 913 |
| No cell line | 425 |
| Material type tissue or primary cells | 154 |

The 154 split by disease state and tissue count:

| Disease state | Single tissue | Multi-tissue | Runs on cluster table |
|---|---|---|---|
| Healthy only | 34 | 4 | 3,609 (36 PXDs) |
| Healthy and diseased mixed | 11 | 6 | 1,156 (15 PXDs) |
| Diseased only | 63 | 21 | 11,075 |
| Disease state unknown | 11 | 4 | 277 |

Tiers for the manuscript:

- **Tier 1, direct** `[P56]`: 38 healthy-only projects. 34 single-tissue
  qualify from the annotation alone; 4 multi-tissue need MLMarker or run-name
  resolution to assign runs to tissues.
- **Tier 2, manual separation** `[P57-P58]`: 17 mixed projects, where the
  healthy runs must be separated from diseased ones by run name, SDRF, or
  MLMarker prediction. This is the tier that quantifies "what still needs a
  person".
- **Tier 3, excluded**: 84 diseased-only, 15 unknown. Report as the reason
  most of the corpus cannot feed a healthy atlas.

Of the 55 tier 1 and 2 tissue terms, 32 fall inside the 34 existing classes
(brain 6, lung 4, liver 3, placenta 3, heart 3, and so on) and 23 do not.
Among the latter, blood (13) is a biofluid and stays excluded by design; skin
(5), breast/mammary gland (5), omentum and subcutaneous fat, uterus, nerve,
vessels (umbilical vein and artery, carotid, internal mammary artery) are
candidate **new classes** `[P54-P55]`. Skin and breast are the only ones with
more than one project, so realistically two new classes, each thin.

Candidate lists are in the scratchpad only; the script to regenerate them
belongs in `final_analyses/ExtendedTrainingSet/scripts/select_candidates.py`.

Refinements before the counts are quotable:

- Use the ontology identifiers, not lower-cased labels, and collapse
  synonyms (muscle vs skeletal muscle, heart vs cardiac chambers).
- `material_type` is multi-valued; a project tagged both `tissue` and
  `cell line` is currently kept if any value is tissue. Decide.
- Cross-check tier 1 against `disease_state` evidence sentences; "normal"
  often means adjacent normal tissue in a tumour study.
- Collapse fractions with `MLMarker/Reprocessing_database/parition_combiner/fraction_groups.tsv`
  before counting samples `[P53]`.

### Run-level counts (build_run_labels.py, 2026-09-14)

Labels per run with priority SDRF > run name > HAMLET, MLMarker excluded.
Of 85,897 runs in 1,719 projects:

| Set | Runs | Projects |
|---|---|---|
| Human, healthy, tissue known, no cell line | 7,571 | 133 |
| of which in old-atlas projects (circularity) | 850 | 20 |
| Human, tissue ambiguous (multi-tissue project, run name silent) | 3,097 | 53 |
| Human, tissue known but disease mixed (needs run separation) | 7,523 | 99 |

Candidate new classes among the healthy runs after biofluid and cultured
exclusion, with at least 10 samples: zone of skin (253 runs, 6 projects),
metencephalon (189, 2), PBMCs (148, 7), cervix (99, 1), peritoneum (75, 1),
breast (46, 5), enamel (35, 1), vocal fold (27, 1), nerve (14, 1), pancreas
(13, 2), ureter (12, 1). Only those with two or more projects can be
evaluated on a project-disjoint split; single-project classes can be trained
on but not scored, and the manuscript should say so.

## 3. Can we retrain here?

**Software**: yes. The package source, models, atlas and training notebook
are in `../MLMarker`; Random Forest training on ~1,400 samples x 6,000
features takes minutes on this machine (256 cores, 1 TB RAM, no GPU).

**Quantification for the new samples**: this is the constraint.

| Source | Coverage | Quantity | Status |
|---|---|---|---|
| `~/git/Tissue_prediction_dev/repro_db/run_protein_quant_view.parquet` | 336 projects, 11,429 runs, all inside the 1,737 | protein-group intensity (DIA-NN/Sage reprocessing), not spectral counts | present locally |
| `nsaf_diann.parquet` / `pride_quant.parquet` used by `finalize_dataset.ipynb` and `run_meta_mlmarker.tsv` | 89,626 runs, 1,719 projects | NSAF-like values derived from the reprocessing intensities | on Sander's machine or the cluster, not here |
| Existing atlas | 183 projects | ionbot spectral-count NSAF | present |

Only 7 of the 41 strict candidates have local quant; 36 of 38 tier 1 and 15
of 17 tier 2 projects have runs in the cluster run table. So retraining needs
the full quant matrix from the reprocessing database (ask Sander for
`nsaf_diann.parquet` plus `run_metadata_combined.tsv` of the same date, or
export it from the cluster DB).

**The domain-shift question must be settled before retraining.** The
existing atlas is spectral-count NSAF from ionbot; the new samples would be
intensity-based from the DIA-NN/Sage reprocessing. Two defensible designs:

1. **Extend the existing atlas** (what the manuscript text assumes): append
   new samples, retrain, report macro-F1 before and after and per-class recall
   for new classes `[P59-P61]`. Requires converting the reprocessing
   quantities to something comparable to the atlas (NSAF from counts is not
   available from DIA-NN; intensity-based NSAF would need protein lengths from
   `../MLMarker/uniprot_reviewed_lengths.tsv` and a validation that mixed
   quantities do not create a batch effect the classifier learns).
2. **Rebuild the atlas entirely from the reprocessing database** (what
   `finalize_dataset.ipynb` already does: 53,134 human runs after the 500-protein
   filter, 90/5/5 split by PXD). Then the HAMLET contribution is the label,
   not the sample, and the control experiment is "project-level labels only"
   vs "run-level resolved labels" `[P62]`. Cleaner, but it is a new model, not
   an extension, and the headline changes from "F1 rises from X to Y" to
   "run-level annotation adds Z over project-level annotation".

Recommendation: design 2. It avoids mixing two quantification regimes,
`finalize_dataset.ipynb` already implements most of it, and the control
experiment isolates exactly what this paper contributes. Fraction handling
must then be fixed as the draft notes: sum intensities across fractions
first, normalise once (the notebook sums per-run normalised NSAF, which
dilutes fraction-confined proteins).

## 4. Work plan

1. Get the quant matrix (Sander / cluster). Blocking.
2. `select_candidates.py`: tiered eligibility from `HAMLET_normalized.tsv`
   with ontology IDs; output tier tables and the new-class list. 1 day.
3. Manual separation of the 17 mixed projects using run names, cached SDRFs
   (`MLMarker/Reprocessing_database/agentic_metadata/cache/sdrf/`) and
   MLMarker predictions; record decisions per run. 2-3 days.
4. Build the extended atlas (design 2) with fraction collapse fixed; healthy
   filter from HAMLET labels; per-PXD split. 1-2 days.
5. Train RF with `tissue_predictor_training.ipynb` settings: baseline
   (existing atlas or project-level labels), extended, and the control.
   Report macro-F1, per-class recall for new classes, confusion on holdout.
   1-2 days.
6. Write `[P52-P62]` and the Methods subsection. 1 day.

Total about 6-9 working days once the quant matrix is available.

## 5. Retraining results (2026-09-14, `final_analyses/ExtendedTrainingSet/results/`)

Three rounds of the same pipeline with progressively stricter material
context. Evaluation is leave-one-project-out (LOPO) over all projects, inner
hyperparameter folds grouped by project, macro-F1 over classes that have at
least the stated number of projects. Random Forest, MLMarker paper settings.

| Round | Material rule | Samples | Classes | Projects | Evaluable classes | LOPO macro-F1 run-level labels | LOPO macro-F1 project-level labels (control) |
|---|---|---|---|---|---|---|---|
| 1 (`results/round1/`) | none beyond tissue name and cell-line field | 2,870 | 24 | 64 | 11 (>= 3 projects) | 0.48 | 0.37 |
| 2 (`results/round2/`) | any culture word in material, source or cell type excludes | 854 | 13 | 18 | 2 | 0.89 | 0.80 |
| 3 (current `results/`) | material_type must say tissue/primary cells/fibres/aspirate and not culture; sperm, fibroblast, stem cell, perfusate excluded | 1,607 | 14 | 21 | 2 (>= 3 projects) / 6 (>= 2) | 0.997 / **0.54** | 0.99 / 0.38 |

Round 3 per-class LOPO recall at >= 2 projects, run-level labels: PBMCs
1.00, liver 1.00, skeletal muscle 1.00, testis 0.97, breast 0.32, brain 0.00.
Brain is two projects: PXD053929 (3 samples) and PXD071075 (1,001 samples,
`material_type: tissue`, `sample_source: cell culture`), and no model,
including the published MLMarker (28% recall), calls PXD071075 brain. That
project needs a manual look before anything is concluded about brain.

Published MLMarker on the same round-3 samples, in-vocabulary classes:

| Subset | n | Projects | Accuracy | Macro-F1 | Accuracy at confidence >= 0.3 (share of runs) |
|---|---|---|---|---|---|
| its own training projects | 362 | 7 | 0.92 | 0.87 | 0.996 (74%) |
| projects it never saw | 1,127 | 7 | 0.34 | 0.76 | 0.91 (7%) |

Reading for the manuscript:
- Project-disjoint evaluation is the honest one. The published 98% comes
  from a random split; on unseen projects the same model is right on a third
  of runs and only its confidence threshold restores 91%, on 7% of runs. The
  HAMLET-vs-MLMarker concordance (B6) shows the same curve.
- Run-level labels beat project-level labels in every round (0.48 vs 0.37,
  0.54 vs 0.38) `[P62]`: SDRF and run-name separation is what the text
  pipeline contributes beyond a project label.
- After honest material filtering, the healthy solid-tissue extension in
  the 1,737 datasets is 21 projects. Most of the corpus is disease, cell
  culture or biofluid. Six classes reach two projects, two reach four. The
  paper can claim PBMCs and breast as new learnable classes only with that
  caveat; skin fell out entirely once cultured fibroblasts were excluded.
- Round 1's failures and Arnaud's audit name the same projects
  (cultured MSCs as bone marrow, perfusate as liver, sperm as testis,
  fibroblasts as skin). HAMLET's tissue label is right as text and wrong as
  a proteome; `material_type` is the field that separates them, and it must
  be used.

Next: manual check of PXD071075; per-run separation of the 99 mixed-phenotype
projects (`results/mixed_projects_ranked.tsv`) to grow project counts; then a
final round and the `[P59-P62]` numbers.
