# HAMLET vs MLMarker concordance: penalised low-coverage scoring, biofluids not assessable

Arnaud's analysis (`mlmarker_hamlet/script/MLMarker_v_HAMLET.ipynb`, commit
8ba4885f) redone with two rules, on his own input matrix. This folder holds
the final version only; earlier iterations were kept in the session
scratchpad and are described in the to-do log.

## Rules

1. **Coverage-dependent penalty.** Coverage = share of MLMarker's 5,979
   features detected in a sample. Coverage >= 0.10: Random Forest
   `predict_proba`, identical to Arnaud's scoring. Coverage < 0.10: the
   package's SHAP prediction path with `penalty_factor = 1`, which subtracts
   the contribution of absent proteins and removes the Bone marrow default
   those samples otherwise receive. `scoring_mode` and `coverage` are carried
   in the prediction file. SHAP-path scores are base value plus SHAP sum, not
   calibrated probabilities.
2. **Biofluids not assessable.** MLMarker was trained on solid tissues and
   purified cell types. Biofluid terms are dropped from the HAMLET side before
   matching; a sample whose only tissue is a biofluid is tiered
   "Not assessable (biofluid)" and excluded from every concordance
   denominator. Ontology proximity of blood to bone marrow therefore no
   longer counts as near-agreement. Gland, vesicle and bladder organs are
   exempt from the pattern.

## Input

`MLMarker/Reprocessing_database/agentic_metadata/data/sample_nsaf_matrix.parquet`
(1.3 GB, gitignored, from Arnaud's CNRS share): 62,329 fraction-collapsed
samples, intensity-based NSAF, human projects. Scoring it with raw
`predict_proba` reproduces Arnaud's `run_meta_mlmarker_all.tsv` on 100% of
rows. Model: package default `TP_full_quant_update_20250113` (34 classes).

## Layout

```
scripts/predict_mlmarker_coverage.py   sharded prediction; needs a venv with shap >= 0.46 (miniconda's shap is NumPy-1 only)
scripts/merge_shards.py                -> results/run_meta_mlmarker_all_penalty.tsv and the confidence >= 0.3 file
scripts/MLMarker_v_HAMLET_penalty.ipynb Arnaud's notebook, repository paths, rule 2 added in the evaluation cell
results/run_meta_mlmarker_all_penalty.tsv      62,329 rows, gitignored
results/run_meta_mlmarker_penalty.tsv          confidence >= 0.3 subset
results/MLMarker_v_HAMLET/                     every table and figure the notebook writes
results/MLMarker_v_HAMLET_penalty_executed.ipynb
```

Reproduce:
```
for i in $(seq 0 31); do venv/bin/python scripts/predict_mlmarker_coverage.py --shard $i --nshards 32 --no-dedupe & done; wait
python3 scripts/merge_shards.py results
jupyter nbconvert --to notebook --execute scripts/MLMarker_v_HAMLET_penalty.ipynb --output-dir results --output MLMarker_v_HAMLET_penalty_executed.ipynb
```

## Results

Predictions: 32,018 samples scored by `predict_proba`, 30,311 by the penalty
path. Bone marrow top-1: 33.5% and 1.2% respectively, 17.8% overall (Arnaud's
run: 52%).

Healthy native human cohort (183 training projects excluded), 3,135 runs of
which 1,406 are biofluid-only (blood 1,180, saliva 101, dental plaque 75,
plasma 20, amniotic fluid 14, semen 11):

| | All assessable | Coverage >= 0.10 | Coverage < 0.10, penalty 1 |
|---|---|---|---|
| Runs / projects | 1,729 / 36 | 823 | 906 |
| Exact | 43.4% | 79.7% | 10.4% |
| Exact + near | 49.9% | 84.8% | 18.2% |
| Mean Lin similarity | 0.63 | 0.90 | 0.39 |
| In vocabulary, exact | 55.2% (n=1,298) | 87.1% (n=753) | 11.0% (n=545) |
| In vocabulary, confidence >= 0.3 | | 96.8% (561 runs, 75%) | |

Cancer / diseased cohort: 15,115 runs, 1,261 biofluid-only excluded, 13,854
assessable, 8.6% exact, 18.3% exact + near, mean Lin 0.40. Healthy-vs-cancer
Lin gap 0.63 vs 0.40 (Mann-Whitney in the executed notebook).

How the same healthy-native cohort moved under the two rules:

| Analysis | Runs | Exact | Exact + near | Mean Lin |
|---|---|---|---|---|
| Arnaud, raw `predict_proba`, biofluids scored | 3,135 | 23.3% | 58.9% | 0.63 |
| + penalty 1 below 10% coverage | 3,135 | 23.9% | 38.6% | 0.53 |
| + biofluids not assessable | 1,729 | 43.4% | 49.9% | 0.63 |

Reading: the penalty removes the default class and changes accuracy by a
point or two; below 10% coverage the model has no usable signal and those
samples (48.6% of the matrix) should be reported as unclassifiable. The drop
in "exact + near" under the penalty is the removal of blood-as-bone-marrow
calls that the ontology tiers had scored as near-agreement; the biofluid rule
then takes those samples out of the denominator altogether. Quote the
coverage >= 0.10 group and the confidence threshold for `[P46-P51]`, with the
biofluid and low-coverage shares stated.

## Run-level granularity and healthy vs diseased (2026-09-14)

`scripts/granularity_and_disease.py` -> `results/granularity/` (tables,
`summary.md`, `figure_granularity_disease.{png,svg}` and per-panel files).
Multi-tissue projects: MLMarker's top-1 lands in HAMLET's tissue list for
63 % of usable runs (chance 26 %); against SDRF / run-name truth (2 projects,
41 runs) the prediction restricted to HAMLET's list is right 78 % of the time
vs 34 % unrestricted. Healthy vs diseased: no separation within the one
project with run-level labels and none across projects once material and
vocabulary are controlled.
