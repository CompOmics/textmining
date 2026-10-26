# Debug report: `LLm_as_judge.py` (v3) and its results

Script: `llm_judge_results/LLm_as_judge.py` (3,551 lines, `PROMPT_VERSION = proteomics_sdrf_judge_v3`).
Results examined: `results_train/` (107 papers, 3,657 rows) and `results_test/` (30 papers, 1,057 rows).
Companion report on the input texts: `DEBUG_REPORT_llm_judge_input.md`.
Paper integration: `PAPER_SECTION_llm_judge.md`.

No judge call was made for this report. All numbers come from the existing result files and from reading the script.

## 0. Correction of the earlier premise: what was judged

The earlier report treated the judged SDRFs as HAMLET output. The files and the draft say otherwise.

| Evidence | Finding |
|---|---|
| `NLP_traintest/test_set/README.MD` | "test set PXDs with manuscript files and SDRFs directly from bigbio github human annotated" |
| Draft, Methods, "Benchmark ground truth and matching" | 107 development datasets with community SDRFs. Test ground truth comes from the SDRF (12 datasets) or from curated annotation files in the same schema (18 datasets). |
| `comment[tool metadata]` in the 30 test SDRFs | 18 contain `json_to_sdrf from <PXD>_annotation.json`. 12 are empty. |
| Train SDRFs | 106 have no tool column. 1 reads `lesSDRF v0.1.0`. Headers are legacy PRIDE style (`SourceName`, `OrganismPart`). |

So all 137 judged SDRFs are the **benchmark reference**, not HAMLET extractions:

| Origin | PXDs | Rows (benchmark fields only) |
|---|---:|---:|
| Community SDRF, train | 107 | 3,438 |
| Community SDRF, test | 12 | 444 |
| Curated JSON converted by `json_to_sdrf`, test | 18 | 570 |
| Total | 137 | 4,452 |

Consequences:

1. The run measures how well the **reference** is supported by the manuscript text. It does not measure HAMLET.
2. "Hallucination" is the wrong word for human-curated values. A curator may have used the PRIDE page, supplementary files, raw files or the authors. The correct label is "not supported by the supplied text".
3. The converter findings in section 10 of the input report describe `json_to_sdrf` on the curated JSON, not a HAMLET export. The name "HAMLET JSON" in that section should read "curated annotation JSON". Please confirm this reading, because it contradicts what you told me earlier.
4. The 1,737-dataset HAMLET output has not been judged at all.

Split of the 390 flags by origin (all fields):

| Origin | Flags | In manuscript.txt (A) | PRIDE text only (C) | Nowhere (D) | Bare number (u) |
|---|---:|---:|---:|---:|---:|
| Community | 343 | 14 | 39 | 271 | 19 |
| Curated JSON | 47 | 6 | 32 | 8 | 1 |

With the six excluded fields dropped: 363 flags (community 316, curated 47), 252 of them class D.

Interpretation:
- Curated JSON: 32 of 47 flags are values that exist only in the PRIDE descriptor. The curation used the descriptor. This is a genuine finding, because a manuscript-only HAMLET run cannot reach these values.
- Community: 244 of 316 are found nowhere in the paper or the PRIDE record. They are curator knowledge or sample sheet content (age, individual, phenotype, labels, modification lists, tolerances).

## 1. Is this the script that produced the results?

Yes. The result CSVs contain `GUARD:` lines that only this version writes: `sample-sheet-guard` (726 train, 102 test), `cell-line-guard` (93), `range-guard` (82), `runs-guard` (6), `unit-guard` (4), `mod-guard` (1), `mismatch-veto` (1), `regroup-guard` (1), `verdict-reconcile` (1). The match types `NOT_IN_TEXT` (901 train, 102 test) come from the sample-sheet guard (lines 2426, 2456). The HAMLET repository copy of `sdrf_judge.py` writes none of these.

Not shared, but written by this script: `llm_judge_annotation_review_full.csv` (with `checks`, `n_runs`, `filter_action`), `llm_judge_per_paper.csv`, `llm_judge_per_field.csv`, `llm_judge_coverage.csv`, `llm_judge_recovered.csv`. The shared `llm_judge_annotation_review.csv` has 12 columns and lacks these.

## 2. How the script works

1. `load_manuscript` reads `manuscript.txt` and drops repeated lines of 200 or more characters (line 3242).
2. `build_units` turns each SDRF into evaluation units. Kind `value` is one distinct cell of a column. Kind `modification` is one distinct modification string. Kind `runs` is a whole column mapping run to value (replicates, fractions, factor values, individual).
3. `unit_checks` adds deterministic hints (line 1877): whether the term occurs verbatim in the text, tolerance number printed, missing `MT`, `TA` or `PP`.
4. One chat call per unit through DeepEval GEval. Model: `google/gemma-4-31b-it` through OpenRouter, thinking on, temperature 0. The full paper text is resent for every unit and cached on disk by hash of prompt, paper and context.
5. `interpret_reason` parses the verdict and flags, then applies guards (line 1943).
6. `_sample_sheet_guard` rewrites some LOW or HALLUCINATED rows to `unverifiable` (line 2394).
7. `recover_missed_values` asks the judge what the SDRF omitted (output not in the share).
8. `apply_to_sdrf` writes the filtered SDRF: rejected cells become `not available`, MEDIUM cells are replaced by the judge's correction.

Match types: CORRECT (high), PARTIAL (medium), NO_MATCH (low, not hallucinated), HALLUCINATED, TYPE_MISMATCH, NOT_IN_TEXT (unverifiable).

## 3. Findings

Severity: **High** changes a number that would go into the paper. **Medium** distorts a sub-result. **Low** hygiene.

### F1 (High). Wrong construct for the reference set
See section 0. The "hallucination" rate of 8.3% (390 of 4,714) cannot be reported as an error rate of annotators or of HAMLET. It is the share of reference values the text does not support.

### F2 (High). Three different text inputs, one prompt wording
| Group | Manuscript content | PXDs |
|---|---|---:|
| Train | `ABSTRACT:` + `METHODS:` | 107 |
| Test, community | flat text, no headers | 12 |
| Test, curated JSON | TITLE, ABSTRACT, INTRO, METHODS, RESULTS, DISCUSS | 18 (median 46.7k characters against 14.5k for community) |

`_build_messages` labels every text "PAPER TEXT (abstract and methods)" (line 1296). The system prompt says "the introduction never describes the analysed samples" (lines 430 and 431), while 18 papers include the introduction and discussion. Pipeline inconsistency: the paper's HAMLET input is abstract plus methods (draft, Results, section 2).

Rates are not comparable across groups. Curated JSON papers have the longest text and the highest wrong-or-unsupported rate (21.9% against 14.2%), but they are also a different annotator, so the cause cannot be separated.

### F3 (High). Papers with no usable methods text
Ten papers have a thin manuscript (under 3,000 characters, or no digestion or instrument term): PXD002080, 081, 082, 084, 086, 087, 088 (identical 2,676-character files), PXD003539, PXD013737, and PXD034195 (61k characters but no instrument or digestion term; likely a structure problem). Over the technical fields (instrument, cleavage, fragmentation, tolerances, modification, label) the hallucination share is **68.8% (44 of 64)** in these papers against **10.0% (129 of 1,285)** elsewhere. These flags measure missing input text. They should be excluded or reported separately.

### F4 (High). The accuracy statistic penalises unverifiable rows and counts distinct cells, not runs
`_compute_single_paper_stats` computes `judge_accuracy = n_correct / total` (line 2902). `NOT_IN_TEXT` rows stay in `total`. 985 of 4,452 rows (22%) are `NOT_IN_TEXT`. The script also writes `judge_accuracy_verifiable`, but that CSV was not shared.

The unit is a distinct value, not a run or a sample. PXD006003 contributes 259 rows (220 `NOT_IN_TEXT`) and the median paper 27. The top 5 papers hold 48% of all `NOT_IN_TEXT` rows. Pooled row rates are therefore dominated by a few papers. Report per-paper medians or weight by paper.

As-run correct share among verifiable rows, benchmark fields only: community 76.5% (2,217 of 2,897), curated JSON 68.6% (391 of 570).

### F5 (High). Post-hoc guards change verdicts after the judge answers
- `sample-sheet-guard` turns a HALLUCINATED row into `unverifiable` when its column has at least 5 distinct values (`UNVERIFIABLE_MIN_DISTINCT`, line 59). For `sex` the threshold is 2. These are arbitrary cut-offs and they only fire when the judge already said HALLUCINATED. A judged-correct value in the same column is unaffected, so the two outcomes are asymmetric.
- 726 train rows are rewritten. This is why `unverifiable` is 25% of train rows and 10% of test rows.
- `range-guard` keeps an age inside a printed cohort range as unverifiable and sets one outside to LOW. Reasonable, but it only looks 120 characters after an "age" keyword.
- The modification guard (`_searched_mod_sites`, line 1078) needs the text pattern `Name (site)`. It fired **once in 4,714 rows**. It is inert for papers that write "carbamidomethylation of cysteine".

### F6 (Medium). The `checks` hints and the completeness rule drive PARTIAL on modifications
For modifications the script tells the judge "no TA residues or PP position" and "no MT=Fixed or MT=Variable" (lines 1899 and 1901). The prompt defines a complete modification cell as one with residue or position and MT (prompt, COMPLETENESS CHECK). Test modification rows: 17 CORRECT, 27 PARTIAL, 13 HALLUCINATED, 1 TYPE_MISMATCH, none NOT_IN_TEXT. PARTIAL is therefore partly a function of the hint, not of the paper. It is still a fair finding about the reference (section 10 of the input report: the paper states residue for 32 of 36 entries, the curated JSON for 17), but it should be reported as "incomplete against the SDRF template", not as error.

### F7 (Medium). Evaluation items leak into the prompt
Worked examples 3, 4, 5, 7 and 8 use PXD000070 (Plasmodium schizont, erythrocytes, Ammonia-loss, ETD decision tree runs, DT aliquots). PXD000070 is in the train set. Its 22 rows are therefore not independent: 8 CORRECT, 10 PARTIAL, 2 HALLUCINATED, 1 TYPE_MISMATCH, 1 NO_MATCH. Exclude it from any reported statistic or replace the examples.

### F8 (Medium). Generic and identifier columns
Unknown columns fall into the generic field `other_column` and are judged against a generic definition. Examples: `sampleChecksum` (PXD002619, 58 rows, all NOT_IN_TEXT), `overall survival`, `progression free survival` (88 and 79 train rows, NOT_IN_TEXT), `heart weight`, `characteristics[phenotype]` (22 hallucination flags). `_IDENTIFIER_KEYS` skips file and accession columns but not checksums or per-sample clinical variables. Run-file strings such as `file.raw = 1 | ...` make up 404 rows.

### F9 (Medium). Corrections are applied without verification
MEDIUM verdicts replace the SDRF cell by the judge's `corrected_value`. Only tolerance fields are checked against the paper (`number-guard`, line 2075). Known wrong changes in `filtered_sdrf`: `300 nl minutes-1` became `300 µL∙min−1` (PXD005982), and `three biological replicates` became `1`. Any filtered SDRF should not be used as a corrected reference without a verbatim-support check on every corrected value.

### F10 (Medium). Judge validity is untested
- One judge model, one run per unit. Temperature 0 with a disk cache replays the same answer, so the cache hides run-to-run variance. No repeat or second-judge agreement exists.
- No human audit of judge decisions exists. Without one, "judge flagged X%" is a claim about Gemma, not about the data.
- Gemma 4 31B is also a benchmarked extractor in the draft. If the judge is later applied to HAMLET output, a same-family judge can favour its own style. Use a different judge family for HAMLET output, or report both.
- The judge sees the SDRF value and a name for the field, never the model that wrote it. That part is correct and should be kept.

### F11 (Low). Reproducibility
Absolute paths (`/Users/fateme/...`, lines 34 and 37). Credentials are read from `OPENROUTER_API_KEY`, which is correct. Papers go to a third-party API. That is acceptable for published open-access papers, but not for unpublished manuscripts. The shared results lack the files in section 1.

### F12 (Low). My own helper file
`all_rows_support_check.csv` carried the label "in judge-visible text (TITLE/ABSTRACT/METHODS)" from an early wrong assumption. I corrected the label to "in manuscript.txt". A new file `all_rows_support_check_benchmark_fields.csv` has the six excluded fields removed and an `origin` column.

## 4. Suggested improvements, in priority order

**Before any number enters the paper**
1. State what the run measures: text support of the reference (section 0). Rename HALLUCINATED to `UNSUPPORTED` in the output and in all figures.
2. Fix the input: one manuscript definition for all papers, equal to the HAMLET input (abstract plus methods). Rebuild the 18 curated-JSON manuscripts to abstract plus methods, and fix the label and the system prompt sentence.
3. Mark papers without methods text (F3) and report them separately.
4. Report accuracy over verifiable rows, per paper, with the 6 excluded fields removed. Add run-weighted numbers as a second view.
5. Drop PXD000070 or replace the worked examples with synthetic ones (F7).
6. Skip checksum, survival and other non-protocol columns, or judge them under a separate "sample sheet" class.

**Make the judge defensible**
7. Human audit: draw about 200 rows stratified by match type and origin, label them blind, and report agreement (Cohen's kappa) with the judge. This is the single step that makes a judge usable in a paper.
8. Run a second judge of another family and report agreement. Repeat one run without the cache.
9. Replace the arbitrary distinct-count rule (F5) by an explicit class: "per-sample value, not expected in the paper". Decide that class from the field, not from the judge answer.
10. Apply a verbatim-support check to every `corrected_value` (F9) and keep the unfixed original when it fails.
11. Split PARTIAL for modifications into "missing TA", "missing PP" and "missing MT", or drop the completeness rule from the paper-grounding score and report it as a template check.

**Extend the run to what the paper claims**
12. Judge the HAMLET output itself: the 30 test PXDs for each of the six models (`six_model_benchmark/<model>/reports_test_v2_*_nopride`), and the public set in `bigbio_pre2026_test_set/` ([N] PXDs, see the manifest). For the 1,736-dataset corpus draw a stratified sample, not all 1,736.
13. Add a source-aware column for HAMLET output: manuscript only, PRIDE only, combined. For the PRIDE and combined arms the judge must see the descriptor, otherwise the flags repeat the finding in section 3 of the input report. Keep the manuscript-only arm as designed.
14. Save the full CSV set and the cache directory with the results.

## 5. Open points that need your input

1. Confirm that the 18 curated JSON files are the curated reference (draft wording) and not HAMLET extractions. This changes the reading of section 10 of the input report.
2. Was `json_to_sdrf` written by you or the student? I could not find its code in either repository.
3. Which judge model do you want for HAMLET output, given that Gemma is both extractor and judge here?
