# Debug report: input text and conversion errors in the LLM-as-a-judge run

> **Correction (2026-10-26).** The judged SDRFs are the benchmark reference (119 community SDRFs and 18 curated annotation JSON files converted by `json_to_sdrf`), not HAMLET output. See `DEBUG_REPORT_judge_script.md`, section 0. Where this report says "HAMLET JSON" or "HAMLET annotation", read "curated annotation JSON". The measured numbers are unchanged. The 18-PXD conversion analysis (section 10) applies to the curated reference only.


Scope: `llm_judge_results/` (train set 107 PXDs, test set 30 PXDs).
Date of analysis: 2026-10-26.
All paths are relative to `/home/tine/git/Publication/textmining/llm_judge_results/` unless stated otherwise.

## 1. Summary

Design (confirmed by the group): the judge compares an SDRF to the manuscript text only. It must never see prior descriptors (the PRIDE project metadata). The SDRFs are produced from HAMLET annotations (`*_annotation.json`), and the extraction model did see the PRIDE descriptors.

1. The judge scored the values of the SDRF files (601 of 604 judged values for the 18 test PXDs that have an annotation JSON are verbatim SDRF cells). It did not score `*_annotation.json` directly.
2. The judge saw `manuscript.txt` only, as designed. The text that the extraction model saw (field `text` in `*_annotation.json`) is longer. The difference is exactly the three PRIDE descriptor sections (`projectDescription`, `sampleProcessingProtocol`, `dataProcessingProtocol`). The paper part is identical to `manuscript.txt` for all 18 test PXDs with a JSON. So the character difference is the descriptor text that was removed on purpose.
3. A flag on a value that exists only in the descriptor is therefore a correct flag, not a judge input error. It shows that HAMLET took the value from the descriptor, not from the paper. This applies to 71 of 390 HALLUCINATED rows (18%). Section 3 and 4 explain how to read these.
4. The JSON to SDRF conversion is almost lossless (section 10): no field that HAMLET extracted is missing in the SDRF, and 357 of 424 field values are copied verbatim. The 67 changed values are normalisations (instrument, cleavage agent, organism, label, replicate, modification).
5. Modifications are the exception that matters. The judge's complaints about residue (`TA`), position (`PP`) and fixed/variable (`MT`) are mostly fair. The manuscript text mentions a residue within 120 characters of the modification name for 32 of 36 modification entries and fixed/variable status for 28 of 36 (loose window, so these are upper bounds). HAMLET recorded the residue for 17 entries and the status for 6. The converter dropped the residue in 3 cases and turned one misspelled name into the wrong modification (section 10).
6. Of 390 HALLUCINATED rows (test 69, train 321), 279 (72%) are not found in the manuscript or the PRIDE metadata by string matching, 71 (18%) are only in the descriptor, 20 (5%) are in `manuscript.txt` (judge error) and 20 (5%) are bare numbers that cannot be checked. Of the 279 unsupported rows, 41 are converter artefacts (run-file to value strings) and 27 belong to excluded fields.
7. Other conversion steps around the judge also introduce errors. Section 5 lists four.
8. The judge script in `HAMLET/src/python/sdrf_judge.py` is not the version that produced these results. Section 6 shows the evidence.

## 2. What was given to the judge

| Folder | Content per PXD | PXDs |
|---|---|---:|
| `NLP_traintest/train_set/<PXD>/` | `manuscript.txt`, `<PXD>.sdrf.tsv` | 107 |
| `NLP_traintest/test_set/<PXD>/` | `manuscript.txt`, `<PXD>.sdrf.tsv` | 12 |
| `NLP_traintest/test_set/<PXD>/` | `manuscript.txt`, `<PXD>_annotation.json` | 18 |

The README in both folders says that the SDRFs come directly from the bigbio GitHub repository and are human annotated. I could not verify this independently.

The judge output rows (`results_*/llm_judge_annotation_review.csv`) hold the SDRF values. For the 18 PXDs with a JSON:

| Judged value is found verbatim in | Rows (of 604) |
|---|---:|
| the SDRF | 601 |
| the annotation JSON (`characteristics` and `comments`) | 502 |

Example: PXD005982 `modification = NT=Acetyl;MT=Variable`, `collision_energy = 30`, `ionization_type = nanoESI` are SDRF cells. The corresponding JSON entries are `acetylation on Lys (variable)`, `30` and `nanoESI`.

## 3. The two texts (example PXD005982)

| Text | Path | Size |
|---|---|---:|
| Text seen by the extraction model | `NLP_traintest/test_set/PXD005982/PXD005982_annotation.json`, field `text` | 34,656 chars |
| Text seen by the judge | `NLP_traintest/test_set/PXD005982/manuscript.txt` | 30,580 chars (30,706 bytes) |
| Difference | | 4,076 chars |

Structure of the 34,656 characters:

| Block | Section | Chars | In `manuscript.txt`? |
|---|---|---:|---|
| PRIDE metadata (header `pride_meta:`) | `projectDescription` | 315 | No |
| | `sampleProcessingProtocol` | 1,007 | No |
| | `dataProcessingProtocol` | 2,653 | No |
| Paper (header `pmcid_PMC5535869:`) | `TITLE` | 88 | Yes |
| | `ABSTRACT` | 1,454 | Yes |
| | `INTRO` | 4,653 | Yes |
| | `METHODS` | 4,683 | Yes |
| | `RESULTS` | 10,406 | Yes |
| | `DISCUSS` | 9,209 | Yes |

The three PRIDE sections add up to 3,975 chars. The remaining 101 chars are the headers (`pride_meta:`, the `=== name ===` lines and the `pmcid_...` line).

Sentences that exist only in the PRIDE block of PXD005982 (word-boundary search, count in annotation text vs `manuscript.txt`):

| Term | Annotation `text` | `manuscript.txt` |
|---|---:|---:|
| `NCE` (as in "NCE setting as 30") | 1 | 0 |
| `NSI` (as in "NSI source") | 1 | 0 |
| `electrospray voltage` | 1 | 0 |
| `Acclaim PepMap` | 2 | 0 |
| `EASY-nLC` | 2 | 1 |
| `MaxQuant` | 4 | 2 |
| `Trypsin/P` | 2 | 1 |

Consequence for PXD005982: the judge flagged `collision_energy = 30` and `ionization_type = nanoESI` as HALLUCINATED. The support is in `dataProcessingProtocol`: "Peptides were selected for MS/MS using NCE setting as 30" and "The peptides were subjected to NSI source". `nanoESI` is a normalisation of "NSI source". Because the judge must not see descriptors, these flags are correct: the values are not in the paper text. They show that HAMLET annotated them from the PRIDE descriptor. The label HALLUCINATED is misleading for these rows. "Not supported by the paper text (descriptor only)" is accurate.

Descriptor text in `manuscript.txt`: the removal of descriptors is incomplete or not distinguishable. Over all 137 PXDs, 799 of 3,444 descriptor sentences (23%, sentences of 60 or more characters) occur verbatim in `manuscript.txt`. This happens in 95 of 137 PXDs (22 test, 73 train). Most hits are in the 119 headerless manuscripts (701 sentences). In the 18 headered manuscripts they sit in `METHODS` (90) and `ABSTRACT`/`INTRO` (8). PRIDE descriptors often copy the abstract and methods of the paper, so a verbatim overlap does not prove that a descriptor leaked. It does mean that the manuscript and descriptor cannot be separated by string matching alone. The check against the original papers is still open.

### All 18 test PXDs with an annotation JSON

Full table: `debug_report/text_input_comparison.csv`.

| PXD | annotation.json `text` | manuscript.txt | difference | projectDescription | sampleProcessingProtocol | dataProcessingProtocol | PRIDE sentences absent from manuscript.txt |
|---|---:|---:|---:|---:|---:|---:|---:|
| PXD001856 | 45,589 | 39,656 | 5,933 | 1,398 | 3,448 | 986 | 19/41 |
| PXD005982 | 34,656 | 30,580 | 4,076 | 315 | 1,007 | 2,653 | 29/30 |
| PXD006847 | 43,971 | 43,577 | 394 | 99 | 63 | 131 | 1/1 |
| PXD012439 | 58,208 | 52,728 | 5,480 | 1,734 | 2,822 | 823 | 35/36 |
| PXD013543 | 48,964 | 42,805 | 6,159 | 835 | 3,047 | 2,176 | 32/39 |
| PXD013809 | 96,188 | 88,487 | 7,701 | 913 | 4,996 | 1,691 | 47/59 |
| PXD014762 | 40,290 | 31,166 | 9,124 | 1,467 | 4,695 | 2,861 | 44/53 |
| PXD015272 | 32,789 | 31,915 | 874 | 359 | 284 | 130 | 8/8 |
| PXD015289 | 67,215 | 63,002 | 4,213 | 301 | 2,752 | 1,059 | 14/27 |
| PXD021383 | 35,323 | 33,569 | 1,754 | 274 | 767 | 612 | 4/12 |
| PXD028781 | 71,146 | 63,671 | 7,475 | 1,458 | 4,562 | 1,354 | 44/51 |
| PXD029997 | 68,783 | 68,099 | 684 | 73 | 331 | 179 | 4/4 |
| PXD034195 | 66,288 | 61,023 | 5,265 | 734 | 2,993 | 1,437 | 26/38 |
| PXD036655 | 53,238 | 47,140 | 6,098 | 1,395 | 3,577 | 1,025 | 31/37 |
| PXD040722 | 60,500 | 53,942 | 6,558 | 675 | 4,098 | 1,684 | 39/47 |
| PXD042574 | 37,374 | 33,559 | 3,815 | 1,040 | 1,026 | 1,648 | 19/20 |
| PXD043392 | 66,464 | 62,823 | 3,641 | 1,445 | 1,066 | 1,029 | 19/19 |
| PXD060381 | 49,190 | 46,309 | 2,881 | 702 | 1,340 | 738 | 23/23 |
| **Total** | 976,176 | 894,051 | 82,125 | 15,217 | 42,874 | 22,216 | |

Reading the last column: "19/41" means that 19 of the 41 PRIDE sentences longer than 40 characters do not occur in `manuscript.txt`. Some PRIDE protocol text repeats the methods section, so the overlap is partial.

### The other 119 PXDs

The 12 test PXDs without a JSON and the 107 train PXDs have no stored extraction text. I downloaded the PRIDE project metadata for all 137 PXDs from the public PRIDE Archive API (`https://www.ebi.ac.uk/pride/ws/archive/v2/projects/<PXD>`). The files are in `debug_report/pride_metadata/<PXD>.json`. For the 18 PXDs with a JSON, the downloaded text matches the PRIDE block inside `text` (18 of 18).

Format note: the 119 `manuscript.txt` files without a JSON have no `=== SECTION ===` headers (they use `ABSTRACT:` style headers). They are 1.6k to 47k chars, median 14.5k.

## 4. Row-level check of every judged value

Method (`debug_report/all_rows_support_check.csv`, 4,714 rows): each judged value is reduced to search terms. For SDRF syntax such as `NT=Oxidation;MT=Variable;TA=M` the term is the first 6 letters of the modification name. Other values are matched as a whole, case-insensitive, with a word boundary at the start. Bare numbers are only accepted when a field keyword is within 120 characters (for example `NCE` for collision energy, "missed" for missed cleavages). Otherwise they are marked undecidable.
Classes: A = found in `manuscript.txt`; C = not in `manuscript.txt`, but in PRIDE metadata; D = in neither; u = undecidable (bare number).
This is string matching. It misses paraphrases and normalised values, so class D is an upper bound for unsupported values.

HALLUCINATED rows:

| Split | Rows | A in manuscript.txt | C only in PRIDE metadata | D in neither | u |
|---|---:|---:|---:|---:|---:|
| test (30 PXDs) | 69 | 6 | 32 | 28 | 3 |
| train (107 PXDs) | 321 | 14 | 39 | 251 | 17 |
| **Total** | **390** | **20 (5%)** | **71 (18%)** | **279 (72%)** | **20 (5%)** |

Other flagged rows that are supported only by PRIDE metadata: NO_MATCH 8, PARTIAL 9, TYPE_MISMATCH 2. In addition, 62 rows judged CORRECT are not in `manuscript.txt` either (16 test, 46 train). The judge therefore accepted some values that cannot be verified from the text it was given, which shows an inconsistent standard.

Breakdown of the 279 class D rows:

| Group | Rows | Comment |
|---|---:|---|
| Other biological or design fields (organ, age, disease state, individual, sample counts, and so on) | 111 | Needs manual review. Several are inferable values. |
| Technical values (labeling, instrument, enzyme, tolerance) | 56 | Normalised terms such as `TMT127`, `AC=MS:1001910;NT=LTQ Orbitrap Elite`. |
| Modification strings | 44 | Normalised SDRF strings, for example `NT=Oxidation;MT=Variable;TA=M;AC=UNIMOD:35`. |
| Converter artefact (file to value mapping strings) | 41 | See section 5.1. |
| Excluded benchmark fields | 27 | `material_type`, `ethnicity`, and so on. Drop these. |

### Worked example: all HALLUCINATED rows of PXD005982

| Judged value | Where the support is | Verdict on the flag |
|---|---|---|
| `collision_energy = 30` | `dataProcessingProtocol`: "NCE setting as 30" | Correct flag: descriptor only, not in the paper text |
| `ionization_type = nanoESI` | `dataProcessingProtocol`: "NSI source" | Correct flag: descriptor only. The SDRF value is a normalisation. |
| `modification = NT=Acetyl;MT=Variable` | Paper (`METHODS`) and descriptor: "acetylation on Lys protein N-terminal" | Correct flag. HAMLET split one phrase into two entries (`acetylation on Lys` and `protein N-terminal acetylation`). The converter then dropped `K`. |
| `characteristics[concentration of compound] = 300 nl minutes-1` | Flow rate, found in `METHODS` and `dataProcessingProtocol` | Genuine SDRF error (flow rate in the wrong column). |

## 5. Conversion errors around the judge

### 5.1 SDRF to judge rows (flattening)

- 404 judged values (of 4,714) are strings that map raw files or row numbers to values, for example `TDM_M1808_198.raw = 1 | TDM_M1808_186.raw = 2 | ...` or `row 1 = 1 | row 2 = 2`. They come from per-run columns that the loader collapses into one string. Such a string cannot occur in a paper. The judge gave 247 of them CORRECT, 59 PARTIAL, 56 NO_MATCH and 41 HALLUCINATED. Fields: technical_replicate 113, fraction_identifier 109, biological_replicate 105, individual 45, factor_value 32.
- 64 rows of the type `sampleChecksum` (a hash, PXD002619) were judged (58 NOT_IN_TEXT). This is a bookkeeping column and should not be judged.
- 262 rows belong to fields that you exclude from the benchmark (`material_type`, `ptm`, `technology_type`, `developmental_stage`, `ethnicity`, `mass_analyzer`). 27 of the 390 hallucination flags are in these fields.

### 5.2 Annotation JSON to SDRF (information loss)

- The JSON keeps residue and position in free text: `acetylation on Lys (variable)`. The SDRF string for the same item in PXD005982 is `NT=Acetyl;MT=Variable`. `TA=K` is lost, and so is the distinction from the N-terminal entry `NT=Acetyl;PP=Protein N-term;MT=Variable`.
- Across the 18 test PXDs, 26 of 36 JSON modification entries name a residue or a position, while the SDRFs hold 14 `TA`, 6 `MT` and 8 `PP` annotations (distinct strings per PXD, so these counts are indicative).
- Across train and test, 88 of 384 judged modification strings lack `TA`. The judge marked 63 modification rows PARTIAL, and every one of its corrections only adds `TA`, `MT` or `PP`. These are incomplete values, not invented values. The label "hallucination" does not fit them.
- I could not confirm whether the SDRFs were generated from the JSON by a script or edited by hand. The README says human annotated.

### 5.3 Judge to `filtered_sdrf`

- `results_*/filtered_sdrf/<PXD>.sdrf.tsv` differs from the input SDRF. In PXD005982, 39 cells changed. Judged values were replaced with `not available` (collision energy, ionization type, compounds, times, temperatures, sample treatment). Judge corrections were applied (`TA=C` and `TA=M` added).
- Two applied changes are wrong in PXD005982:
  - flow rate `300 nl minutes-1` became `300 µL∙min−1`. The paper and PRIDE text both say nl.
  - `three biological replicates` became `1`. I did not trace the origin of this value.
- Values that the judge flagged because of the missing PRIDE text were blanked in `filtered_sdrf` (verified for collision energy and ionization type in PXD005982). Annotations that were correct were lost. I checked this on PXD005982 and PXD001856 only, not on all 137 PXDs.

### 5.4 Fields that cannot be answered from a paper

- The judge prompt for `modification` says: "THE PAPER TEXT IS THE ONLY REFERENCE" and marks modifications that are not named in the text as HALLUCINATED. UNIMOD accessions (`AC=UNIMOD:35`) and instrument accessions (`AC=MS:1001910`) in the SDRF string are never in the paper. The judge needs to compare the term, not the accession string.

## 6. Which judge script was used

The scripts are in `/home/tine/git/Publication/HAMLET/src/python/` (`sdrf_judge.py`, `LLm_as_judge.py`). They contain hard-coded paths of another user (`/Users/fateme/Desktop/...`).

- `sdrf_judge.py` (`_extract_sections_or_full`) keeps only `TITLE`, `ABSTRACT` and `METHODS` when these headers exist. If none exists, it uses the full text. That would hide `INTRO`, `RESULTS` and `DISCUSS` for the 18 headered test PXDs.
- The judge's own quotes do not support this. I located 578 quoted passages in the SOURCE_CHECK text of the 18 PXDs. 239 are in `TITLE`, `ABSTRACT` or `METHODS`. 35 appear verbatim only in `INTRO`, `RESULTS` or `DISCUSS`. Only 2 appear only in PRIDE metadata. The run therefore used the full `manuscript.txt`, not title, abstract and methods only.
- `SKIP_ANNOTATION_FIELDS` in `sdrf_judge.py` is `{"phenotype", "ethnicity", "technology_type"}`, but the review CSVs contain these fields (for example 9 ethnicity hallucination flags). The HAMLET copy is therefore not the exact version that was run.
- The filtering step (`filtered_sdrf`, `*_sdrf_overrides.json`) belongs to `finalize_sdrf.py`. The override JSON files are not in `json_outputs/` here. I did not trace the filter logic.

## 7. What this means for the manuscript

1. Keep the judge input as designed: manuscript text only, no PRIDE descriptors. An earlier version of this report recommended appending the descriptor sections. That recommendation is withdrawn.
2. Do not report the current hallucination rate (390 of 4,714 rows, 8.3%) as an error rate of HAMLET. Split the flags into: (a) value only in the descriptor (71), (b) value not found anywhere (279, upper bound, needs manual review), (c) judge error (20), (d) converter artefact or excluded field (41 and 27 within the 279). Use "not supported by the paper text" instead of "hallucinated".
3. Fix the converter and the annotation before a rerun:
   - Parse three-letter residues after "on" (`on Cys`, `on Met`, `on Lys`) into `TA`.
   - Do not let a misspelled name fall back to a substring match (`carbamamidomethylation` became `NT=Methyl`). Use an exact or fuzzy match on the full name, and fail loudly.
   - Ask the extraction agent for residue and fixed/variable status of every search modification. The manuscript states them for most entries (section 10), and HAMLET recorded the status for 6 of 36.
   - Prevent split entries for one phrase ("acetylation on Lys protein N-terminal").
4. Fix the judge-side artefacts before a rerun:
   - Remove `sampleChecksum` rows, excluded fields and the run-to-value strings.
   - Compare modification names and do not require accession strings (`AC=UNIMOD:35`) in the text.
   - Do not apply judge corrections to the SDRFs without a unit and number check.
5. Check the descriptor overlap in `manuscript.txt` against the original papers (section 3).

## 8. Files in this report folder

| File | Content |
|---|---|
| `DEBUG_REPORT_llm_judge_input.md` | This report |
| `text_input_comparison.csv` | Per-PXD text sizes and PRIDE block sizes (18 PXDs) |
| `all_rows_support_check.csv` | All 4,714 judged rows with support class A, C, D or u |
| `json_to_sdrf_field_comparison.json` | HAMLET JSON value versus SDRF value for every field of the 18 PXDs (424 field instances) |
| `modification_localization_check.csv` | The 36 modification entries: JSON text, SDRF string, residue/position/status in JSON, SDRF and manuscript, judge verdict |
| `pride_metadata/<PXD>.json` | PRIDE Archive project metadata for all 137 PXDs (used for classification only, not as judge input) |

## 9. Limits of this analysis

- Support classes use string matching. Normalised values (for example `nanoESI`, `TMT127`) can fall into class D although the paper or the descriptor supports them.
- The SDRFs are taken to be derived from HAMLET annotations, as stated by the group. For the 18 test PXDs this fits the data (section 10). For the 12 test and 107 train PXDs without an annotation JSON, I could not check the conversion.
- I could not find the code that converts the HAMLET JSON to the SDRF (`HAMLET/src/python/sdrf_builder.py` builds a different SDRF from RunAssessor output). The converter behaviour in section 10 is inferred from 36 input and output pairs.
- I could not run or inspect the original judge run, so the input version is inferred from the output files (section 6).

## 10. HAMLET annotation and JSON to SDRF conversion (18 test PXDs with a JSON)

Question: if the conversion loses information, the judge flags it. Is that where the flags come from, and is the localisation of modifications even derivable from the text?

### 10.1 Field level: the conversion is almost lossless

Comparison of every non-empty HAMLET field with the SDRF column of the same name (`debug_report/json_to_sdrf_field_comparison.json`). HAMLET groups some fields differently from the SDRF (for example `Modification` and `Label` are under `characteristics` in the JSON and `comment[...]` in the SDRF). I matched by name only.

| Result | Field instances (PXD x field) |
|---|---:|
| Value identical in JSON and SDRF | 357 |
| Value changed (normalised) | 67 |
| Field in JSON, empty in SDRF (dropped) | 0 |
| Field in SDRF, empty in JSON (added) | 0 |

The 67 changed values are in six fields only:

| Field | Instances | What the converter does |
|---|---:|---|
| `Instrument` | 18 | Adds `NT=` prefix |
| `CleavageAgent` | 17 | Adds `NT=` prefix |
| `Organism` | 11 of 17 | Removes the taxon id (`9606 (Homo sapiens)` becomes `Homo sapiens`) |
| `Modification` | 13 | Rewrites into `NT=..;PP=..;TA=..;MT=..` (section 10.2) |
| `Label` | 5 of 10 | `label-free` becomes `label free sample` |
| `BiologicalReplicate` | 3 of 4 | Joins values with `; ` into one cell |

So the many flags outside modifications do not come from information lost in the conversion. They come from values that HAMLET extracted and that are not in the paper text, or from the judge.

### 10.2 Modifications: localisation is derivable, and the judge's point is mostly fair

36 modification entries in 13 PXDs. The HAMLET entry and the SDRF string were paired by order (the SDRF keeps the JSON order in all 13 PXDs). "Manuscript states it" means that a residue, an N-terminal position or fixed/variable appears within 120 characters of the modification name in `manuscript.txt`.

| Element | In SDRF | In HAMLET JSON but lost by the converter | Not in JSON, manuscript states it | Not in JSON, not stated near the name |
|---|---:|---:|---:|---:|
| Residue (`TA=`) | 14 | 3 | 17 | 2 |
| Position (`PP=`) | 8 | 0 | 20 | 8 |
| Fixed/variable (`MT=`) | 6 | 0 | 22 | 8 |

Reading:
- **Localisation is possible from the text.** For residue, the manuscript states it for 32 of 36 entries. For fixed/variable, 28 of 36. Position only applies to N- and C-terminal modifications, so the `PP` rows overcount "not in JSON": the 8 entries that HAMLET marks as N-terminal were all carried to the SDRF.
- **HAMLET is the main source of incompleteness.** It recorded the residue for 17 entries and the status for 6. The judge's 25 PARTIAL verdicts on modifications are mostly this: 15 of the 25 lack `TA` and most lack `MT`. The corrected values only add `TA`, `MT` or `PP`. The flag is fair, and the manuscript contains the answer. Example PXD006847: "Carbamidomethylation of cysteine residues was set as a fixed modification" is in the manuscript, while HAMLET recorded the name only.
- **The converter drops three residues.** All three are in PXD005982 and all use the form "on Cys", "on Met", "on Lys":

  | HAMLET JSON | SDRF | Judge |
  |---|---|---|
  | `carbamidomethylation on Cys (fixed)` | `NT=Carbamidomethyl;MT=Fixed` | PARTIAL |
  | `oxidation on Met (variable)` | `NT=Oxidation;MT=Variable` | PARTIAL |
  | `acetylation on Lys (variable)` | `NT=Acetyl;MT=Variable` | HALLUCINATED |

  Forms such as `(C)`, `of cysteine` and `(methionine)` are parsed correctly in other PXDs. The three-letter residue after "on" is not.
- **The converter turns a misspelling into a wrong modification.** PXD006847: the HAMLET entry `carbamamidomethylation` (typo of carbamidomethylation) became `NT=Methyl`. The paper says "Carbamidomethylation of cysteine residues was set as a fixed modification". The judge's HALLUCINATED flag on `NT=Methyl` is correct. The cause is an extraction typo plus a substring match to `methyl`.
- **HAMLET split one phrase into two entries.** In PXD005982 the sentence "acetylation on Lys protein N-terminal were specified as variable modifications" produced `acetylation on Lys (variable)` and `protein N-terminal acetylation (variable)`. The first entry is spurious, and the judge's HALLUCINATED flag on it is fair.

### 10.3 Completeness of HAMLET for modifications

5 of the 18 PXDs have no modification in the HAMLET JSON and none in the SDRF. In four of them, the manuscript names search modifications:

| PXD | Modification words in `manuscript.txt` |
|---|---|
| PXD012439 | oxidation |
| PXD013543 | carbamidomethylation, oxidation, phosphorylation |
| PXD014762 | acetylation, carbamidomethylation, iodoacetamide, oxidation, phosphorylation |
| PXD036655 | acetylation, carbamidomethyl, methylation, TMT10 channels |
| PXD060381 | iodoacetamide only |

The words alone do not prove that these are search parameters. The judge cannot flag them, because the SDRF has no row to judge. This is an omission error that the judge design does not measure.

### 10.4 Which flags remain unexplained

After removing the converter issues above, the judge flags that stay are: values from the descriptor only (71), values found nowhere (279, minus 41 artefact strings and 27 excluded fields), and judge errors (20). For the 12 test and 107 train PXDs without a JSON, I cannot separate extraction errors from conversion errors, because the HAMLET output is not in the folder.
