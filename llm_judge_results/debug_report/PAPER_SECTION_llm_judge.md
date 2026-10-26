# Integrating the LLM judge into `HAMLET_draft_092026.docx`

Written for: the corresponding author, as a working proposal. Nothing was written into the .docx.

## 1. What the judge can support today

The existing run judged the **reference** SDRFs (107 community train, 12 community test, 18 curated JSON converted by `json_to_sdrf`), against the manuscript text. It did not judge HAMLET output (see `DEBUG_REPORT_judge_script.md`, section 0).

That still fits the draft, in two places:

| Draft location | Claim already made | What the judge adds |
|---|---|---|
| Results 1, Discussion (paragraph "That conclusion should be read against the ceiling...") | The SDRF reference is "one reasonable annotation among several" and accuracy is bound by it. | An independent, text-based estimate of how much of the reference the paper text supports. |
| Results 4 (descriptors and the closed-access gap) | The deposited record carries information the paper does not. | Of the reference values the text does not support, a measurable share is found only in the PRIDE descriptor. As run: 32 of 47 flags for the curated JSON papers, 71 of 390 overall. |
| Results 3 and Discussion | Experimental design and technical fields are the weak spot. | The same fields carry most of the unsupported reference values (modification, tolerances, replicate and individual structure). |

What it cannot support today:
- Any statement on HAMLET accuracy or hallucination rate.
- Any statement that the judge is reliable. There is no human audit and no second judge.
- Any comparison across the three manuscript text inputs (F2).

## 2. Two designs

**Design A, reference audit (cheapest, mostly existing data).**
Rerun the 137 papers after fixes F2, F3, F4, F7 (about one OpenRouter run, no new data). Report how much of the reference the text supports and use the result to define a text-attainable ceiling.

**Design B, judge on HAMLET output (recommended, answers the paper's claim).**
Judge the 30 test datasets for the six models and, as a leakage-free check, the PXDs in `bigbio_pre2026_test_set/`. This is the only design that supports a sentence such as "no more than X% of HAMLET values lack support in the manuscript". Keep design A as a one-paragraph companion.

Design B needs: an SDRF or SDRF-like export of each HAMLET run (the judge reads SDRF columns), a different judge family than Gemma (Gemma is also an extractor in the paper), and the human audit of the judge (about 200 rows, kappa).

A metric that follows from design A and costs no new model calls:

**Text-attainable F1.** Recompute the benchmark F1 of the manuscript-only run after removing reference values that the judge marked as unsupported by the text. This separates "the model missed it" from "the reference never had text support". It uses the existing matcher and the judge rows. Report it next to the standard F1, not instead of it.

## 3. Where to put it

- **Results**: a new subsection of about 250 words and one figure, after the six-model benchmark and before "Dataset descriptors are essential to navigate the closed-access gap". The next section then starts from the finding that some reference values exist only in the descriptor.
- **Methods**: one subsection "Independent text-grounding judge" after "Benchmark ground truth and matching", about 150 words.
- **Discussion**: two sentences in the paragraph on the human ceiling.
- **Supplement**: judge prompt (verbatim), per-field table, list of papers with missing methods text, audit protocol and agreement.
- **Figure** (main, 3 panels): (a) stacked bar of match types per field group, (b) support source of unsupported values (manuscript, PRIDE only, nowhere), (c) benchmark F1 and text-attainable F1 per model or per field.
- **Abstract**: no change until design B exists.
- **Terminology**: use "unsupported by the text" and "text-supported". Do not use "hallucination" for reference values. In the draft, "Three constraints limit hallucinations" (Results 2) is about the extractor and stays.

## 4. Draft text

Bracketed numbers are placeholders. Replace them after the rerun. Do not use the as-run values from section 5 in the manuscript.

### Results subsection (design A wording)

**An independent judge shows how much of the reference the text supports**

The expert agreement ceiling (Fig. 1) limits what any extraction can be scored against. We therefore asked a separate question: how much of the reference itself can be recovered from the manuscript text? An LLM judge (Gemma 4 31B, thinking mode) compared each value of the 137 reference SDRFs with the abstract and methods of its paper, without access to the PRIDE record. It checked field type, source in the text, truth for the runs that carry the value, and completeness, and it returned one of six outcomes per value (Methods). In the 30 test datasets, [X]% of the verifiable reference values were supported by the text. [Y]% were not supported, and [Z]% of these were found only in the PRIDE project descriptor, which matches the closed-access pattern in Fig. 4. Unsupported values concentrated in modification parameters, mass tolerances and replicate and individual structure ([fields]), the same entities where human agreement was lowest (Fig. 1). Removing unsupported values from the reference raised the manuscript-only F1 of [model] from [F1] to [F1-attainable], so [Δ] points of the remaining gap reflect reference content that the text does not contain. A human audit of [N] judged values showed agreement of kappa = [K] with the judge (Supplementary Table [S]).

### Methods subsection

**Independent text-grounding judge.** Each SDRF cell was judged against the manuscript text (abstract and methods) with google/gemma-4-31b-it through OpenRouter, in thinking mode at temperature 0. The judge received the field definition, the cell value, the other values of the same column and the number of runs carrying the value, and never the PRIDE descriptor. It answered four questions in order: type, source, truth and completeness. Outcomes were correct, partial (a clear fixable error), type mismatch, unsupported (content absent from the text and not an inference allowed by the field definition), no match (printed but wrong item) and not in text (per-sample values that papers rarely list, such as individual ages). Not-in-text values were excluded from rates. The six fields excluded from the benchmark were not judged. Whole-column mappings (replicate, fraction, factor value) were judged as one unit. Rates are reported per paper and as medians, and a stratified sample of [N] judged values was labelled blind by [two] annotators to estimate agreement. [A second judge of a different model family agreed in [K2].] The prompt is in Supplementary Note [S].

### Discussion additions (two sentences)

"An independent judge found that [Y]% of the reference values are not supported by the manuscript text, so the ceiling of any text-only method lies below full recall of the reference. Reporting a text-attainable score beside the standard score keeps the two failure sources apart."

### Design B wording, if you choose it

"Applied to the manuscript-only HAMLET extractions of the 30 test datasets, the judge found [A]% of values unsupported by the text, [B]% for the strongest commercial model and [C]% for the open-weights model, compared with [D]% for the reference. On the [N] datasets of the public held-out set, the unsupported share was [E]%."

## 5. As-run values (preliminary, not for the manuscript)

Benchmark fields only (six excluded fields removed), 137 papers, 4,452 rows.

| Origin | CORRECT | PARTIAL | NO_MATCH | HALLUCINATED | TYPE_MISMATCH | NOT_IN_TEXT |
|---|---:|---:|---:|---:|---:|---:|
| Community (119 papers) | 2,217 | 212 | 94 | 316 | 58 | 985 |
| Curated JSON (18 papers) | 391 | 36 | 78 | 47 | 18 | 0 |

Correct share among verifiable rows: community 76.5%, curated JSON 68.6%.

Test set only (30 papers, 1,014 rows): CORRECT 673, PARTIAL 51, NO_MATCH 89, HALLUCINATED 69, TYPE_MISMATCH 30, NOT_IN_TEXT 102.

Per-field values with the highest unsupported share in the test set (rows, unsupported share of verifiable rows): modification (58, 22%), fragment mass tolerance (10, 20%), sex (7, 43%), factor value (5, 40%), technical replicate (7, 29%). `cell_type` shows 12 TYPE_MISMATCH of 15 rows, caused by the value "cultured cells" (the judge treats it as a material, not a cell type). That is a reference convention issue to resolve before reporting.

Files: `all_rows_support_check_benchmark_fields.csv` (row level, with origin and support class), `text_input_comparison.csv`.

## 6. Order of work

1. Confirm the provenance question (`DEBUG_REPORT_judge_script.md`, section 5).
2. Fix the inputs and the statistic (items 2 to 6 of the report), rerun design A on 137 papers.
3. Human audit of about 200 rows.
4. Decide on design B. If yes, build the HAMLET-output export and pick a non-Gemma judge.
5. Compute text-attainable F1.
6. Write the subsection with final numbers and add the figure.
