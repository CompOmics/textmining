# textmining

Working directory for the reduced manuscript ("text-mining-only" scope, see
`outline.txt`). Contents copied out of `../HAMLET` on 2026-08-18, keeping only
what the outline's scope decision keeps IN.

## Layout

- `framework/` -- the agentic extraction pipeline: `agents/` (biological,
  technical, experimental-design, integration, normalization agents),
  `core/` (extractor, prompts, field mappings, LLM wrapper), `normalization/`
  (ontology graph, SapBERT/FAISS index, synonym handling), `validation/`
  (structural rules, gleaning), `tests/`, `docs/`, plus `main.py`,
  `config.yaml`, `prompts_reference/`. This is Section 2 / M&M 5-8 of the
  outline. Source: `HAMLET/src/agentic-metadata/agentic-metadata/`.
- `benchmark/` -- SDRF-derived ground truth and matching code (I9/I10/I11,
  Section 3-4, M&M 10): `benchmark_data/` (matched, sdrf_golden, test_set,
  new_test_set, `run_sdrf_benchmark.py`, `sdrf_to_golden.py`,
  `annotation_to_golden.py`, plotting scripts), `run_bench.sh`.
- `corpus/` -- human annotation corpus and PRIDE-comparison materials
  (I2-I6, I13, Section 1 / Figure 1 / Figure 6, M&M 1-3):
  - `Figure1_source/` -- `data/Select_27_Pubs`, `data/GoldenAnnotations`
    (`Annotations`, `HarmonizedHuman`, `SingleHuman`, `MultiHuman`, `GPT`,
    `Representative20`, `KaggleHoldout`), `field_crosswalk_table.csv`,
    `make_figure1.py` / `make_figure1_v2.py`, prior `output`/`output_v2`.
  - `pride_metadata_comparison/` -- `HamletPXDs.csv` (I13, n=299) and the
    full `pxd_lists/` directory.
  - `pride_census/` -- `pride_survey.py` only. The raw crawl
    (`pride_survey/`, ~6.1 GB) was **not** copied; see `NOTE.md` there.
- `ontology_refs/` -- local CV/ontology source files referenced by
  `framework/normalization/`: `unimod/`, `taxonomy/`, `taxid_lists/`,
  `sdrf-terms.csv` (I14). CL, CLO, UBERON, Mondo, PSI-MS CV, and DOID are
  fetched by `framework/normalization/download.py`, not stored here.
- `docs/` -- `Metadata_paper_v2.1-2.pdf` (current full draft, superseded by
  the reduced outline) for reference.
- `outline.txt` -- the reduced manuscript outline (moved, not copied, from
  the agentic-metadata submodule).

