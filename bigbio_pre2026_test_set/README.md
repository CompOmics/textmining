# bigbio pre-2026 test set

Standalone test set of 30 PXDs with human-curated SDRF-Proteomics gold, for a
rerun of the cumulative ablation (SapBERTStaircase S0-S5). It is separate from
`agentic-benchmark/benchmark_data/` and nothing there was changed.

Built by `build_test_set.py` (2026-09-23). The script can be rerun and gives
the same output, except for text that has to be added by hand (see below).

## Selection

- **Source:** `bigbio/proteomics-sample-metadata`, `annotated-projects/`, at
  commit `c1873bd` (2025-10-16), the last commit before 2026-01-01. That
  repository was later migrated to `bigbio/sdrf-annotated-datasets`, whose
  history starts on 2026-04-16. The cutoff excludes the LLM-assisted
  annotations of 2026. At this commit, `annotated-projects/` holds 232 PXDs.
- **Not in training:** all 107 training PXDs (`agentic-benchmark/benchmark_data/train_set`)
  are among the 232, so they were removed. This leaves 125 candidates.
- **Open access:** the originating publication had to be open access. It was
  identified by (i) the PRIDE `pubmedID`, or (ii) a Europe PMC full-text hit
  for the accession whose title matches the PRIDE project title. The evidence
  for each PXD is in `manifest.tsv` (`paper_match`).
- **Overlap with the old test set:** only PXD018830 was also in the previous
  30-PXD test set.

## Layout

```
<PXD>/sources/*.sdrf.tsv       original bigbio SDRF file(s), unmodified
<PXD>/<PXD>.sdrf.tsv           all SDRF files of the PXD merged (union of columns)
<PXD>/manuscript.txt           abstract + methods, no headings
<PXD>/manuscript_fulltext.txt  title, abstract and all body sections, with headings
gold/<PXD>_<Agent>_golden.json same format as agentic-benchmark/benchmark_data/SDRFS_github
manifest.tsv                   one row per PXD: paper, licence, match evidence, counts, text status
```

The previous test set used both text styles: abstract + methods for its 12
SDRF PXDs and sectioned full text for the other 18. Both files are provided
here. Choose one and use it for all 30.

## Merging and gold conversion

- **Merging:** PXDs with several SDRFs (for example `-dda`/`-dia`,
  `-silac`/`-tmt`, or `Exp1`-`Exp6`) were merged into one table. Columns are
  the union across the files, and repeated headers such as
  `comment[modification parameters]` are aligned by occurrence. Empty cells
  are written as `not available`, which the converter ignores.
- **Conversion:** the gold comes from the unmodified `convert_sdrf()` in
  `agentic-benchmark/benchmark_data/sdrf_to_golden.py`, with two additions:
  - `framework/core/field_mappings.py` only knows the Kaggle-style headers of
    the training SDRFs. The standard headers `comment[cleavage agent details]`,
    `comment[dissociation method]`, `comment[reduction reagent]` and
    `characteristics[strain]` therefore return null. The build script fills
    these fields from those columns, but only when `convert_sdrf()` left them
    null. Without this, `cleavage_agent` would be null for all 30 PXDs. The
    same gap affected the 12 SDRF-derived PXDs of the old test set.
  - `NT=...;AC=...` values in biological fields are reduced to the name.
- **Unmapped:** `ptm` stays null because `comment[modification parameters]`
  is not mapped. It is one of the excluded benchmark fields anyway.
- **Non-null scored fields over the 30 PXDs:** species 30, instrument 30,
  cleavage_agent 30, label 30, fractions 29, replicates 28, organ 26,
  precursor_tolerance 24, fragment_tolerance 23, disease 20, cell_type 19,
  factor_value 19, fragmentation 18, cell_line 14, sex 13, collision_energy 12,
  age 10.

## Text sources and open items

| Group | n | Text |
|---|---|---|
| Europe PMC OA full text (`epmc`) | 22 | fetched |
| NCBI PMC author manuscript (`ncbi`) | 1 (PXD017710) | fetched. Not an OA licence. The file states it may be downloaded for text mining. |
| CC BY, no downloadable full text (`manual`) | 7 | **missing** |

- **Missing text:** PXD001487, PXD001736, PXD002192, PXD004436, PXD004624,
  PXD006542 and PXD012243 are CC BY articles in *Molecular & Cellular
  Proteomics*. PMC does not include them in its OA web service, and PMC and
  the publisher block automated download with a captcha. For these, save the
  text or PDF from a browser, write `manuscript.txt` (and
  `manuscript_fulltext.txt`) into the PXD folder, and set `text_status` in
  `manifest.tsv`. SDRF and gold are already in place.
- **Short methods:** PXD008841 (Johansson et al. 2019, *Nat Commun*) has only
  a short Methods section in the main text; the details are in the
  supplement. Its `manuscript.txt` is therefore short (2,265 characters), and
  most of the information is in `manuscript_fulltext.txt` (Results).
- **Shared paper and SDRF:** PXD018883 and PXD019185 share one publication
  and one SDRF (`PXD019185_PXD018883.sdrf.tsv`). They are one entry here,
  under PXD018883.
- **Reserve:** PXD023650 (author manuscript, text mining permitted, SDRF at
  the same commit) can replace an entry. It is not built.

## Rerunning the ablation on this set

1. Run S0-S5 inference (3 models x 3 replicates) with `<PXD>/manuscript*.txt`
   as input.
2. Score with `final_analyses/SapBERTStaircase/scripts/rescore_replicates.py`,
   after setting its `GOLD` constant (line 35) to `bigbio_pre2026_test_set/gold`.
