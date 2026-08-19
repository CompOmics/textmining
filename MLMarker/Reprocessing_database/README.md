# Proteomics Metadata Extraction

Automated extraction and harmonization of proteomics metadata from multiple sources: scientific manuscripts (via LLM), PRIDE SDRF files, run filenames, and ML-based tissue prediction. The goal is to produce unified, standardized run-level metadata for large-scale proteomics datasets.

## Project structure

```
agentic_metadata/           # LLM manuscript extraction pipeline
  main.py                   # CLI entry point
  core.py                   # Extraction orchestration
  config.yaml               # Pipeline configuration
  schemas.py                # JSON schema for LLM output
  export_tsv.py             # JSON -> TSV conversion
  fetch_manuscripts.py      # PRIDE API + PubMed manuscript fetching
  fetch_sdrfs.py            # SDRF file fetching from PRIDE
  prompt_builder/            # Prompt assembly (instructions + schema + text)
  ollama/                    # Ollama LLM client
  normalization/             # SapBERT-based term normalization
    tracker.py               # Normalization QC reporting
  ontologies/                # Controlled vocabulary files (.txt)
  cache/
    ontology/                # FAISS indices for SapBERT
    sdrf/                    # Cached SDRF files from PRIDE
  benchmark/                 # Evaluation against ground truth
  results/
    manuscripts/             # Downloaded manuscript texts
    extracted_metadata/      # Extracted JSON per PXD
    agent_metadata.tsv       # Combined TSV export
    normalization_map.tsv    # Normalization QC report

run_metadata/               # Run-level metadata extraction pipeline
  main.py                   # CLI entry point
  config.yaml               # Pipeline configuration
  patterns.py               # Regex patterns for filename extraction
  pipeline_name.py          # Extract metadata from run filenames
  pipeline_sdrf.py          # Extract metadata from SDRF files
  pipeline_mlmarker.py      # ML tissue prediction via MLMarker
  pipeline_combine.py       # Merge all sources with evidence tracking

parition_combiner/          # Fraction group detection
  extract_fraction_groups.py # Identify fractions from run names

data/                       # Input data and outputs
  pride_quant.parquet        # Protein quantification (pxd, run, proteins)
  sample_name_mapping.tsv    # Maps generic run names to descriptive names
  run_metadata/              # Run-level pipeline outputs
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed (for agentic extraction only)
- CUDA GPU recommended (for SapBERT normalization and MLMarker)
- Conda environment: `agentic-metadata`

### Install dependencies

```bash
conda activate agentic-metadata
pip install requests pyyaml ollama transformers torch numpy pandas scikit-learn
pip install mlmarker --no-deps
pip install shap plotly seaborn bioservices gprofiler-official
```

---

## Pipeline 1: Agentic Metadata Extraction

Extracts structured metadata from scientific manuscripts using a local LLM. For each PRIDE dataset, it fetches the manuscript, sends it to the LLM with a structured prompt, and normalizes the extracted terms against controlled vocabularies.

### Configuration

Edit `agentic_metadata/config.yaml`:

```yaml
ollama:
  base_url: http://localhost:11434
  model: gurubot/Qwen3.5-35B-A3B-GGUF-unsloth-nothink:UD-Q4_k_XL
  inference:
    num_ctx: 32768
    temperature: 0.0
    timeout: 200
    max_tries: 2

data:
  manuscripts_dir: results/manuscripts     # relative to config file
  fetch_manuscripts:
    enabled: true
    pxd_source: ../data/pride_quant.parquet # reads PXD IDs from parquet
  output_dir: results

normalization:
  enabled: true
  ontology_dir: ontologies/
  cache_dir: cache/ontology/
  similarity_threshold: 0.7
  use_gpu: true
  post_normalization_map:
    disease:
      triple negative breast cancer: breast cancer
      disease free: healthy
```

All paths are relative to the config file location.

### Commands

**Build ontology cache** (run once, or after editing vocabulary files):
```bash
python -m agentic_metadata setup --config agentic_metadata/config.yaml
```
Builds FAISS indices from the `.txt` vocabulary files in `ontologies/`. Required before extraction or QC.

**Fetch manuscripts** (download only, no extraction):
```bash
python -m agentic_metadata fetch --config agentic_metadata/config.yaml
```
Downloads manuscripts from PRIDE/PubMed for all PXDs in the configured `pxd_source`. Skips PXDs that already have a manuscript file (incremental).

**Run full extraction**:
```bash
python -m agentic_metadata run --config agentic_metadata/config.yaml
```
Fetches manuscripts (if enabled), runs LLM extraction, normalizes terms, and exports `agent_metadata.tsv` + `normalization_map.tsv`.

**Normalize existing results** (re-run normalization without re-extracting):
```bash
python -m agentic_metadata normalize agentic_metadata/results/extracted_metadata --config agentic_metadata/config.yaml
```

**Vocabulary QC** (generate normalization report from existing JSONs):
```bash
python -m agentic_metadata qc agentic_metadata/results/extracted_metadata --config agentic_metadata/config.yaml
```
Produces `normalization_map.tsv` showing for each field:
- Which raw values were mapped to which controlled vocabulary terms
- Which values could not be normalized (and their similarity scores)
- Counts for each mapping

Use this to identify gaps in the vocabulary files and add missing terms.

**Run benchmark**:
```bash
python -m agentic_metadata benchmark --config agentic_metadata/config.yaml
```
Extracts on the built-in test set (31 manuscripts) and evaluates against ground truth labels with precision/recall/F1.

### Outputs

| File | Description |
|------|-------------|
| `results/extracted_metadata/PXD*.json` | Raw extraction + normalization per PXD |
| `results/agent_metadata.tsv` | Combined TSV (one row per sample group) |
| `results/normalization_map.tsv` | Normalization QC report |

### Extracted fields (19)

| Category | Fields |
|----------|--------|
| Biology | organism, tissue, disease, cell_part, cell_line |
| Sample prep | enzymes, modifications, labeling, fractionation, enrichment |
| MS config | instrument, fragmentation, collision_energy, acquisition, lc_column, gradient_time_min, ionization |
| Treatments | treatment_type, treatment_name, treatment_class |

---

## Pipeline 2: Run-Level Metadata

Extracts metadata at the individual run level from multiple sources and combines them into a unified file with evidence tracking.

### Configuration

Edit `run_metadata/config.yaml`:

```yaml
input:
  quant_parquet: ../data/pride_quant.parquet

output_dir: ../data/run_metadata/

run_name:
  enabled: true
  output: run_meta_name.tsv

sdrf:
  enabled: true
  output: run_metadata_sdrf.tsv
  sdrf_cache_dir: ../agentic_metadata/cache/sdrf/
  sample_name_mapping: ../data/sample_name_mapping.tsv
  normalization:
    enabled: true
    ontology_dir: ../agentic_metadata/ontologies/
    cache_dir: ../agentic_metadata/cache/ontology/
    similarity_threshold: 0.7
    use_gpu: true

mlmarker:
  enabled: true
  output: run_meta_mlmarker.tsv
  confidence_threshold: 0.3

combine:
  enabled: true
  output: run_metadata_combined.tsv
  agent_metadata: ../agentic_metadata/results/agent_metadata.tsv
```

Each sub-pipeline can be toggled independently with `enabled: true/false`.

### Run all enabled pipelines

```bash
python -m run_metadata --config run_metadata/config.yaml
```

This runs in sequence:

1. **Run name extraction** — regex patterns match instrument names, organisms, tissues, cell lines, etc. from filenames
2. **SDRF extraction** — parses PRIDE SDRF files, normalizes values via SapBERT, drops unmapped terms
3. **MLMarker prediction** — predicts tissue from protein abundance using a Random Forest model (only predictions above confidence threshold are kept)
4. **Combine** — merges all sources with priority-based filling and evidence tracking

### Priority order

When multiple sources provide a value for the same field, the highest priority source wins:

```
sdrf > name > agent > MLMarker
```

### Evidence tracking

Each field has a paired `_evidence` column showing which sources contributed:

| Evidence | Meaning |
|----------|---------|
| `sdrf` | Value from SDRF only |
| `sdrf; agent` | SDRF and agent agree |
| `sdrf; -MLM` | SDRF provided the value, MLMarker disagreed |
| `agent; MLM` | Agent provided the value, MLMarker confirmed |

### Outputs

| File | Description |
|------|-------------|
| `run_meta_name.tsv` | Metadata extracted from run filenames |
| `run_metadata_sdrf.tsv` | Metadata from PRIDE SDRF files |
| `run_meta_mlmarker.tsv` | MLMarker tissue predictions (above threshold) |
| `run_metadata_combined.tsv` | Unified run-level metadata with evidence |

---

## Pipeline 3: Fraction Group Detection

Identifies which runs are fractions of the same sample based on explicit fraction markers in filenames (e.g., `_frac18`, `_fr15`, `_F10`).

### Run

```bash
python parition_combiner/extract_fraction_groups.py --quant data/pride_quant.parquet --metadata data/run_metadata/run_metadata_combined.tsv
```

Options:
- `--quant` — parquet file with pxd and run columns
- `--metadata` — combined metadata TSV (uses the fractionation field to filter to fractionated PXDs)
- `--output` — custom output path (default: `parition_combiner/fraction_groups.tsv`)

### Output

`fraction_groups.tsv` with columns: `pxd`, `run`, `sample_base`, `fraction_id`, `pattern_type`, `group_size`

Only uses high-confidence explicit patterns (`_frac`, `_fr`, `_F`, `_FR`). Filtered to PXDs where the metadata indicates fractionation.

---

## Typical workflow

```bash
# 1. Build ontology indices (once)
python -m agentic_metadata setup --config agentic_metadata/config.yaml

# 2. Fetch manuscripts
python -m agentic_metadata fetch --config agentic_metadata/config.yaml

# 3. Run LLM extraction
python -m agentic_metadata run --config agentic_metadata/config.yaml

# 4. Check vocabulary quality, expand ontologies if needed
python -m agentic_metadata qc agentic_metadata/results/extracted_metadata --config agentic_metadata/config.yaml

# 5. Run all run-level metadata pipelines + combine
python -m run_metadata --config run_metadata/config.yaml

# 6. Extract fraction groups
python parition_combiner/extract_fraction_groups.py \
  --quant data/pride_quant.parquet \
  --metadata data/run_metadata/run_metadata_combined.tsv
```

---

## Adding vocabulary terms

The controlled vocabularies are plain text files in `agentic_metadata/ontologies/` (one term per line, `#` for comments). After editing:

1. Rebuild the ontology cache: `python -m agentic_metadata setup --config agentic_metadata/config.yaml`
2. Re-run QC to verify: `python -m agentic_metadata qc agentic_metadata/results/extracted_metadata --config agentic_metadata/config.yaml`
3. Check `normalization_map.tsv` — previously unmapped terms should now appear as `mapped`

### Post-normalization overrides

For terms that SapBERT maps to a valid but unwanted term, add overrides in `agentic_metadata/config.yaml`:

```yaml
normalization:
  post_normalization_map:
    disease:
      triple negative breast cancer: breast cancer
      disease free: healthy
    tissue:
      heart left ventricle: heart
```
