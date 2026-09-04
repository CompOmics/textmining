# SDRF Benchmark

Evaluates the extraction framework against SDRF ground truth annotations using semantic matching. Compares LLM-extracted metadata fields against per-sample SDRF values to measure precision, recall, and F1.

## Quick Start

```bash
# From the extraction_framework root directory:

# 1. Run full benchmark (converts SDRFs → goldens, extracts, compares, plots)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir matched

# 2. Run on test set only (12 PXDs)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir test_set

# 3. Run on new test set with integration agent (18 PXDs)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir new_test_set \
  --integrate \
  --runassessor-dir /path/to/processed_datasets \
  --skip-conversion

# 4. Re-evaluate without re-extracting (fast, for prompt/evaluation changes)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir test_set \
  --skip-extraction \
  --skip-conversion

# 5. Force re-extraction (e.g., after prompt changes)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir test_set \
  --force-extraction

# 6. Run with a different model
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir matched \
  --config benchmark_data/config_gpt.yaml \
  --model-label gpt
```

> **Note:** `CUDA_VISIBLE_DEVICES=""` forces the semantic matcher (SapBERT) to use CPU, which avoids conflicts if GPU is running the LLM server.

## Pipeline Steps

| Step | Description | Flag to skip |
|------|-------------|------|
| **1. Convert SDRFs → Goldens** | Parses `.sdrf.tsv` files into golden-set JSONs (3 per PXD: Biological, Technical, Experimental Design) | `--skip-conversion` |
| **2. Run Extraction** | Runs the full extraction pipeline (`main.py all --validate --normalize`) on each PXD manuscript | `--skip-extraction` |
| **3. Compare** | Compares LLM outputs against golden set using exact, normalized, ontology, hierarchical, and semantic matching | — |
| **4. Generate Plots** | Creates per-agent field metrics plots, summary charts, and overall metrics | — |

## Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--input-dir` | Input directory within `benchmark_data/` | `matched` |
| `--config` | Path to LLM config YAML | `config.yaml` |
| `--model-label` | Model name for output dirs/plots | auto-detect |
| `--workers` | Parallel extraction workers | `4` |
| `--limit` | Limit to N PXDs (for testing) | all |
| `--force-extraction` | Clear and re-extract all PXDs | off |
| `--skip-extraction` | Skip extraction, only compare+plot | off |
| `--skip-conversion` | Skip SDRF→golden conversion | off |
| `--no-filter` | Include metadata-only fields | off |
| `--integrate` | Enable integration agent | off |
| `--runassessor-dir` | Aggregated results directory for integration | — |

## Data Splits

| Split | Directory | PXDs | Description |
|-------|-----------|------|-------------|
| **Train** | `matched/` | 107 | Full training set with SDRF `.tsv` files |
| **Test** | `test_set/` | 12 | Held-out test set with SDRF `.tsv` files |
| **New Test** | `new_test_set/` | 18 | Additional datasets with JSON annotation goldens + aggregated PRIDE data |

## Outputs

All outputs are saved to `reports_{input_dir}/` (or `reports_{input_dir}_{model_label}/`):

```
reports_test_set/
├── plots/
│   ├── biologicalagent_sdrf_benchmark.png      # Per-field metrics (Biological)
│   ├── technicalagent_sdrf_benchmark.png       # Per-field metrics (Technical)
│   ├── experimentaldesignagent_sdrf_benchmark.png  # Per-field metrics (Exp. Design)
│   ├── sdrf_benchmark_summary.png              # Match type distribution
│   ├── sdrf_benchmark_overall.png              # Overall P/R/F1 bars
│   └── overall_agent_metrics.png               # Per-agent grouped P/R/F1
├── sdrf_benchmark_summary.json                 # Machine-readable metrics
├── sdrf_benchmark_detailed.csv                 # Per-PXD per-field results
├── biologicalagent_sdrf_field_metrics.csv       # Field-level stats
├── technicalagent_sdrf_field_metrics.csv
└── experimentaldesignagent_sdrf_field_metrics.csv
```

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `run_sdrf_benchmark.py` | Main benchmark runner (orchestrates all steps) |
| `sdrf_to_golden.py` | Converts `.sdrf.tsv` → golden-set JSON |
| `annotation_to_golden.py` | Converts annotation JSON → golden-set JSON (for `new_test_set`) |
| `setup_benchmark.py` | Sets up benchmark data directories |
| `dataset_mapping.json` | Maps PXD IDs to their SDRF, manuscript, and aggregated result paths |

## Evaluated Fields

### BiologicalAgent
`species`, `organ`, `cell_type`, `cell_line`, `disease`, `sex`, `age`, `developmental_stage`, `ethnicity`, `material_type`, `strain`, `BMI`

### TechnicalAgent
`instrument`, `cleavage_agent`, `label`, `fragmentation`, `ptm`, `reduction_reagent`

### ExperimentalDesignAgent
`replicates`, `technical_replicates`, `number_of_samples`, `technology_type`

> Fields are only compared when the golden annotation has non-null values. Metadata-only fields (not extractable from manuscripts) are excluded by default — use `--no-filter` to include them.

## Semantic Matching

The evaluator uses a 5-tier matching hierarchy:

1. **Exact** — Case-insensitive string match
2. **Normalized** — After removing common suffixes, abbreviation expansion
3. **Ontology** — Accession-based lookup (e.g., `CL:0000084` matches `T cell`)
4. **Hierarchical** — Parent/child ontology relationships (e.g., `HeLa` is-a `cell line`)
5. **Semantic** — SapBERT cosine similarity above threshold
