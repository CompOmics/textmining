# PRIDE Metadata Extraction Framework

An agentic pipeline for extracting and normalizing metadata from scientific manuscripts, specifically designed for proteomics and mass spectrometry data submissions to the PRIDE database.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/CompOmics/agentic-metadata.git
cd agentic-metadata/extraction_framework

# Run setup (creates venv, installs deps, downloads ontologies)
# Requires 'faiss-cpu' for high-performance indexing
./setup.sh

# Activate environment and run
source venv/bin/activate
python main.py all --input /path/to/documents/
```

## Overview

This framework uses specialized LLM-based agents to extract structured metadata from scientific text, normalize terms against biomedical ontologies, and integrate data from multiple sources.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Documents                          │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Biological      │ │ Technical       │ │ Experimental    │
│ Agent           │ │ Agent           │ │ Design Agent    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ Integration     │
                    │ Agent           │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Normalization   │
                    │ (Ontology-based)│
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Structured      │
                    │ Output (JSON)   │
                    └─────────────────┘
```

## Agents

| Agent | Description |
|-------|-------------|
| **BiologicalAgent** | Extracts species, cell types, tissues, diseases |
| **TechnicalAgent** | Extracts instruments, modifications, labelling methods |
| **ExperimentalDesignAgent** | Extracts experimental design, sample preparation |
| **IntegrationAgent** | Merges multi-source data, resolves conflicts using PRIDE descriptor priority |
| **NormalizationAgent** | Maps terms to ontologies using SapBERT embeddings |

## Supported Ontologies

19 biomedical ontologies are supported for term normalization:

| Category | Ontologies |
|----------|------------|
| Cell/Tissue | CL, CLO, BTO, UBERON |
| Species | Curated NCBITaxon subset, Rat Strains |
| Disease | DOID, Mondo |
| Mass Spec | PSI-MS, PRIDE-CV, UNIMOD, PSI-Mod |
| Other | ChEBI, EFO, PATO, Plant Ontology, FlyBase, ZFA, FBbt |

## Configuration

The pipeline is configured via a `config.yaml` file in the root directory. You can customize paths, backend settings, and agent parameters here:

```yaml
paths:
  input_dir: "./docs"            # Relative path to input documents
  output_dir: "./framework_output"
  ontology_dir: "ontologies"

normalization:
  backend: "faiss"               # Options: faiss (fastest), sklearn, annoy
  use_quantization: true         # Reduces memory usage by ~90%
  use_gpu: true                  # Use GPU for embeddings if available

agents:
  temperatures: [0.0]            # LLM sampling settings
  validate: true                 # Enable the standard Validation Agent
```

### Abbreviation Expansion

Abbreviated species names (e.g. `p.falciparum`, `Plasmodium.falciparum`) are handled by injecting them as **synthetic synonyms directly into the ontology graph** before the embedding index is built. This means SapBERT embeds the abbreviated form as a recognised variant of the correct node — no query rewriting needed.

On every ontology load, two abbreviated forms are automatically generated for each binomial name and added to that node's synonym list:

| Node name | Injected synonyms |
|-----------|------------------|
| `Plasmodium falciparum` | `p.falciparum`, `Plasmodium.falciparum` |
| `Homo sapiens` | `h.sapiens`, `Homo.sapiens` |
| `Mus musculus` | `m.musculus`, `Mus.musculus` |

> [!IMPORTANT]
> Delete `ontology_cache/` and rebuild after upgrading so the new synonyms are embedded in the index:
> ```bash
> rm -rf ontology_cache/ && python -m normalization.build_index
> ```

#### Registering new synonyms at runtime

When you encounter an abbreviation the pipeline misses, register it **without rebuilding** the full index:

```python
# Via the pipeline agent
agent.register_synonym(
    synonym="p.falciparum",
    node_name="Plasmodium falciparum",
    entity_type="species",
)

# Or directly on the normalizer
normalizer.register_synonym("p.falciparum", "Plasmodium falciparum", "species")
```

This updates the live FAISS index (new vector appended immediately) and persists the synonym to `ontology_cache/custom_synonyms.json`, so it is reloaded automatically on every future run.


## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/CompOmics/agentic-metadata.git
cd agentic-metadata/extraction_framework

# Run the setup script
./setup.sh

# For full setup including pre-built indices (~10 min):
./setup.sh --full

# For quick setup (skip large ontologies like ChEBI):
./setup.sh --quick
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/CompOmics/agentic-metadata.git
cd agentic-metadata/extraction_framework

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
# Includes faiss-cpu for indexing and requests for robust downloads
pip install -r requirements.txt

# Download ontologies
python -m normalization.download

# Build ontology indices (optional, ~10 min)
# Indices are also built automatically on first --normalize run
python -m normalization.build_index
```

### Disk Space Requirements

| Component | Size |
|-----------|------|
| Core dependencies | ~2 GB (PyTorch, transformers) |
| Ontology files | ~500 MB |
| Ontology indices | ~4 GB |

### Reproducibility

For reproducible results (e.g., for paper experiments):

```bash
# Use frozen dependencies with exact version pins
pip install -r requirements-frozen.txt

# Run with explicit seed
python main.py all --input ./docs --seed 42

# Or configure in config.yaml:
# reproducibility:
#   seed: 42
#   log_info: true
```

**Reproducibility features:**
- `requirements-frozen.txt`: Exact version pins for all dependencies
- `--seed N`: Set random seed via CLI (overrides config)
- `--no-seed`: Disable seeding for non-deterministic mode
- Automatic seeding of Python, NumPy, and PyTorch
- Seed passed to OpenAI-compatible LLM APIs

## Usage

### Basic Extraction

```bash
# Run all agents
python main.py all --input /path/to/documents/

# Run specific agent
python main.py biological --input /path/to/documents/

# With ontology normalization (uses FAISS backend by default)
python main.py all --input /path/to/documents/ --normalize

# With integration from RunAssessor data
python main.py all --input /path/to/documents/ --integrate --runassessor-dir /path/to/data/
```

### Integration Features

The Integration Agent enriches LLM extractions with RunAssessor data:

- **PRIDE Descriptor Priority**: Curator-submitted PRIDE descriptors (species, tissue, disease, instrument, PTMs) are prioritized over automated tool inference
- **Disagreement Logging**: When PRIDE descriptors disagree with tool predictions, conflicts are logged to `ra_disagreements.json` for pipeline debugging
- **Multi-source Resolution**: Combines LLM extraction, PRIDE metadata, and tool inference with tracked provenance

### Configuration Overrides

You can override `config.yaml` defaults using CLI arguments:
-   `--ontology-dir`: Custom ontology location
-   `--validate`: Force validation on/off
-   `--output`: Custom output directory

### Using the Pipeline Script

```bash
# Run with default settings
./run_pipeline.sh

# With custom input directory
./run_pipeline.sh --input /path/to/documents/

# With normalization
./run_pipeline.sh --normalize
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `mode` | `biological`, `technical`, `experimental`, or `all` |
| `--input` | Input directory with `.txt` files |
| `--output` | Output directory (default: `framework_output/`) |
| `--normalize` | Enable ontology normalization |
| `--integrate` | Enable integration with RunAssessor data |
| `--runassessor-dir` | Directory containing RunAssessor JSON files |
| `--validate` | Enable validation agent |
| `--temperatures` | LLM sampling temperatures |

## Output Format

Extractions are saved as JSON with provenance:

```json
{
  "species": {
    "resolved": "Homo sapiens",
    "confidence": 1.0,
    "status": "AGREE",
    "sources": {
      "runassessor": {"value": "Homo sapiens", "accession": "9606", "score": 1.0},
      "llm": {"value": "Homo sapiens", "evidence": "Human plasma samples..."}
    }
  }
}
```

### Disagreement Log

When using `--integrate`, conflicts between PRIDE descriptors and automated tools are logged to `ra_disagreements.json`:

```json
{
  "filename.txt": [{
    "type": "PRIDE_VS_TOOL",
    "field": "species",
    "pride_value": "Riftia pachyptila",
    "tool_name": "organism_identification (Peptonizer)",
    "tool_value": "Drosophila melanogaster",
    "tool_score": 0.996,
    "resolution": "PRIDE descriptor used (curated data prioritized)"
  }]
}
```

## Benchmarking

The `benchmark_data/` directory contains a full SDRF-based benchmark pipeline. It evaluates extraction accuracy by comparing LLM outputs against SDRF ground truth using multi-tier semantic matching.

```bash
# Run benchmark on the 12-PXD test set
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir test_set

# Run on new test set with integration agent (18 PXDs)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir new_test_set \
  --integrate --runassessor-dir /path/to/aggregated_results \
  --skip-conversion

# Re-evaluate without re-extracting (for prompt/evaluation changes)
CUDA_VISIBLE_DEVICES="" python benchmark_data/run_sdrf_benchmark.py \
  --input-dir test_set --skip-extraction --skip-conversion
```

See [`benchmark_data/README.md`](benchmark_data/README.md) for full documentation.

## Project Structure

```
extraction_framework/
├── config.yaml             # Centralized configuration
├── main.py                 # Pipeline entry point
├── setup.sh                # First-run setup script
├── run_pipeline.sh         # Pipeline runner with checks
├── requirements.txt        # Python dependencies
├── agents/                 # Extraction agents
│   ├── biological_agent.py
│   ├── technical_agent.py
│   ├── experimental_agent.py
│   ├── integration_agent.py
│   └── normalization_agent.py
├── core/                   # Core modules
│   ├── extractor.py
│   ├── llm.py
│   └── prompts.py
├── normalization/          # Ontology normalization
│   ├── config.py
│   ├── normalizer.py
│   ├── ontology.py
│   ├── index.py
│   ├── download.py
│   └── build_index.py
├── benchmark_data/         # SDRF benchmark pipeline
│   ├── run_sdrf_benchmark.py
│   ├── sdrf_to_golden.py
│   ├── annotation_to_golden.py
│   ├── matched/            # 107 train PXDs (SDRF + manuscript)
│   ├── test_set/           # 12 test PXDs
│   └── new_test_set/       # 18 new test PXDs (annotation JSON)
├── validation/             # Output validation
│   └── validator.py
├── ontologies/             # Ontology files (gitignored)
└── ontology_cache/         # Cached indices (gitignored)
```

## Troubleshooting

### "Module not found" errors

Make sure you've activated the virtual environment:
```bash
source venv/bin/activate
```

### Ontology download fails

Some ontology servers may be temporarily unavailable. Try:
```bash
# Retry failed downloads
python -m normalization.download

# Check which ontologies exist
python -m normalization.download --check
```

### Normalization is slow on first run

The first `--normalize` run builds embedding indices (~10 min). Subsequent runs use cached indices. To pre-build:
```bash
python -m normalization.build_index
```

### GPU out of memory

Disable GPU for embeddings in `config.yaml`:
```yaml
normalization:
  use_gpu: false
```
or via code:
```python
# In your script
from normalization.config import NormalizationConfig
config = NormalizationConfig(use_gpu=False)
```

### Installation fails on "faiss"

If `faiss-cpu` fails to install, ensure you have a compatible Python version (3.8-3.11 recommended). You can fallback to the legacy backend by editing `config.yaml`:
```yaml
normalization:
  backend: "sklearn"  # Slower but fewer dependencies
```

### Ontology term not found for abbreviated species names

Standard forms like `p.falciparum` and `Plasmodium.falciparum` are injected automatically. If a new abbreviation is still missed, register it at runtime:

```python
agent.register_synonym(
    synonym="p.falciparum",
    node_name="Plasmodium falciparum",
    entity_type="species",
)
```

The synonym is added to the live index immediately and persisted to `ontology_cache/custom_synonyms.json` for future runs. If the issue persists across all terms, the index may be stale — rebuild it:

```bash
rm -rf ontology_cache/ && python -m normalization.build_index
```

## License

[Specify license]

## Citation

[Add citation if applicable]
