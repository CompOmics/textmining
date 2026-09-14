#!/usr/bin/env bash
# Remove code and notebooks in MLMarker/Reprocessing_database that nothing in
# the manuscript analyses uses. Staged with git rm, so recoverable until commit.
# Review, then run from the repository root:
#   bash MLMarker/Reprocessing_database/cleanup_unused.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/MLMarker/Reprocessing_database"

# Old Ollama/Qwen3.5 extraction pipeline, superseded by framework/
git rm -r -q \
  agentic_metadata/core.py agentic_metadata/main.py agentic_metadata/__main__.py \
  agentic_metadata/__init__.py agentic_metadata/schemas.py agentic_metadata/export_tsv.py \
  agentic_metadata/fetch_manuscripts.py agentic_metadata/fetch_sdrfs.py agentic_metadata/config.yaml \
  agentic_metadata/ollama agentic_metadata/prompt_builder agentic_metadata/normalization \
  agentic_metadata/benchmark agentic_metadata/ontologies agentic_metadata/cache/ontology \
  agentic_metadata/results/extracted_metadata

# Executed notebook copies of pipeline scripts, exploration, and duplicates
git rm -q run_metadata/run_meta_name.ipynb run_metadata/run_meta_mlmarker.ipynb \
  notebooks/combined_metadata.ipynb notebooks/ml_marker_test.ipynb combine_metadata.py

# Ported to final_analyses/ExtendedTrainingSet/scripts/build_atlas.py
git rm -q finalize_dataset.ipynb

# Tracked bytecode
git rm -r -q --cached agentic_metadata/__pycache__ run_metadata/__pycache__ 2>/dev/null || true
find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Kept on purpose (decide separately):
#   agentic_metadata/results/manuscripts/   1,738 manuscript texts
#   agentic_metadata/cache/sdrf/            63 PRIDE SDRF files
#   agentic_metadata/results/agent_metadata.tsv, normalization_map.tsv
#   agentic_metadata/data/nsaf_diann.zip    gitignored quant matrix
echo "staged deletions: $(git status --short . | grep -c '^D')"
