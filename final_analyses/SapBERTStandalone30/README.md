# SapBERT standalone-30 benchmark

This experiment re-scores the frozen standalone Qwen 3.8 27B extraction of the
30-paper test set. It compares the historical SciBERT semantic encoder with
SapBERT while preserving all other benchmark behavior:

- SDRF gold files and manuscript-extractable field filtering
- exact, normalized, ontology, hierarchical, and semantic matching order
- semantic threshold `0.70`
- historical final weighted-metric threshold `0.50`
- normalized standalone extraction outputs (no new inference)

The source annotations are not copied or modified. `prepare_inputs.py` creates
a legacy-layout adapter made from relative symbolic links, and the two result
trees are written below `results/`.

