"""Benchmark scoring for the extraction framework.

Holds the five-level matcher used to score extracted metadata against
SDRF-derived gold standards. Imported as `benchmark.semantic_matcher` by
agentic-benchmark/benchmark_data/run_sdrf_benchmark.py and by
framework/tests/test_hierarchical_matching.py, both of which put `framework/`
on sys.path, which is why this package sits beside `core/` rather than at the
repository root.
"""
