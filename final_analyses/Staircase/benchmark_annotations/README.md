# Staircase benchmark annotations

Self-contained snapshot of the annotations consumed by the Staircase benchmark.
The directory structure preserves model, replicate, staircase arm, PXD, and
agent identity. Symlinks from the runtime `benchmark_inputs` tree were
materialized so this snapshot remains usable in a fresh clone without the raw
model-output directories.

The snapshot contains Qwen 3.8 27B, Gemma 4 31B, and GLM 4.7 annotations for
S0-S5 across the three benchmark replicates. Runtime requests, logs, manuscript
inputs, model caches, and rendered figures are excluded.
