# Additive staircase analysis

This directory is isolated from the production pipeline. It freezes the
master prompt and the three agent prompts, derives immutable S1-S5 arm
configs, restricts inputs to Title+Abstract+Methods, and retains runtime configs,
parsed DocETL responses, checksums, timings, QC flags, and S0 BRAT-to-field output.

S0 uses the single-master-prompt BRAT configuration. Its raw `.ann` files are validated against
the exact model input and then mapped to the common field schema. Exact spans and
uniquely realignable surfaces receive resolved offsets; repeated or absent surfaces
remain explicitly flagged without guessing. Offset quality is reported separately
and does not erase the model's predicted value. Unmapped labels are retained rather
than guessed. S1-S4 do not normalize; S5 alone runs ontology
normalization. PRIDE integration is disabled for all primary arms.

The 18 headed test manuscripts are deterministically restricted to TITLE, ABSTRACT,
and METHODS. The other 12 benchmark manuscripts are already supplied as unheaded,
pre-restricted abstract/methods text; they are preserved byte-for-byte and explicitly
identified as `PRE_RESTRICTED_UNHEADED` in the input manifest.

All inference uses the project `llm_stuff` environment, project-backed caches,
temperature zero, BF16, cache bypass, thinking disabled, and model-specific vLLM
tool parsers. A run is complete only when it contains a `SUCCESS` marker.

The figures and headline metrics in `benchmark_results/` use the SDRF benchmark
and report three-run means with sample standard deviations.
The supported Figure 3 panels and their source tables are under
`figures/figure3/`; the missing leave-one-out panel is documented there.

## SDRF benchmark

The matcher is located at `agentic-metadata/benchmark/semantic_matcher.py`. The
benchmark uses a semantic classification threshold of 0.70 and a final metric threshold
of 0.50. Compatibility inputs are generated as relative symlinks (plus small S0
agent adapters) under `benchmark_inputs/`. Per-run reports are retained
under `benchmark_runs/`, while presentation-ready aggregate
results and figures are written to `benchmark_results/`.

Prepare and submit it with:

```bash
module load Python/3.11.3-GCCcore-12.3.0
llm_stuff/bin/python textmining/final_analyses/Staircase/scripts/prepare_benchmark.py \
  --run qwen3_8_27b=textmining/final_analyses/Staircase/outputs/test/qwen3_8_27b/job_940301 \
  --run gemma4_31b=textmining/final_analyses/Staircase/outputs/test/gemma4_31b/job_940302 \
  --run glm4_7=textmining/final_analyses/Staircase/outputs/test/glm4_7/job_948623 \
  --output textmining/final_analyses/Staircase/benchmark_inputs
sbatch textmining/final_analyses/Staircase/run_glm_benchmark.sbatch
```

For variability estimates, the first completed extraction is replicate 1.
Two further full inference runs per model are benchmarked independently, then
`scripts/aggregate_replicates.py` reports the mean and sample standard
deviation (`ddof=1`) and generates the staircase figure.
