# Gemma 4 runtime on B300

## Validated stack

- Model: `google/gemma-4-31B-it`
- Precision: BF16 weights and BF16 KV cache (`quantization=None`, `kv_cache_dtype=auto`)
- Hardware: one NVIDIA B300 SXM6
- Context: 32,768 tokens
- Server: vLLM 0.27.1
- Compatibility pin: Transformers 5.14.1, loaded from the project-local Python layer
- Mode: language-model-only; multimodal towers disabled for text mining
- Attention backend: `TRITON_ATTN`, selected explicitly because Gemma 4 has heterogeneous 256/512-dimensional attention heads

The base vLLM SIF remains immutable. Qwen continues to use the base image's
Transformers 5.15.0; only Gemma jobs source `configs/gemma4_runtime.env`.

## Validation

Slurm job `940106` completed successfully on 2026-08-25:

- Exit code: `0:0`
- `/health`: HTTP 200
- `/v1/models`: HTTP 200 and advertised 32,768-token context
- `/v1/chat/completions`: HTTP 200
- Fixed test response: `GEMMA4_OK`
- Startup to health: 300 seconds
- Total through the fixed completion: 302 seconds
- No DocETL pipeline or manuscript input was used

Validation artifacts are stored under `runtime_checks/gemma4_31b/940106/`.

## Future DocETL use

Before submitting a Gemma annotation job, source:

```bash
source textmining/framework/configs/gemma4_runtime.env
```

Then submit `smoke_docetl_b300.sbatch` with explicit `INPUT_PATH` and
`OUTPUT_PATH`. The job script automatically applies the compatibility layer,
text-only mode, and Triton attention backend when those exported settings are
present. Do not run Gemma through the base image without the runtime settings.
