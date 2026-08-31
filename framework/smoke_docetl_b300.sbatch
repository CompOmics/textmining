#!/usr/bin/env bash
#SBATCH --job-name=docetl-smoke
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/framework/logs/docetl-smoke-%x-%A_%a.out

set -euo pipefail

JOB_START_EPOCH=$(date +%s)

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
FRAMEWORK="$PROJECT/textmining/framework"
SIF="${VLLM_SIF:-$PROJECT/containers/vllm-openai-v0.27.1.sif}"
INPUT="${INPUT_PATH:-$PROJECT/textmining/agentic-benchmark/benchmark_data/test_set}"
if [[ -n "${SMOKE_SHARD_ROOT:-}" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?SMOKE_SHARD_ROOT requires a Slurm array job}"
    INPUT="$SMOKE_SHARD_ROOT/shard_$SLURM_ARRAY_TASK_ID"
fi

: "${MODEL_ID:?Submit with --export=ALL,MODEL_ID=...,MODEL_TAG=...,MODEL_CONFIG=...}"
: "${MODEL_TAG:?Submit with --export=ALL,MODEL_ID=...,MODEL_TAG=...,MODEL_CONFIG=...}"
: "${MODEL_CONFIG:?Submit with --export=ALL,MODEL_ID=...,MODEL_TAG=...,MODEL_CONFIG=...}"

source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0

# DocETL uses a SQLite-backed disk cache that is not safe for concurrent
# writers. Give every Slurm array task a private project-backed cache.
TASK_LOG_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}"
export DOCETL_HOME_DIR="$PROJECT/project_cache/docetl/$MODEL_TAG/$TASK_LOG_ID"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/$MODEL_TAG/$TASK_LOG_ID/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/project_cache/runtime/$MODEL_TAG/$TASK_LOG_ID/torchinductor"
export TRITON_CACHE_DIR="$PROJECT/project_cache/runtime/$MODEL_TAG/$TASK_LOG_ID/triton"
mkdir -p "$DOCETL_HOME_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
# Optional project-local Python compatibility layer.  Gemma 4 currently needs
# Transformers 5.14.1 with the vLLM 0.27.1 image; leave unset for Qwen.
VLLM_LAYER="${HAMLET_VLLM_PYTHONPATH:-${VLLM_PYTHONPATH:-}}"
if [[ -n "$VLLM_LAYER" ]]; then
    export APPTAINERENV_PYTHONPATH="$VLLM_LAYER"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

# Array tasks may share one eight-GPU node, so each vLLM server needs its own port.
PORT=$((8000 + ${SLURM_ARRAY_TASK_ID:-0}))
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"

MODEL_CACHE_DIR="$HUGGINGFACE_HUB_CACHE/models--${MODEL_ID//\//--}/snapshots"
mapfile -t MODEL_SNAPSHOTS < <(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#MODEL_SNAPSHOTS[@]}" -ne 1 ]]; then
    echo "Expected exactly one downloaded snapshot for $MODEL_ID; found ${#MODEL_SNAPSHOTS[@]}" >&2
    exit 1
fi
MODEL_PATH="${MODEL_SNAPSHOTS[0]}"

OUTPUT="${OUTPUT_PATH:-$FRAMEWORK/smoke_outputs/$MODEL_TAG/test_set}"
if [[ -n "${SMOKE_OUTPUT_ROOT:-}" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?SMOKE_OUTPUT_ROOT requires a Slurm array job}"
    OUTPUT="$SMOKE_OUTPUT_ROOT/shard_$SLURM_ARRAY_TASK_ID"
fi
export DOCETL_TRUNCATION_LOG="$OUTPUT/truncations.tsv"
VLLM_LOG="$FRAMEWORK/logs/vllm-$MODEL_TAG-$TASK_LOG_ID.log"
mkdir -p "$OUTPUT" "$FRAMEWORK/logs"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Model: $MODEL_ID"
echo "vLLM image: $SIF"
echo "Offline snapshot: $MODEL_PATH"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
nvidia-smi

EXTRA_VLLM_ARGS=()
if [[ "${HAMLET_VLLM_LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
    EXTRA_VLLM_ARGS+=(--language-model-only)
fi
if [[ -n "${HAMLET_VLLM_ATTENTION_BACKEND:-}" ]]; then
    EXTRA_VLLM_ARGS+=(--attention-backend "$HAMLET_VLLM_ATTENTION_BACKEND")
fi

apptainer exec --nv --bind "$PROJECT:$PROJECT" "$SIF" vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-65536}" \
    --host 127.0.0.1 \
    --port "$PORT" \
    "${EXTRA_VLLM_ARGS[@]}" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for attempt in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready" >&2
        tail -n 200 "$VLLM_LOG" >&2
        exit 1
    fi
    if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "vLLM ready after $attempt checks"
        VLLM_READY_EPOCH=$(date +%s)
        break
    fi
    if [[ "$attempt" -eq 120 ]]; then
        echo "Timed out waiting for vLLM" >&2
        tail -n 200 "$VLLM_LOG" >&2
        exit 1
    fi
    sleep 10
done

cd "$FRAMEWORK"
PIPELINE_START_EPOCH=$(date +%s)
MANUSCRIPT_COUNT=$(find -L "$INPUT" -type f -name '*.txt' | wc -l)
read -r -a AGENT_ARGS <<< "${DOCETL_AGENTS:-biological technical experimental}"
EXTRA_DOCETL_ARGS=()
if [[ "${DOCETL_MULTI_VALUE:-0}" == "1" ]]; then
    EXTRA_DOCETL_ARGS+=(--multi-value)
fi
"$PROJECT/llm_stuff/bin/python" docetl_pipeline/run_docetl.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --config "$MODEL_CONFIG" \
    --agents "${AGENT_ARGS[@]}" \
    --bypass-cache \
    --no-integrate \
    --model-tag "$MODEL_TAG" \
    "${EXTRA_DOCETL_ARGS[@]}"

printf 'normalized_files='
find "$OUTPUT/NormalizedAgent" -type f -name '*.json' | wc -l
for AGENT_KEY in "${AGENT_ARGS[@]}"; do
    case "$AGENT_KEY" in
        biological) AGENT_DIR=BiologicalAgent ;;
        technical) AGENT_DIR=TechnicalAgent ;;
        experimental) AGENT_DIR=ExperimentalDesignAgent ;;
        *) echo "Unknown agent: $AGENT_KEY" >&2; exit 1 ;;
    esac
    RAW_COUNT=$(find "$OUTPUT/$AGENT_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)
    NORMALIZED_COUNT=$(find "$OUTPUT/NormalizedAgent/$AGENT_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)
    if [[ "$RAW_COUNT" -ne "$MANUSCRIPT_COUNT" || "$NORMALIZED_COUNT" -ne "$MANUSCRIPT_COUNT" ]]; then
        echo "Incomplete $AGENT_DIR: expected=$MANUSCRIPT_COUNT raw=$RAW_COUNT normalized=$NORMALIZED_COUNT" >&2
        exit 1
    fi
done

# A multi-value run is complete only when every field has the requested
# [{"value": ..., "evidence": ...}] shape, normalization preserves entry
# cardinality, and every evidence quote is present in the prepared input.
if [[ "${DOCETL_MULTI_VALUE:-0}" == "1" ]]; then
    "$PROJECT/llm_stuff/bin/python" "$PROJECT/textmining/audit_multivalue_outputs.py" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        > "$OUTPUT/multivalue_audit.json"
fi
PIPELINE_FINISH_EPOCH=$(date +%s)

{
    printf 'phase\tseconds\n'
    printf 'vllm_startup\t%s\n' "$((VLLM_READY_EPOCH - JOB_START_EPOCH))"
    printf 'docetl_and_normalization\t%s\n' "$((PIPELINE_FINISH_EPOCH - PIPELINE_START_EPOCH))"
    printf 'total\t%s\n' "$((PIPELINE_FINISH_EPOCH - JOB_START_EPOCH))"
    printf 'manuscripts\t%s\n' "$MANUSCRIPT_COUNT"
    printf 'pipeline_seconds_per_manuscript\t%s\n' "$(( (PIPELINE_FINISH_EPOCH - PIPELINE_START_EPOCH + MANUSCRIPT_COUNT - 1) / MANUSCRIPT_COUNT ))"
} > "$OUTPUT/timing.tsv"

cat "$OUTPUT/timing.tsv"
touch "$OUTPUT/SUCCESS"
echo "Finished: $(date --iso-8601=seconds)"
