#!/usr/bin/env bash
#SBATCH --job-name=qwen-prod-pride-1737
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --array=0-7
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/framework/logs/%x-%A_%a.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
FRAMEWORK="$PROJECT/textmining/framework"
MODEL_ID=Qwen/Qwen3.8-27B
MODEL_TAG=qwen3_8_27b_abstract_methods_pride_1737
MODEL_CONFIG="$FRAMEWORK/configs/qwen3_8_27b_vllm.yaml"
SIF="$PROJECT/containers/vllm-openai-v0.27.1.sif"
INPUT_ROOT="$PROJECT/textmining/production_inputs/abstract_methods_pride_1737_8shards"
OUTPUT_ROOT="$FRAMEWORK/production_outputs/$MODEL_TAG"
INPUT="$INPUT_ROOT/shard_$SLURM_ARRAY_TASK_ID"
OUTPUT="$OUTPUT_ROOT/shard_$SLURM_ARRAY_TASK_ID"
JOB_START_EPOCH=$(date +%s)

source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0

TASK_LOG_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

PORT=$((8000 + SLURM_ARRAY_TASK_ID))
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
MODEL_CACHE_DIR="$HUGGINGFACE_HUB_CACHE/models--${MODEL_ID//\//--}/snapshots"
mapfile -t MODEL_SNAPSHOTS < <(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#MODEL_SNAPSHOTS[@]}" -ne 1 ]]; then
    echo "Expected exactly one downloaded snapshot for $MODEL_ID; found ${#MODEL_SNAPSHOTS[@]}" >&2
    exit 1
fi
MODEL_PATH="${MODEL_SNAPSHOTS[0]}"

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
echo "Input: $INPUT"
echo "Output: $OUTPUT"

apptainer exec --nv --bind "$PROJECT:$PROJECT" "$SIF" vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 65536 \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for attempt in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready" >&2
        tail -n 200 "$VLLM_LOG" >&2
        exit 1
    fi
    if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
        VLLM_READY_EPOCH=$(date +%s)
        echo "vLLM ready after $attempt checks"
        break
    fi
    if [[ "$attempt" -eq 120 ]]; then
        echo "Timed out waiting for vLLM" >&2
        exit 1
    fi
    sleep 10
done

cd "$FRAMEWORK"
PIPELINE_START_EPOCH=$(date +%s)
MANUSCRIPT_COUNT=$(find -L "$INPUT" -type f -name '*.txt' | wc -l)
"$PROJECT/llm_stuff/bin/python" docetl_pipeline/run_docetl.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --config "$MODEL_CONFIG" \
    --agents biological technical experimental \
    --bypass-cache \
    --no-integrate \
    --model-tag "$MODEL_TAG"
PIPELINE_FINISH_EPOCH=$(date +%s)

for AGENT in BiologicalAgent TechnicalAgent ExperimentalDesignAgent; do
    RAW_COUNT=$(find "$OUTPUT/$AGENT" -type f -name '*.json' | wc -l)
    NORMALIZED_COUNT=$(find "$OUTPUT/NormalizedAgent/$AGENT" -type f -name '*.json' | wc -l)
    if [[ "$RAW_COUNT" -ne "$MANUSCRIPT_COUNT" || "$NORMALIZED_COUNT" -ne "$MANUSCRIPT_COUNT" ]]; then
        echo "Incomplete $AGENT: expected $MANUSCRIPT_COUNT, raw=$RAW_COUNT normalized=$NORMALIZED_COUNT" >&2
        exit 1
    fi
done

{
    printf 'phase\tseconds\n'
    printf 'vllm_startup\t%s\n' "$((VLLM_READY_EPOCH - JOB_START_EPOCH))"
    printf 'docetl_and_normalization\t%s\n' "$((PIPELINE_FINISH_EPOCH - PIPELINE_START_EPOCH))"
    printf 'total\t%s\n' "$((PIPELINE_FINISH_EPOCH - JOB_START_EPOCH))"
    printf 'manuscripts\t%s\n' "$MANUSCRIPT_COUNT"
} > "$OUTPUT/timing.tsv"
touch "$OUTPUT/SUCCESS"
cat "$OUTPUT/timing.tsv"
echo "Finished: $(date --iso-8601=seconds)"
