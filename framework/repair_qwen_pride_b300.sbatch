#!/usr/bin/env bash
#SBATCH --job-name=qwen-pride-repair
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --array=0-7
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/framework/logs/%x-%A_%a.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
FRAMEWORK="$PROJECT/textmining/framework"
REPAIR_ROOT="$PROJECT/textmining/repair_inputs/qwen3_8_27b_abstract_methods_pride_1737_missing"
MANIFEST="$REPAIR_ROOT/task_manifest.tsv"
MODEL_ID=Qwen/Qwen3.8-27B
MODEL_TAG=qwen3_8_27b_abstract_methods_pride_1737
MODEL_CONFIG="$FRAMEWORK/configs/qwen3_8_27b_vllm_repair.yaml"
SIF="$PROJECT/containers/vllm-openai-v0.27.1.sif"
JOB_START_EPOCH=$(date +%s)

IFS=$'\t' read -r TASK_ID ORIGINAL_SHARD AGENT OUTPUT_DIR_NAME EXPECTED INPUT OUTPUT PXD_IDS \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
if [[ "$TASK_ID" != "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Manifest task mismatch: expected $SLURM_ARRAY_TASK_ID, got $TASK_ID" >&2
    exit 1
fi

source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0

TASK_LOG_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export DOCETL_HOME_DIR="$PROJECT/project_cache/docetl/qwen-pride-repair/$TASK_LOG_ID"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/qwen-pride-repair/$TASK_LOG_ID/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/project_cache/runtime/qwen-pride-repair/$TASK_LOG_ID/torchinductor"
export TRITON_CACHE_DIR="$PROJECT/project_cache/runtime/qwen-pride-repair/$TASK_LOG_ID/triton"
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

REPAIR_LOG_ROOT="$FRAMEWORK/production_outputs/qwen3_8_27b_abstract_methods_pride_1737/repair_status"
mkdir -p "$REPAIR_LOG_ROOT"
export DOCETL_TRUNCATION_LOG="$REPAIR_LOG_ROOT/truncations-task_$TASK_ID.tsv"
VLLM_LOG="$FRAMEWORK/logs/vllm-qwen-pride-repair-$TASK_LOG_ID.log"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Started: $(date --iso-8601=seconds)"
echo "Task: $TASK_ID original_shard=$ORIGINAL_SHARD agent=$AGENT expected=$EXPECTED"
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
"$PROJECT/llm_stuff/bin/python" docetl_pipeline/run_docetl.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --config "$MODEL_CONFIG" \
    --agents "$AGENT" \
    --bypass-cache \
    --no-integrate \
    --model-tag "$MODEL_TAG"
PIPELINE_FINISH_EPOCH=$(date +%s)

MISSING=0
while IFS= read -r SOURCE; do
    PXD=$(basename "$SOURCE" .txt)
    RAW_COUNT=$(find "$OUTPUT/$OUTPUT_DIR_NAME" -maxdepth 1 -type f -name "${PXD}_*.json" | wc -l)
    NORMALIZED_COUNT=$(find "$OUTPUT/NormalizedAgent/$OUTPUT_DIR_NAME" -maxdepth 1 -type f -name "${PXD}_*.json" | wc -l)
    if [[ "$RAW_COUNT" -lt 1 || "$NORMALIZED_COUNT" -lt 1 ]]; then
        echo "Still incomplete: $PXD raw=$RAW_COUNT normalized=$NORMALIZED_COUNT" >&2
        MISSING=$((MISSING + 1))
    fi
done < <(find -L "$INPUT" -maxdepth 1 -type f -name '*.txt' | sort)
if [[ "$MISSING" -ne 0 ]]; then
    echo "Repair task incomplete: $MISSING of $EXPECTED papers still missing" >&2
    exit 1
fi

{
    printf 'phase\tseconds\n'
    printf 'vllm_startup\t%s\n' "$((VLLM_READY_EPOCH - JOB_START_EPOCH))"
    printf 'repair_pipeline\t%s\n' "$((PIPELINE_FINISH_EPOCH - PIPELINE_START_EPOCH))"
    printf 'total\t%s\n' "$((PIPELINE_FINISH_EPOCH - JOB_START_EPOCH))"
    printf 'papers\t%s\n' "$EXPECTED"
} > "$REPAIR_LOG_ROOT/timing-task_$TASK_ID.tsv"
touch "$REPAIR_LOG_ROOT/SUCCESS-task_$TASK_ID"
cat "$REPAIR_LOG_ROOT/timing-task_$TASK_ID.tsv"
echo "Finished: $(date --iso-8601=seconds)"
