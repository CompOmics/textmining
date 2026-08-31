#!/usr/bin/env bash
#SBATCH --job-name=gemma4-runtime-check
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/framework/logs/%x-%j.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
FRAMEWORK="$PROJECT/textmining/framework"
source "$PROJECT/project_env.sh"
source "$FRAMEWORK/configs/gemma4_runtime.env"
module load Python/3.11.3-GCCcore-12.3.0

TASK_ID="${SLURM_JOB_ID}"
CHECK_DIR="$FRAMEWORK/runtime_checks/gemma4_31b/$TASK_ID"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/gemma4_31b/$TASK_ID/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/project_cache/runtime/gemma4_31b/$TASK_ID/torchinductor"
export TRITON_CACHE_DIR="$PROJECT/project_cache/runtime/gemma4_31b/$TASK_ID/triton"
mkdir -p "$CHECK_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FRAMEWORK/logs"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export APPTAINERENV_PYTHONPATH="$HAMLET_VLLM_PYTHONPATH"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_CACHE_DIR="$HUGGINGFACE_HUB_CACHE/models--${MODEL_ID//\//--}/snapshots"
mapfile -t MODEL_SNAPSHOTS < <(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#MODEL_SNAPSHOTS[@]}" -ne 1 ]]; then
    echo "Expected exactly one downloaded snapshot for $MODEL_ID; found ${#MODEL_SNAPSHOTS[@]}" >&2
    exit 1
fi
MODEL_PATH="${MODEL_SNAPSHOTS[0]}"

PORT=8000
SERVER_LOG="$CHECK_DIR/vllm.log"
START_EPOCH=$(date +%s)

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Runtime validation only: no DocETL or manuscript inputs"
echo "Model: $MODEL_ID"
echo "Snapshot: $MODEL_PATH"
echo "Container: $VLLM_SIF"
echo "Compatibility layer: $HAMLET_VLLM_PYTHONPATH"
nvidia-smi

apptainer exec --nv --bind "$PROJECT:$PROJECT" "$VLLM_SIF" vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --language-model-only \
    --attention-backend "$HAMLET_VLLM_ATTENTION_BACKEND" \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
VLLM_PID=$!

for attempt in $(seq 1 150); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready" >&2
        tail -n 200 "$SERVER_LOG" >&2
        exit 1
    fi
    if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
        READY_EPOCH=$(date +%s)
        break
    fi
    if [[ "$attempt" -eq 150 ]]; then
        echo "Timed out waiting for vLLM" >&2
        tail -n 200 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 10
done

curl --fail --silent "http://127.0.0.1:$PORT/v1/models" > "$CHECK_DIR/models.json"
curl --fail --silent \
    -H 'Content-Type: application/json' \
    -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"Reply with exactly: GEMMA4_OK"}],"temperature":0,"max_tokens":16}' \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$CHECK_DIR/completion.json"
FINISH_EPOCH=$(date +%s)

"$PROJECT/llm_stuff/bin/python" - "$CHECK_DIR/completion.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    response = json.load(handle)
assert response.get("choices"), response
content = response["choices"][0]["message"].get("content", "")
assert content.strip(), response
print(f"Completion: {content.strip()}")
PY

{
    printf 'phase\tseconds\n'
    printf 'startup_to_health\t%s\n' "$((READY_EPOCH - START_EPOCH))"
    printf 'total_with_completion\t%s\n' "$((FINISH_EPOCH - START_EPOCH))"
} > "$CHECK_DIR/timing.tsv"

cat "$CHECK_DIR/timing.tsv"
echo "GEMMA4_RUNTIME_CHECK=PASS"
