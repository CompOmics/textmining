#!/usr/bin/env bash
#SBATCH --job-name=stair-glm47
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=300G
#SBATCH --time=12:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/final_analyses/Staircase/logs/%x-%A.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
STAIRCASE="$PROJECT/textmining/final_analyses/Staircase"
SIF="$PROJECT/containers/vllm-openai-v0.27.1.sif"
MODEL_ID="zai-org/GLM-4.7"
MODEL_TAG="glm4_7"
MODEL_CONFIG="$PROJECT/textmining/framework/configs/glm4_7_vllm.yaml"
: "${PHASE:?PHASE must be smoke or test}"

case "$PHASE" in
  smoke) INPUT="$STAIRCASE/inputs/dev_smoke" ;;
  test) INPUT="$STAIRCASE/inputs/test_30" ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac

source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
TASK_ID="${SLURM_JOB_ID}"
export DOCETL_HOME_DIR="$PROJECT/project_cache/docetl/staircase/$MODEL_TAG/$TASK_ID"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/staircase/$MODEL_TAG/$TASK_ID/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/project_cache/runtime/staircase/$MODEL_TAG/$TASK_ID/torchinductor"
export TRITON_CACHE_DIR="$PROJECT/project_cache/runtime/staircase/$MODEL_TAG/$TASK_ID/triton"
mkdir -p "$DOCETL_HOME_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

PORT=$((8400 + SLURM_JOB_ID % 200))
BASE_URL="http://127.0.0.1:$PORT/v1"
MODEL_CACHE_DIR="$HUGGINGFACE_HUB_CACHE/models--${MODEL_ID//\//--}/snapshots"
mapfile -t SNAPSHOTS < <(find "$MODEL_CACHE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
[[ "${#SNAPSHOTS[@]}" -eq 1 ]] || {
  echo "expected one downloaded snapshot for $MODEL_ID, found ${#SNAPSHOTS[@]}" >&2
  exit 1
}
MODEL_PATH="${SNAPSHOTS[0]}"
RUN_ROOT="$STAIRCASE/outputs/$PHASE/$MODEL_TAG/job_$SLURM_JOB_ID"
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing existing run root $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
VLLM_LOG="$RUN_ROOT/vllm.log"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

date --iso-8601=seconds > "$RUN_ROOT/started.txt"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
env | sort > "$RUN_ROOT/environment.txt"
"$PROJECT/llm_stuff/bin/pip" freeze > "$RUN_ROOT/llm_stuff-pip-freeze.txt"
printf '%s\n' "$MODEL_PATH" > "$RUN_ROOT/model_snapshot.txt"

apptainer exec --nv --bind "$PROJECT:$PROJECT" "$SIF" vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --generation-config vllm \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --host 127.0.0.1 --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for attempt in $(seq 1 360); do
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 300 "$VLLM_LOG" >&2; exit 1; }
  if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  [[ "$attempt" -lt 360 ]] || { echo "vLLM startup timeout" >&2; exit 1; }
  sleep 10
done
"$PROJECT/llm_stuff/bin/python" "$STAIRCASE/scripts/check_tool_call.py" \
  --base-url "$BASE_URL" --model "$MODEL_ID" --tool-choice auto \
  > "$RUN_ROOT/tool_call_gate.json"

EXPECTED=$(jq -r .count "$INPUT/manifest.json")
for ARM in S0 S1 S2 S3 S4 S5; do
  ARM_START=$(date +%s)
  OUT="$RUN_ROOT/$ARM"
  export DOCETL_TRUNCATION_LOG="$OUT/truncations.tsv"
  if [[ "$ARM" == S0 ]]; then
    "$PROJECT/llm_stuff/bin/python" "$STAIRCASE/scripts/run_s0.py" \
      --pipeline "$STAIRCASE/arms/S0/pipeline_master_prompt.yaml" --input "$INPUT" \
      --output "$OUT" --config "$MODEL_CONFIG" --model-tag "$MODEL_TAG" \
      --base-url "$BASE_URL" --timeout 1800
    FOUND=$(find "$OUT/field_adapter" -type f -name '*.json' | wc -l)
  else
    NORMALIZE=(--no-normalize)
    [[ "$ARM" == S5 ]] && NORMALIZE=()
    "$PROJECT/llm_stuff/bin/python" "$STAIRCASE/scripts/run_docetl_staircase.py" \
      --input "$INPUT" --output "$OUT" --config "$MODEL_CONFIG" \
      --pipeline-dir "$STAIRCASE/arms/$ARM" --base-url "$BASE_URL" \
      --agents biological technical experimental --bypass-cache --no-integrate --no-confidence \
      --model-tag "$MODEL_TAG" "${NORMALIZE[@]}"
    FOUND=$(find "$OUT/BiologicalAgent" "$OUT/TechnicalAgent" "$OUT/ExperimentalDesignAgent" \
      -type f -name '*.json' | wc -l)
    [[ "$FOUND" -eq $((EXPECTED * 3)) ]] || { echo "$ARM incomplete raw count $FOUND" >&2; exit 1; }
    if [[ "$ARM" == S5 ]]; then
      NORMALIZED=$(find "$OUT/NormalizedAgent" -type f -name '*.json' | wc -l)
      [[ "$NORMALIZED" -eq $((EXPECTED * 3)) ]] || {
        echo "$ARM incomplete normalized count $NORMALIZED" >&2
        exit 1
      }
    fi
  fi
  [[ "$ARM" != S0 || "$FOUND" -eq "$EXPECTED" ]] || {
    echo "S0 incomplete count $FOUND" >&2
    exit 1
  }
  printf '%s\t%s\t%s\n' "$ARM" "$(( $(date +%s) - ARM_START ))" "$FOUND" >> "$RUN_ROOT/arm_timing.tsv"
done

date --iso-8601=seconds > "$RUN_ROOT/completed.txt"
touch "$RUN_ROOT/SUCCESS"
