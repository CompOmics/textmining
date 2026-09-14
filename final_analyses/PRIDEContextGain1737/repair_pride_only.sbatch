#!/usr/bin/env bash
#SBATCH --job-name=pride-only-repair
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=pride-repair-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
BASE="$PROJECT/textmining/final_analyses/PRIDEContextGain1737"
RUN="$BASE/pride_only_1737"
STAGE="$RUN/repair_$SLURM_JOB_ID"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export TMPDIR="$PROJECT/project_cache/runtime/pride-repair-$SLURM_JOB_ID/tmp"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/pride-repair-$SLURM_JOB_ID/xdg"
export DOCETL_HOME_DIR="$PROJECT/project_cache/docetl/pride-repair-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$DOCETL_HOME_DIR"
"$PROJECT/llm_stuff/bin/python" -u "$BASE/repair_pride_only.py" recover
export MODEL_ID=Qwen/Qwen3.8-27B MODEL_TAG=qwen3_8_27b_pride_only_multivalue_1737 DOCETL_MULTI_VALUE=1
export MODEL_CONFIG="$RUN/framework/configs/qwen3_8_27b_vllm_repair.yaml"
export VLLM_SIF="$PROJECT/containers/vllm-openai-v0.27.1.sif" VLLM_MAX_MODEL_LEN=65536
unset SMOKE_SHARD_ROOT SMOKE_OUTPUT_ROOT HAMLET_VLLM_PYTHONPATH VLLM_PYTHONPATH HAMLET_VLLM_LANGUAGE_MODEL_ONLY HAMLET_VLLM_ATTENTION_BACKEND
for kind in biological technical experimental; do
    if [[ -d "$STAGE/retry_inputs/$kind" ]]; then
        export INPUT_PATH="$STAGE/retry_inputs/$kind" OUTPUT_PATH="$STAGE/retry_outputs/$kind" DOCETL_AGENTS="$kind"
        bash "$RUN/worker.sh" > "$STAGE/retry-$kind.log" 2>&1
    fi
done
"$PROJECT/llm_stuff/bin/python" -u "$BASE/repair_pride_only.py" finish
"$PROJECT/llm_stuff/bin/python" -u "$BASE/plot_three_sources.py"
