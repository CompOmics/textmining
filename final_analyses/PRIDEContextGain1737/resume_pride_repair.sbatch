#!/usr/bin/env bash
#SBATCH --job-name=pride-last-retry
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=pride-last-retry-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
BASE="$PROJECT/textmining/final_analyses/PRIDEContextGain1737"
RUN="$BASE/pride_only_1737"
export PRIDE_REPAIR_STAGE="$RUN/repair_966346"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
# Unix-domain socket paths must fit sockaddr_un (107 characters).
export TMPDIR="$PROJECT/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export MODEL_ID=Qwen/Qwen3.8-27B MODEL_TAG=qwen3_8_27b_pride_only_multivalue_1737 DOCETL_MULTI_VALUE=1
export MODEL_CONFIG="$RUN/framework/configs/qwen3_8_27b_vllm_repair.yaml"
export VLLM_SIF="$PROJECT/containers/vllm-openai-v0.27.1.sif" VLLM_MAX_MODEL_LEN=65536
export INPUT_PATH="$PRIDE_REPAIR_STAGE/retry_inputs/technical"
export OUTPUT_PATH="$PRIDE_REPAIR_STAGE/retry_outputs/technical"
export DOCETL_AGENTS=technical
unset SMOKE_SHARD_ROOT SMOKE_OUTPUT_ROOT HAMLET_VLLM_PYTHONPATH VLLM_PYTHONPATH HAMLET_VLLM_LANGUAGE_MODEL_ONLY HAMLET_VLLM_ATTENTION_BACKEND
bash "$RUN/worker.sh" > "$PRIDE_REPAIR_STAGE/retry-technical-$SLURM_JOB_ID.log" 2>&1
"$PROJECT/llm_stuff/bin/python" -u "$BASE/repair_pride_only.py" finish
"$PROJECT/llm_stuff/bin/python" -u "$BASE/plot_three_sources.py"
