#!/usr/bin/env bash
#SBATCH --job-name=qwen-manuscript-control
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --nodelist=p-computegpu-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=12
#SBATCH --gpus-per-node=b300:8
#SBATCH --mem-per-gpu=256G
#SBATCH --time=08:00:00
#SBATCH --output=control-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
CONTROL="$PROJECT/textmining/final_analyses/PRIDEContextGain1737/manuscript_control_1737"
test -f "$CONTROL/PREPARED"
test ! -e "$CONTROL/COMPLETE"
printf '%s\n' "$SLURM_JOB_ID" > "$CONTROL/SUBMITTED_JOB_$SLURM_JOB_ID"
source "$PROJECT/project_env.sh"
export MODEL_ID=Qwen/Qwen3.8-27B
export MODEL_TAG=qwen3_8_27b_manuscript_only_multivalue_1737
export MODEL_CONFIG="$CONTROL/framework/configs/qwen3_8_27b_vllm.yaml"
export SMOKE_SHARD_ROOT="$CONTROL/inputs"
export SMOKE_OUTPUT_ROOT="$CONTROL/outputs"
export DOCETL_MULTI_VALUE=1
export DOCETL_AGENTS='biological technical experimental'
export VLLM_MAX_MODEL_LEN=65536
export VLLM_SIF="$PROJECT/containers/vllm-openai-v0.27.1.sif"
unset HAMLET_VLLM_PYTHONPATH VLLM_PYTHONPATH HAMLET_VLLM_LANGUAGE_MODEL_ONLY HAMLET_VLLM_ATTENTION_BACKEND
export CONTROL
mkdir -p "$CONTROL/logs"
echo "Starting eight BF16 Qwen workers on $(hostname) at $(date --iso-8601=seconds)"
env -u SLURM_CPUS_PER_GPU srun --ntasks=8 --cpus-per-task=12 --gpus-per-task=b300:1 --gpu-bind=single:1 \
    --output="$CONTROL/logs/worker-%j-%t.out" \
    bash -c 'export SLURM_ARRAY_TASK_ID="$SLURM_PROCID"; export SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID"; bash "$CONTROL/worker.sh"'
for shard in {0..7}; do test -f "$CONTROL/outputs/shard_$shard/SUCCESS"; done
touch "$CONTROL/COMPLETE"
echo "All eight shards complete at $(date --iso-8601=seconds)"
