#!/usr/bin/env bash
#SBATCH --job-name=qwen-renorm-v2
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --array=0-7
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/framework/logs/%x-%A_%a.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
FRAMEWORK="$PROJECT/textmining/framework"
SOURCE="$FRAMEWORK/production_outputs/qwen3_8_27b_abstract_methods_pride_multivalue_1737_validated"
DESTINATION="$FRAMEWORK/production_outputs/qwen3_8_27b_abstract_methods_pride_multivalue_1737_normalization_v2_20260828"

source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/llm_stuff/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/project_cache/runtime/qwen-renorm-v2/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID/torchinductor"
export TRITON_CACHE_DIR="$PROJECT/project_cache/runtime/qwen-renorm-v2/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FRAMEWORK/logs"

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$FRAMEWORK"
PYTHONPATH="$FRAMEWORK" python rerun_normalization_outputs.py \
    --source-root "$SOURCE" \
    --output-root "$DESTINATION" \
    --shard "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date --iso-8601=seconds)"
