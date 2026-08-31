#!/usr/bin/env bash
#SBATCH --job-name=staircase-benchmark
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/final_analyses/Staircase/logs/%x-%A.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
STAIRCASE="$PROJECT/textmining/final_analyses/Staircase"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PROJECT/llm_stuff/bin/python" "$STAIRCASE/scripts/run_benchmark.py" \
  --inputs "$STAIRCASE/benchmark_inputs" \
  --gold-dir "$PROJECT/textmining/agentic-benchmark/benchmark_data/SDRFS_github" \
  --output "$STAIRCASE/benchmark_runs" \
  --model qwen3_8_27b --model gemma4_31b
