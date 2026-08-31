#!/usr/bin/env bash
#SBATCH --job-name=staircase-repbench
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_b300_96C_192T_3TB_co_pi
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/final_analyses/Staircase/logs/%x-%A.out

set -euo pipefail

: "${QWEN_JOB_2:?QWEN_JOB_2 is required}"
: "${GEMMA_JOB_2:?GEMMA_JOB_2 is required}"
: "${QWEN_JOB_3:?QWEN_JOB_3 is required}"
: "${GEMMA_JOB_3:?GEMMA_JOB_3 is required}"

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
STAIRCASE="$PROJECT/textmining/final_analyses/Staircase"
PYTHON="$PROJECT/llm_stuff/bin/python"
GOLD="$PROJECT/textmining/agentic-benchmark/benchmark_data/SDRFS_github"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for REP in 2 3; do
  QWEN_VAR="QWEN_JOB_${REP}"
  GEMMA_VAR="GEMMA_JOB_${REP}"
  QWEN_JOB="${!QWEN_VAR}"
  GEMMA_JOB="${!GEMMA_VAR}"
  INPUTS="$STAIRCASE/benchmark_inputs/replicate_${REP}"
  RESULTS="$STAIRCASE/benchmark_runs/replicate_${REP}"
  "$PYTHON" "$STAIRCASE/scripts/prepare_benchmark.py" \
    --run "qwen3_8_27b=$STAIRCASE/outputs/test/qwen3_8_27b/job_${QWEN_JOB}" \
    --run "gemma4_31b=$STAIRCASE/outputs/test/gemma4_31b/job_${GEMMA_JOB}" \
    --output "$INPUTS"
  "$PYTHON" "$STAIRCASE/scripts/run_benchmark.py" \
    --inputs "$INPUTS" --gold-dir "$GOLD" --output "$RESULTS" \
    --model qwen3_8_27b --model gemma4_31b
done

"$PYTHON" "$STAIRCASE/scripts/aggregate_replicates.py" \
  --replicate "replicate_1=$STAIRCASE/benchmark_runs/index.json" \
  --replicate "replicate_2=$STAIRCASE/benchmark_runs/replicate_2/index.json" \
  --replicate "replicate_3=$STAIRCASE/benchmark_runs/replicate_3/index.json" \
  --output "$STAIRCASE/benchmark_results"
