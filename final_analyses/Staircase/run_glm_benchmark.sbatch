#!/usr/bin/env bash
#SBATCH --job-name=stair-glm-bench
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
PYTHON="$PROJECT/llm_stuff/bin/python"
GOLD="$PROJECT/textmining/agentic-benchmark/benchmark_data/SDRFS_github"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

QWEN_JOBS=(940301 940605 940607)
GEMMA_JOBS=(940302 940621 940834)
GLM_JOBS=(948623 948624 948625)

for INDEX in 0 1 2; do
  REP=$((INDEX + 1))
  if [[ "$REP" -eq 1 ]]; then
    INPUTS="$STAIRCASE/benchmark_inputs"
    RESULTS="$STAIRCASE/benchmark_runs"
  else
    INPUTS="$STAIRCASE/benchmark_inputs/replicate_${REP}"
    RESULTS="$STAIRCASE/benchmark_runs/replicate_${REP}"
  fi
  "$PYTHON" "$STAIRCASE/scripts/prepare_benchmark.py" \
    --run "qwen3_8_27b=$STAIRCASE/outputs/test/qwen3_8_27b/job_${QWEN_JOBS[$INDEX]}" \
    --run "gemma4_31b=$STAIRCASE/outputs/test/gemma4_31b/job_${GEMMA_JOBS[$INDEX]}" \
    --run "glm4_7=$STAIRCASE/outputs/test/glm4_7/job_${GLM_JOBS[$INDEX]}" \
    --output "$INPUTS"
  "$PYTHON" "$STAIRCASE/scripts/run_benchmark.py" \
    --inputs "$INPUTS" --gold-dir "$GOLD" --output "$RESULTS" \
    --model glm4_7 --append
done

"$PYTHON" "$STAIRCASE/scripts/aggregate_replicates.py" \
  --replicate "replicate_1=$STAIRCASE/benchmark_runs/index.json" \
  --replicate "replicate_2=$STAIRCASE/benchmark_runs/replicate_2/index.json" \
  --replicate "replicate_3=$STAIRCASE/benchmark_runs/replicate_3/index.json" \
  --output "$STAIRCASE/benchmark_results"

"$PYTHON" "$STAIRCASE/scripts/build_figure3.py"
