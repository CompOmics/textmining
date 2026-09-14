#!/usr/bin/env bash
#SBATCH --job-name=sapbert-test30
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_h100_64C_128T_2TB_co_pi
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
EXPERIMENT="$PROJECT/textmining/final_analyses/SapBERTStandalone30"
mkdir -p "$EXPERIMENT/logs"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PROJECT/llm_stuff/bin/python" "$EXPERIMENT/scripts/prepare_inputs.py"
"$PROJECT/llm_stuff/bin/python" "$EXPERIMENT/scripts/run_benchmark.py"
