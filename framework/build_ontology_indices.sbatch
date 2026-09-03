#!/bin/bash
#SBATCH --job-name=ontology-index
#SBATCH --account=lennart_martens
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/framework/logs/ontology-index-%j.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
FRAMEWORK="$PROJECT/textmining/framework"

module load Python/3.11.3-GCCcore-12.3.0

source "$PROJECT/project_env.sh"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$HF_HOME" "$DOCETL_HOME_DIR" "$FRAMEWORK/logs"
cd "$FRAMEWORK"

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Python: $($PROJECT/llm_stuff/bin/python --version 2>&1)"
nvidia-smi

"$PROJECT/llm_stuff/bin/python" -m normalization.build_index

echo "Finished: $(date --iso-8601=seconds)"
