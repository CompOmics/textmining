#!/usr/bin/env bash
#SBATCH --job-name=pride-three-levels
#SBATCH --account=lennart_martens
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=three-levels-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PROJECT/llm_stuff/bin/python" -u "$PROJECT/textmining/final_analyses/PRIDEContextGain1737/compare_three_levels.py"
