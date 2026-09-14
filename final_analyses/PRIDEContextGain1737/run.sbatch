#!/usr/bin/env bash
#SBATCH --job-name=pride-context-audit
#SBATCH --account=lennart_martens
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=audit-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
"$PROJECT/llm_stuff/bin/python" -u "$PROJECT/textmining/final_analyses/PRIDEContextGain1737/compare.py"
