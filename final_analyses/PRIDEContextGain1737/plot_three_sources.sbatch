#!/usr/bin/env bash
#SBATCH --job-name=pride-three-source-fig
#SBATCH --account=lennart_martens
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=three-source-fig-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
"$PROJECT/llm_stuff/bin/python" -u "$PROJECT/textmining/final_analyses/PRIDEContextGain1737/plot_three_sources.py"
