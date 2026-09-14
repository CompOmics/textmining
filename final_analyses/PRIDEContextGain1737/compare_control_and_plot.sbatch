#!/usr/bin/env bash
#SBATCH --job-name=pride-matched-figures
#SBATCH --account=lennart_martens
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=matched-figures-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$PROJECT/project_cache/runtime/matched-figures-$SLURM_JOB_ID/tmp"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/matched-figures-$SLURM_JOB_ID/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
"$PROJECT/llm_stuff/bin/python" -u "$PROJECT/textmining/final_analyses/PRIDEContextGain1737/compare_control_and_plot.py"
