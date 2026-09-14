#!/usr/bin/env bash
#SBATCH --job-name=control-cache-recovery
#SBATCH --account=lennart_martens
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=recovery-%j.out
set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$PROJECT/project_cache/runtime/control-recovery-$SLURM_JOB_ID/tmp"
export XDG_CACHE_HOME="$PROJECT/project_cache/runtime/control-recovery-$SLURM_JOB_ID/xdg"
export DOCETL_HOME_DIR="$PROJECT/project_cache/docetl/control-recovery-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$DOCETL_HOME_DIR"
"$PROJECT/llm_stuff/bin/python" -u "$PROJECT/textmining/final_analyses/PRIDEContextGain1737/recover_control.py"
