#!/usr/bin/env bash
#SBATCH --job-name=qwen-proto-mv-audit
#SBATCH --account=lennart_martens
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
PROJECT="${HAMLET_ROOT:?Set HAMLET_ROOT to the HAMLET project directory}"
source "$PROJECT/project_env.sh"
module load Python/3.11.3-GCCcore-12.3.0
"$PROJECT/llm_stuff/bin/python" "$PROJECT/textmining/framework/audit_protocol_multivalue_repair.py"
