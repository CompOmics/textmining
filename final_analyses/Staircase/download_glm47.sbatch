#!/usr/bin/env bash
#SBATCH --job-name=download-glm47
#SBATCH --account=lennart_martens
#SBATCH --partition=cpu_vibai_co_pi
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/groups/cmb/lennart.martens/HAMLET/textmining/final_analyses/Staircase/logs/%x-%A.out

set -euo pipefail

PROJECT=/data/groups/cmb/lennart.martens/HAMLET
source "$PROJECT/project_env.sh"

date --iso-8601=seconds
"$PROJECT/download_models.sh" zai-org/GLM-4.7 --max-workers 2
date --iso-8601=seconds

MODEL_CACHE_DIR="$HUGGINGFACE_HUB_CACHE/models--zai-org--GLM-4.7"
SNAPSHOT_COUNT=$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | wc -l)
[[ "$SNAPSHOT_COUNT" -eq 1 ]] || { echo "expected one snapshot, found $SNAPSHOT_COUNT" >&2; exit 1; }
SNAPSHOT=$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d)
BROKEN_LINKS=$(find -L "$SNAPSHOT" -maxdepth 1 -type l | wc -l)
[[ "$BROKEN_LINKS" -eq 0 ]] || { echo "found $BROKEN_LINKS broken snapshot links" >&2; exit 1; }
# A successful resumable download can leave superseded .incomplete files from
# interrupted attempts. They are not referenced by the verified snapshot.
find "$MODEL_CACHE_DIR/blobs" -type f -name '*.incomplete' -delete
du -sh "$MODEL_CACHE_DIR"
touch "$MODEL_CACHE_DIR/DOWNLOAD_COMPLETE"
