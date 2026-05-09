#!/bin/bash
#SBATCH --job-name=streaming-rls-glm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

set -euo pipefail
cd /home/mhough/dev/hippy-feat
mkdir -p /data/derivatives/rtmindeye_paper/logs

TIERS=${TIERS:-Fast Slow EoR}
KEEP_BLANKS=${KEEP_BLANKS:-0}
KB_FLAG=""
if [[ "$KEEP_BLANKS" == "1" ]]; then
    KB_FLAG="--keep-blanks"
fi
.venv/bin/python -u scripts/run_streaming_rls_glm.py --tiers $TIERS $KB_FLAG
