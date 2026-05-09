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
INPROC_ACOMPCOR=${INPROC_ACOMPCOR:-0}
EXTRA=""
[[ "$KEEP_BLANKS" == "1" ]] && EXTRA="$EXTRA --keep-blanks"
[[ "$INPROC_ACOMPCOR" == "1" ]] && EXTRA="$EXTRA --inprocess-acompcor"
.venv/bin/python -u scripts/run_streaming_rls_glm.py --tiers $TIERS $EXTRA
