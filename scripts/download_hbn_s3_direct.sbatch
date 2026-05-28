#!/bin/bash
#SBATCH --job-name=hbn-s3-direct
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Pulls HBN-EEG RestingState data directly from OpenNeuro S3, bypassing
# eegdash's registry. Use for releases where eegdash's metadata is
# incomplete (R7=ds005511, R11=ds005516 — see comment in the script).
#
# Usage:
#   RELEASES="7 11" sbatch scripts/download_hbn_s3_direct.sbatch

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae
CACHE_DIR=/data/derivatives/eegdash_cache

mkdir -p "$OUT_DIR/logs" "$CACHE_DIR"

RELEASES=${RELEASES:?"set RELEASES='7 11' or similar"}
TASK=${TASK:-RestingState}
WORKERS=${WORKERS:-8}

echo "Job:    download_hbn_s3_direct.sbatch"
echo "Cache:  $CACHE_DIR"
echo "Releases: $RELEASES   task=$TASK   workers=$WORKERS"

PY_CMD=( python scripts/download_hbn_s3_direct.py
    --cache-dir "$CACHE_DIR"
    --task "$TASK"
    --workers "$WORKERS"
    --releases ${RELEASES}
)

apptainer exec --no-init \
    --env PYTHONPATH="$PROJECT_ROOT" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    -B "$CACHE_DIR":"$CACHE_DIR" \
    "$SIF" bash -c '
        set -euo pipefail
        cd "$1"; shift
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" "${PY_CMD[@]}"

echo "[done] direct S3 download finished"
