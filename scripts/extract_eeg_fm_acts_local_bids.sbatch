#!/bin/bash
#SBATCH --job-name=eeg-fm-extract-localbids
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Local-BIDS extraction — bypass eegdash entirely for R7 + R11 whose
# MongoDB registry returns 0 subjects. Uses extract_eeg_fm_acts_local_bids.py
# which reads cached .set files via mne_bids and builds a braindecode
# BaseConcatDataset directly.
#
# Usage:
#   RELEASE=7 LAYER=6 sbatch scripts/extract_eeg_fm_acts_local_bids.sbatch

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae
CACHE_DIR=/data/derivatives/eegdash_cache
HF_CACHE=/data/derivatives/eeg_sae/hf_cache
HF_TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token

MODEL=${MODEL:-brain-bzh/reve-base}
LAYER=${LAYER:--1}
RELEASE=${RELEASE:?"set RELEASE=N (7 or 11 for the registry-gap releases)"}
TASK=${TASK:-RestingState}
WIN_SECONDS=${WIN_SECONDS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_WINDOWS=${MAX_WINDOWS:-}

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/acts" "$CACHE_DIR" "$HF_CACHE"

if [[ ! -s "$HF_TOKEN_FILE" ]]; then
    echo "ERROR: HF token not at $HF_TOKEN_FILE" >&2; exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

SAFE_MODEL=${MODEL//\//_}
RUN_TAG="${SAFE_MODEL}_L${LAYER}_EEG2025R${RELEASE}_${TASK}"
OUT_NPZ="$OUT_DIR/acts/${RUN_TAG}.npz"

echo "Job:    extract_eeg_fm_acts_local_bids.sbatch"
echo "Model:  $MODEL  layer=$LAYER"
echo "Data:   R${RELEASE} (local BIDS) / task=$TASK"
echo "Out:    $OUT_NPZ"

PY_CMD=( python scripts/extract_eeg_fm_acts_local_bids.py
    --model "$MODEL"
    --layer "$LAYER"
    --release "$RELEASE"
    --task "$TASK"
    --cache-dir "$CACHE_DIR"
    --win-seconds "$WIN_SECONDS"
    --batch-size "$BATCH_SIZE"
    --out "$OUT_NPZ"
)
[[ -n "$MAX_WINDOWS" ]] && PY_CMD+=( --max-windows "$MAX_WINDOWS" )

apptainer exec --nv --no-init \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="$HF_CACHE" \
    --env PYTHONPATH="$PROJECT_ROOT" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    -B "$CACHE_DIR":"$CACHE_DIR" \
    -B "$OUT_DIR":"$OUT_DIR" \
    -B "$HF_CACHE":"$HF_CACHE" \
    "$SIF" bash -c '
        set -euo pipefail
        cd "$1"; shift
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" "${PY_CMD[@]}"

unset HF_TOKEN
echo "[done] extraction finished: $OUT_NPZ"
