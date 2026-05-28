#!/bin/bash
#SBATCH --job-name=hbn-predownload
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite. Expect hours.

# Pre-download all 11 HBN-EEG releases' RestingState task into the eegdash
# cache so subsequent extraction jobs read from local disk. Runs on CPU only
# — pure I/O. Sequential across releases so the OpenNeuro S3 endpoint isn't
# hammered with parallel requests.

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae
CACHE_DIR=/data/derivatives/eegdash_cache

mkdir -p "$OUT_DIR/logs" "$CACHE_DIR"

echo "Job:    predownload_hbn_all.sbatch"
echo "Cache:  $CACHE_DIR"
echo "Releases: ${RELEASES:-1..11} / task=${TASK:-RestingState}"

# Build the python command in the OUTER shell so array expansion is
# deterministic. Same idiom as extract_eeg_fm_acts.sbatch — passing
# the array through a double-quoted `bash -c "..."` drops elements
# (e.g. `--releases 2 3 4` becomes just `--releases`).
PY_CMD=( python scripts/predownload_hbn_all.py
    --cache-dir "$CACHE_DIR"
    --task "${TASK:-RestingState}"
)
if [[ -n "${RELEASES:-}" ]]; then
    PY_CMD+=( --releases ${RELEASES} )
fi

# Use the pytorch SIF since it already has the right Python + we need only
# eegdash + braindecode + mne (which we pip-install into user-site).
# No GPU needed.
apptainer exec --no-init \
    --env PYTHONPATH="$PROJECT_ROOT" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    -B "$CACHE_DIR":"$CACHE_DIR" \
    "$SIF" bash -c '
        set -euo pipefail
        # NO pip install (see extract_eeg_fm_acts.sbatch for the reason).
        cd "$1"; shift
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" "${PY_CMD[@]}"

echo "[done] predownload complete"
