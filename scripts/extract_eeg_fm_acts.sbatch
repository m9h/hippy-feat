#!/bin/bash
#SBATCH --job-name=eeg-fm-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Extract block-k activations from REVE (or LaBraM) into an .npz, ready for
# TopK-SAE training. Runs inside the existing PyTorch 26.03 NGC SIF, which
# has CUDA + torch + transformers; eegdash + braindecode get pip-installed
# into a per-run overlay so the SIF stays clean and reproducible.
#
# Usage:
#   sbatch scripts/extract_eeg_fm_acts.sbatch                       # HBN R1 MINI (20 subj)
#   RELEASE=1 FULL=1 LAYER=8 sbatch scripts/extract_eeg_fm_acts.sbatch  # HBN R1 full (136 subj), block 8
#   RELEASE=3 sbatch scripts/extract_eeg_fm_acts.sbatch              # HBN R3 MINI
#   MODEL=labram-base sbatch scripts/extract_eeg_fm_acts.sbatch
#   MODEL=mhough/zuna-base LAYER=-1 sbatch scripts/extract_eeg_fm_acts.sbatch  # ZUNA encoder, 256 Hz
#
# Pre-req: REVE only — accept the brain-bzh/reve-base Responsible Use
# Agreement on HF and have HF_TOKEN in ~/.cache/huggingface/token. ZUNA
# (mhough/zuna-base) is public; no token needed.

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae
CACHE_DIR=/data/derivatives/eegdash_cache
HF_CACHE=/data/derivatives/eeg_sae/hf_cache
HF_TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token

MODEL=${MODEL:-brain-bzh/reve-base}
LAYER=${LAYER:--1}
RELEASE=${RELEASE:-1}                   # HBN release 1..11
TASK=${TASK:-RestingState}
WIN_SECONDS=${WIN_SECONDS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_WINDOWS=${MAX_WINDOWS:-}            # empty = unlimited

# Model-specific knobs. REVE/LaBraM run at 200 Hz; ZUNA was pretrained at
# 256 Hz and its tokeniser needs T divisible by 32 (a 5 s window = 1280
# samples = 40·32). ZUNA's trust_remote_code module imports the ``zuna``
# package + its --target-installed deps from zunalibs, so that goes on
# PYTHONPATH for zuna runs only.
ZUNALIBS=/data/derivatives/eeg_sae/zunalibs
if [[ "$MODEL" == "mhough/zuna-base" ]]; then
    TARGET_SFREQ=${TARGET_SFREQ:-256}
    EXTRA_PYTHONPATH=":$ZUNALIBS"
else
    TARGET_SFREQ=${TARGET_SFREQ:-200}
    EXTRA_PYTHONPATH=""
fi
FULL_FLAG=""
DATASET_TAG="EEG2025R${RELEASE}MINI"
if [[ "${FULL:-0}" == "1" ]]; then
    FULL_FLAG="--full"
    DATASET_TAG="EEG2025R${RELEASE}"
fi

# Output filename encodes the run config so multiple extractions don't collide.
SAFE_MODEL=${MODEL//\//_}
RUN_TAG="${SAFE_MODEL}_L${LAYER}_${DATASET_TAG}_${TASK}"
OUT_NPZ="$OUT_DIR/acts/${RUN_TAG}.npz"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/acts" "$CACHE_DIR" "$HF_CACHE"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found." >&2; exit 1
fi
# Gated-model token is REVE-only; ZUNA + LaBraM load without it.
HF_TOKEN=""
if [[ "$MODEL" == brain-bzh/* ]]; then
    if [[ ! -s "$HF_TOKEN_FILE" ]]; then
        echo "ERROR: HF token not at $HF_TOKEN_FILE — needed for REVE gated model." >&2
        exit 1
    fi
    HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

echo "Job:    extract_eeg_fm_acts.sbatch"
echo "Model:  $MODEL  layer=$LAYER"
echo "Data:   ${DATASET_TAG} / task=$TASK   ${WIN_SECONDS}s @ ${TARGET_SFREQ}Hz"
echo "Out:    $OUT_NPZ"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Build the python command in the OUTER shell so the args list is concrete
# before we hand it to apptainer. Earlier attempts at expanding bash arrays
# inside a double-quoted `bash -c "..."` argument dropped `--max-windows 200`'s
# value silently (job 1381). Single source of quoting truth = outer shell.
PY_CMD=( python scripts/extract_eeg_fm_acts.py
    --model "$MODEL"
    --layer "$LAYER"
    --release "$RELEASE"
    --task "$TASK"
    --cache-dir "$CACHE_DIR"
    --win-seconds "$WIN_SECONDS"
    --target-sfreq "$TARGET_SFREQ"
    --batch-size "$BATCH_SIZE"
    --out "$OUT_NPZ"
)
[[ -n "$FULL_FLAG" ]] && PY_CMD+=( "$FULL_FLAG" )
[[ -n "$MAX_WINDOWS" ]] && PY_CMD+=( --max-windows "$MAX_WINDOWS" )

# --no-init  : the NGC pytorch SIF's /.singularity.d/env/*.sh startup
#   scripts hang (NGC license / nvidia-cli probe). Bypassing them is safe
#   because we pass HF_TOKEN/HF_HOME/PYTHONPATH explicitly via --env.
# -B /data:/data : apptainer doesn't auto-bind NFS paths.
apptainer exec --nv --no-init \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="$HF_CACHE" \
    --env PYTHONPATH="$PROJECT_ROOT$EXTRA_PYTHONPATH" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    -B "$CACHE_DIR":"$CACHE_DIR" \
    -B "$OUT_DIR":"$OUT_DIR" \
    -B "$HF_CACHE":"$HF_CACHE" \
    "$SIF" bash -c '
        set -euo pipefail
        # NO pip install. pip install --user historically pulled torch +
        # torchaudio + triton + nvidia_* CUDA wheels into ~/.local, which
        # then shadow the SIF stack and break subsequent runs (silent triton
        # AttrsDescriptor errors, torchaudio/torch CUDA mismatch). eegdash +
        # braindecode + mne are pre-installed in user-site; the extract
        # script stubs torchaudio with MagicMocks so braindecode imports
        # cleanly without needing it bundled.
        cd "$1"; shift
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" "${PY_CMD[@]}"

unset HF_TOKEN
echo "[done] extraction finished: $OUT_NPZ"
