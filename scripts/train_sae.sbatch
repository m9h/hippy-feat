#!/bin/bash
#SBATCH --job-name=train-sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Train a TopK SAE on extracted EEG-FM activations.
# Runs inside the NGC JAX SIF — build it first with:
#     sbatch scripts/build_jax_arm64_sif.sbatch
#
# Usage:
#   ACTS=/data/derivatives/eeg_sae/acts/brain-bzh_reve-base_L-1_hbn_R3_mini_RestingState.npz \
#     sbatch scripts/train_sae.sbatch
#
# Optional env knobs:
#   K=32 D_DICT=12288 LR=1e-3 EPOCHS=20 BATCH=4096

set -euo pipefail

PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/jax-arm64-cu13.sif
OUT_DIR=/data/derivatives/eeg_sae

ACTS=${ACTS:?"set ACTS=/path/to/activations.npz"}
K=${K:-32}
D_DICT=${D_DICT:-}     # blank = 16 * d_model (auto)
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-4096}
AUX_K=${AUX_K:-0}      # 0 = disabled. Set AUX_K=$K to turn on Gao 2024 aux_k.
AUX_COEF=${AUX_COEF:-0.03125}
AUX_KILL=${AUX_KILL:-200}
MAX_TOKENS=${MAX_TOKENS:-}  # blank = no cap; e.g. 8000000 for memory-bounded sweep

# Output prefix derived from input filename + SAE hyperparams. Encode aux_k
# in the tag so a plain k=32 run and a k=32 + aux_k=32 run don't collide.
ACTS_TAG=$(basename "$ACTS" .npz)
AUX_TAG=""
if [[ "$AUX_K" != "0" ]]; then
    AUX_TAG="_auxk${AUX_K}"
fi
OUT_PREFIX="$OUT_DIR/sae/${ACTS_TAG}_k${K}_dd${D_DICT:-auto}_lr${LR}${AUX_TAG}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/sae"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found. Submit scripts/build_jax_arm64_sif.sbatch first." >&2
    exit 1
fi
if [[ ! -f "$ACTS" ]]; then
    echo "ERROR: activations not found: $ACTS" >&2; exit 1
fi

echo "Job:    train_sae.sbatch"
echo "Acts:   $ACTS"
echo "Out:    ${OUT_PREFIX}.{npz,json}"
echo "Cfg:    k=$K d_dict=${D_DICT:-auto} lr=$LR epochs=$EPOCHS batch=$BATCH"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Build the python command in the OUTER shell to dodge the array-in-quoted-
# bash-c bug that drops `--d-dict 2048` to just `--d-dict`. Same idiom as
# extract_eeg_fm_acts.sbatch + predownload_hbn_all.sbatch.
PY_CMD=( python scripts/train_sae.py
    --activations "$ACTS"
    --out-prefix "$OUT_PREFIX"
    --k "$K"
    --lr "$LR"
    --epochs "$EPOCHS"
    --batch-size "$BATCH"
    --aux-k "$AUX_K"
    --aux-coef "$AUX_COEF"
    --aux-steps-to-kill "$AUX_KILL"
)
if [[ -n "$D_DICT" ]]; then
    PY_CMD+=( --d-dict "$D_DICT" )
fi
if [[ -n "$MAX_TOKENS" ]]; then
    PY_CMD+=( --max-tokens "$MAX_TOKENS" )
fi

# -B /data:/data : apptainer does NOT auto-bind NFS paths; without this
#   JAX inside the SIF can't see JAX_COMPILATION_CACHE_DIR (/data/mhough/cache/jax).
# LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu : --nv strips standard /usr/lib
#   from the linker search, breaking jaxlib's dlopen(libcudnn.so.9).
# PYTHONNOUSERSITE=1 : the pytorch SIF previously pip-installed a CPU-only
#   jax 0.10.x into ~/.local — that user-site shadows the JAX SIF's own
#   CUDA-enabled jax 0.9.2 and the bundled jax_cuda13_plugin then refuses
#   to load (plugin/jaxlib version mismatch → silent CPU fallback). Disabling
#   user-site picks up the SIF's matched stack instead. The SIF already ships
#   optax 0.2.8, jax/jaxlib 0.9.2/0.10.x, flax — no pip install needed here.
apptainer exec --nv \
    --env PYTHONPATH="$PROJECT_ROOT" \
    --env PYTHONNOUSERSITE=1 \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    "$SIF" bash -c '
        set -euo pipefail
        export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
        cd "$1"; shift
        echo "[exec] $*"
        exec "$@"
    ' bash "$PROJECT_ROOT" "${PY_CMD[@]}"

echo "[done] SAE checkpoint: ${OUT_PREFIX}.npz"
