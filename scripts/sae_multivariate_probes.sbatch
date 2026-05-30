#!/bin/bash
#SBATCH --job-name=sae-mv-probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out

# Companion to sae_concept_probes.sbatch: fit multivariate ridge probes
# (all SAE features → concept) instead of per-feature univariate. Picks
# up compositional encoding the univariate metric misses.

set -euo pipefail
PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/jax-arm64-cu13.sif
OUT_DIR=/data/derivatives/eeg_sae

ACTS=${ACTS:?"set ACTS=/path/to/activations.npz"}
SAE=${SAE:-}                # blank when RAW=1
RAW=${RAW:-0}               # set RAW=1 to bypass the SAE
SEED=${SEED:-0}
TEST_FRAC=${TEST_FRAC:-0.25}

if [[ "$RAW" == "1" ]]; then
    ACTS_TAG=$(basename "$ACTS" .npz)
    OUT_PREFIX="$OUT_DIR/probes/${ACTS_TAG}_RAW"
elif [[ -n "$SAE" ]]; then
    SAE_TAG=$(basename "$SAE" .npz)
    OUT_PREFIX="$OUT_DIR/probes/${SAE_TAG}"
else
    echo "ERROR: must set SAE=... or RAW=1" >&2; exit 1
fi
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/probes"

echo "Job:    sae_multivariate_probes.sbatch"
echo "Acts:   $ACTS"
echo "SAE:    $SAE"
echo "Out:    ${OUT_PREFIX}_multivariate.{csv,json}"

PY_ARGS=( --activations "$ACTS"
          --out-prefix "$OUT_PREFIX"
          --test-frac "$TEST_FRAC"
          --seed "$SEED" )
if [[ "$RAW" == "1" ]]; then
    PY_ARGS+=( --raw )
else
    PY_ARGS+=( --sae "$SAE" )
fi

apptainer exec --nv \
    --env PYTHONPATH="$PROJECT_ROOT" \
    --env PYTHONNOUSERSITE=1 \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    "$SIF" bash -c "
        set -euo pipefail
        export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:\$LD_LIBRARY_PATH
        cd $PROJECT_ROOT
        python scripts/sae_multivariate_probes.py $(printf '%q ' "${PY_ARGS[@]}")
    "

echo "[done] multivariate probes: ${OUT_PREFIX}_multivariate.csv"
