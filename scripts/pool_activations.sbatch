#!/bin/bash
#SBATCH --job-name=pool-acts
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster; partition max is infinite.

# Pool a random subsample of activations from each release into one .npz.
# CPU-only — no GPU needed (just I/O + concat + savez_compressed).
#
# Usage:
#   sbatch scripts/pool_activations.sbatch
# Optional env knobs:
#   PER_RELEASE=1000000   tokens to draw per release
#   GLOB="brain-bzh_reve-base_L6_EEG2025R*"   match the inputs
#   OUT=/data/derivatives/eeg_sae/acts/pool_L6_11release_11M.npz

set -euo pipefail
PROJECT_ROOT=/home/mhough/dev/hippy-feat
ACTS_DIR=/data/derivatives/eeg_sae/acts
PER_RELEASE=${PER_RELEASE:-1000000}
GLOB=${GLOB:-brain-bzh_reve-base_L6_EEG2025R*_RestingState.npz}
OUT=${OUT:-$ACTS_DIR/pool_L6_11release_${PER_RELEASE}.npz}

cd "$PROJECT_ROOT"
mapfile -t INPUTS < <(ls $ACTS_DIR/$GLOB | sort)
echo "Pooling ${#INPUTS[@]} inputs -> $OUT (per_release=$PER_RELEASE)"
python scripts/pool_activations.py \
    --inputs "${INPUTS[@]}" \
    --per-release "$PER_RELEASE" \
    --out "$OUT"
echo "[done] $OUT"
