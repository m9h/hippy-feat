#!/bin/bash
#SBATCH --job-name=ww-eegfm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, infinite walltime.

# WeightWatcher (HT-SR) alpha-spectrum analysis across the NeuralBench EEG
# foundation models. CPU-only (WW reads weight-matrix ESDs — no GPU needed),
# so this can run alongside GPU jobs without contention.
#
# Usage:
#   sbatch scripts/analyze_eegfm_weightwatcher.sbatch          # all models
#   MODELS=zuna sbatch scripts/analyze_eegfm_weightwatcher.sbatch
# The summary table merges by model name, so adding one model (e.g. zuna)
# updates all_models_summary.{csv,json} without clobbering prior rows.

set -euo pipefail
PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae/weightwatcher
CACHE_DIR=/data/derivatives/eeg_sae/hf_cache
MODELS=${MODELS:-zuna}

# NOTE: PYTHONNOUSERSITE is deliberately NOT set here. The NGC PyTorch SIF
# supplies torch + numpy/pandas/scipy, but the HF + analysis stack
# (transformers, weightwatcher, powerlaw, scikit-learn) lives only in
# ~/.local. Blocking user-site breaks the run. Safe because ~/.local has no
# torch/numpy to shadow the SIF's GPU stack (only CPU jax, unused here).

mkdir -p "$OUT_DIR" /data/derivatives/eeg_sae/logs

echo "Job:    analyze_eegfm_weightwatcher.sbatch"
echo "Models: $MODELS"
echo "Out:    $OUT_DIR/all_models_summary.csv"

apptainer exec \
    --env PYTHONPATH="$PROJECT_ROOT" \
    --env HF_HOME="$CACHE_DIR" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    "$SIF" bash -c "
        set -euo pipefail
        cd $PROJECT_ROOT
        python scripts/analyze_eegfm_weightwatcher.py \
            --cache-dir '$CACHE_DIR' \
            --out-dir '$OUT_DIR' \
            --models $MODELS
    "

echo "[done] $OUT_DIR/all_models_summary.csv"
