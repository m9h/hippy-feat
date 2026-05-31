#!/bin/bash
#SBATCH --job-name=ww-zuna-block
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/eeg_sae/logs/%x-%j.out
# No --time: single-user cluster, infinite walltime.

# Encoder/decoder per-block WeightWatcher (HT-SR) alpha breakdown for ZUNA
# (mhough/zuna-base, Zyphra 380M masked-diffusion EEG AE). Unlike the pooled
# global alpha in analyze_eegfm_weightwatcher.py, this splits the 16 encoder
# + 16 decoder blocks so the encoder-only alpha lines up against REVE block-6.
#
# We analyse ZUNA's raw safetensors state_dict (not the live transformers
# custom-code module): the keys already carry full typing
# (model.encoder.layers.{k}.attention.wq.weight …), and WeightWatcher's torch
# walker chokes on ZUNA's custom lingua modules. So this needs only
# huggingface_hub + safetensors + weightwatcher — all in ~/.local, no zuna
# package required. The HF + WW stack lives only in ~/.local, so
# PYTHONNOUSERSITE is deliberately NOT set — safe because ~/.local has no
# torch/numpy to shadow the SIF GPU stack.

set -euo pipefail
PROJECT_ROOT=/home/mhough/dev/hippy-feat
SIF=/data/derivatives/containers/pytorch_26.04.sif
OUT_DIR=/data/derivatives/eeg_sae/weightwatcher
CACHE_DIR=/data/derivatives/eeg_sae/hf_cache

mkdir -p "$OUT_DIR" /data/derivatives/eeg_sae/logs

echo "Job:    analyze_zuna_weightwatcher.sbatch"
echo "Out:    $OUT_DIR/zuna_base_by_block.csv"

apptainer exec \
    --env PYTHONPATH="$PROJECT_ROOT" \
    --env HF_HOME="$CACHE_DIR" \
    -B "$PROJECT_ROOT":"$PROJECT_ROOT" \
    -B /data:/data \
    "$SIF" bash -c "
        set -euo pipefail
        cd $PROJECT_ROOT
        python scripts/analyze_zuna_weightwatcher.py \
            --model-id mhough/zuna-base \
            --cache-dir '$CACHE_DIR' \
            --out-prefix '$OUT_DIR/zuna_base'
    "

echo "[done] $OUT_DIR/zuna_base_by_block.csv"
