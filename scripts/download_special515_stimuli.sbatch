#!/bin/bash
#SBATCH --job-name=dl-special515
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Pull the 50 special515 test-set images from rt3t HF dataset.
# Paper's test set = 50 special515 images × 3 repeats = 150 trials in ses-03.
# Selective include keeps this at ~MB-scale instead of the 20k-file full tree.

set -euo pipefail

OUT=/data/derivatives/rtmindeye_paper/rt3t
mkdir -p /data/derivatives/rtmindeye_paper/logs

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

hf download --repo-type dataset rishab-iyer1/3t \
    --local-dir "$OUT" \
    --token "$HF_TOKEN" \
    --max-workers 4 \
    --include "data/all_stimuli/special515/*"

echo "special515 stimuli:"
ls "$OUT/data/all_stimuli/special515/" 2>/dev/null | wc -l
du -sh "$OUT/data/all_stimuli/special515/"
unset HF_TOKEN
