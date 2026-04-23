#!/bin/bash
#SBATCH --job-name=dl-rt3t
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Pull the paper's 'real-time' sub-005 dataset selectively.
# Full repo is 32k files (~100+ GB with stimuli). We take only:
#   data/model/        (5 files)     - trained checkpoints
#   data/events/       (92 files)    - per-run events + TR labels
#   data/real_time_betas/ (16 files) - 2792-vox baseline betas (reference)
#   data/scripts/      (4 files)     - any helper scripts
#   data/*.nii.gz /*.npy /*.mat /*.tsv at root (~20 files) - masks, T1, HRF lib
# Skipping: all_stimuli (20k imgs), dicom_ses-03 (raw), derivatives (9k files),
#           raw_bids (already /data/3t/data/raw_bids locally).

set -euo pipefail

OUT=/data/derivatives/rtmindeye_paper/rt3t
mkdir -p "$OUT" /data/derivatives/rtmindeye_paper/logs

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

hf download --repo-type dataset rishab-iyer1/3t \
    --local-dir "$OUT" \
    --token "$HF_TOKEN" \
    --max-workers 4 \
    --include "data/model/*" \
    --include "data/events/*" \
    --include "data/real_time_betas/*" \
    --include "data/scripts/*" \
    --include "data/*.nii.gz" \
    --include "data/*.npy" \
    --include "data/*.mat" \
    --include "data/*.tsv" \
    --include "data/*.json" \
    --include "data/README.md" \
    --include "README.md"

echo "Downloaded to $OUT:"
du -sh "$OUT"
echo "---top-level data/ contents:"
ls "$OUT/data" 2>/dev/null | head -30
unset HF_TOKEN
