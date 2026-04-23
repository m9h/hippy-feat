#!/bin/bash
#SBATCH --job-name=dl-paper-ckpt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Pull the paper's canonical checkpoint (sample=10 = full data) plus any
# realtime-dump weights. Skipping `seedwise_runs_dump/` (7346 eval files).
#
# Paper-canonical .pth is under data_scaling_exp/concat_glmsingle/checkpoints/
# (10 variants by training-data amount — we grab them all, they're small).

set -euo pipefail

OUT=/data/derivatives/rtmindeye_paper/checkpoints
mkdir -p "$OUT" /data/derivatives/rtmindeye_paper/logs

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

hf download --repo-type model macandro96/mindeye_offline_ckpts \
    --local-dir "$OUT" \
    --token "$HF_TOKEN" \
    --max-workers 4 \
    --include "data_scaling_exp/concat_glmsingle/checkpoints/*.pth" \
    --include "data_scaling_exp/concat_glmsingle_unionmask/checkpoints/*.pth" \
    --include "realtime-dump/**/*.pth" \
    --include "realtime-dump/**/*.pt"

echo "Downloaded checkpoints:"
find "$OUT" -name "*.pth" -exec du -h {} \;
unset HF_TOKEN
