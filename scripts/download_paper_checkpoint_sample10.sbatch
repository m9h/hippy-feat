#!/bin/bash
#SBATCH --job-name=dl-paper-ckpt-s10
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Targeted fetch: the canonical paper checkpoint (sample=10 = full training
# data, finalmask). The bigger dl-paper-ckpt job (884) timed out at 2h before
# reaching this file after pulling samples 1-9 (~77 GB).

set -euo pipefail

OUT=/data/derivatives/rtmindeye_paper/checkpoints
mkdir -p "$OUT" /data/derivatives/rtmindeye_paper/logs

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

hf download --repo-type model macandro96/mindeye_offline_ckpts \
    --local-dir "$OUT" \
    --token "$HF_TOKEN" \
    --max-workers 2 \
    --include "data_scaling_exp/concat_glmsingle/checkpoints/sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_avgrepeats_finalmask_epochs_150.pth"

ls -la "$OUT/data_scaling_exp/concat_glmsingle/checkpoints/" | grep sample=10
unset HF_TOKEN
