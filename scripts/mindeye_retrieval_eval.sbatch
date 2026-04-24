#!/bin/bash
#SBATCH --job-name=mindeye-retrieval-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# MindEye retrieval-only inference on per-trial betas. Uses the latest
# NGC PyTorch 26.03 arm64 SIF as the runtime, pip-installs MindEye-specific
# Python deps at job start inside a user-site directory so the SIF stays
# reusable across conditions.
#
# Prereqs (verified before sbatch submission):
#   - /data/derivatives/containers/pytorch_26.03.sif                 (job 892)
#   - /data/derivatives/rtmindeye_paper/repos/rtcloud-projects/      (job 891)
#   - /data/derivatives/rtmindeye_paper/rt3t/data/all_stimuli/special515/  (890)
#   - /data/derivatives/rtmindeye_paper/task_2_1_betas/<condition>_<session>_betas.npy
#   - /data/derivatives/rtmindeye_paper/checkpoints/.../sample=10_...finalmask...pth

set -euo pipefail

CONDITION=${CONDITION:-A_fmriprep_glover}
SESSION=${SESSION:-ses-03}
# All MindEye runtime deps are baked into this SIF (paper pins +
# OpenAI CLIP + sgm). See containers/mindeye_pytorch.def and
# scripts/build_mindeye_sif.sbatch. Retrieval jobs no longer install
# anything at job start — 0-sec startup overhead.
SIF=/data/derivatives/containers/mindeye_pytorch_26.03.sif
mkdir -p /data/derivatives/rtmindeye_paper/logs

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found." >&2
    echo "Build it first: sbatch scripts/build_mindeye_sif.sbatch" >&2
    exit 1
fi

apptainer exec --nv \
    --bind /data:/data \
    --bind /home/mhough:/home/mhough \
    "$SIF" \
    python /home/mhough/dev/hippy-feat/scripts/mindeye_retrieval_eval.py \
        --condition "$CONDITION" \
        --session "$SESSION" \
        --out-json /data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_${CONDITION}.json

echo "Done."
