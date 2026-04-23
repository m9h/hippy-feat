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
SIF=/data/derivatives/containers/pytorch_26.03.sif
PYUSER_SITE=/data/derivatives/rtmindeye_paper/pyuser_site
mkdir -p "$PYUSER_SITE" /data/derivatives/rtmindeye_paper/logs

apptainer exec --nv \
    --bind /data:/data \
    --bind /home/mhough:/home/mhough \
    --env "PYTHONUSERBASE=$PYUSER_SITE" \
    "$SIF" \
    bash -c '
        set -euo pipefail
        export PATH=$PYTHONUSERBASE/bin:$PATH
        # Install MindEye runtime deps (idempotent, cached in PYTHONUSERBASE).
        pip install --user --no-cache-dir --quiet \
            open_clip_torch accelerate omegaconf einops \
            "transformers>=4.40" timm nilearn nibabel imageio \
            webdataset dalle2_pytorch scikit-image 2>&1 | tail -5
        # models.py top-level imports OpenAI CLIP (not open_clip); add it.
        pip install --user --no-cache-dir --quiet \
            "git+https://github.com/openai/CLIP.git" 2>&1 | tail -3
        # utils_mindeye imports generative_models.sgm.* — Stability SDXL repo.
        pip install --user --no-cache-dir --quiet \
            "git+https://github.com/Stability-AI/generative-models.git" \
            2>&1 | tail -3
        echo "[env] torch + open_clip ready"
        cd /home/mhough/dev/hippy-feat
        python scripts/mindeye_retrieval_eval.py \
            --condition '"$CONDITION"' \
            --session '"$SESSION"' \
            --out-json /data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_'"$CONDITION"'.json
    '

echo "Done."
