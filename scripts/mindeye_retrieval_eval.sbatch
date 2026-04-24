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
        # Use uv (Rust-based pip replacement) — 10-100x faster, parallel
        # downloads, better caching. --reinstall on uv is much faster than
        # pip --force-reinstall on the container squashfs filesystem.
        python -m pip install --user --quiet uv 2>&1 | tail -3
        echo "[env] uv bootstrapped"

        # Use MedARC paper pins exactly — pyproject.toml at
        # /data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/conf/
        # pins diffusers==0.23.0 (not 0.21.4), accelerate==0.24.1, etc.
        # That version of diffusers still exposes diffusers.models.vae.Decoder.
        TGT="$PYTHONUSERBASE/lib/python3.12/site-packages"
        UVPIP="uv pip install --python $(which python) --target $TGT --quiet"

        $UVPIP \
            "open_clip_torch" "accelerate==0.24.1" "omegaconf" "einops==0.8.1" \
            "transformers>=4.40" "timm" "nilearn" "nibabel" "imageio" \
            "webdataset" "dalle2-pytorch==1.15.6" "scikit-image" \
            "diffusers==0.23.0" "h5py" "scipy" 2>&1 | tail -5
        # OpenAI CLIP (git, not pypi — pypi clip is a different package)
        $UVPIP "git+https://github.com/openai/CLIP.git" 2>&1 | tail -3
        # sgm (Stability generative-models) is packaged locally in the
        # rtcloud-projects tree as an installable wheel — no more sys.modules
        # stub needed.
        $UVPIP "/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/models/generative_models" 2>&1 | tail -3 || true
        echo "[env] torch + open_clip ready"
        cd /home/mhough/dev/hippy-feat
        python scripts/mindeye_retrieval_eval.py \
            --condition '"$CONDITION"' \
            --session '"$SESSION"' \
            --out-json /data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_'"$CONDITION"'.json
    '

echo "Done."
