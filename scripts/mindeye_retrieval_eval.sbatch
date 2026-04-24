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
VENV_DIR=/data/derivatives/rtmindeye_paper/mindeye_venv
UV_CACHE=/data/derivatives/uv_cache
MARKER_VER=v2
MARKER=$VENV_DIR/.install_complete_${MARKER_VER}
mkdir -p "$(dirname "$VENV_DIR")" "$UV_CACHE" /data/derivatives/rtmindeye_paper/logs

apptainer exec --nv \
    --bind /data:/data \
    --bind /home/mhough:/home/mhough \
    --env "UV_CACHE_DIR=$UV_CACHE" \
    --env "VENV_DIR=$VENV_DIR" \
    --env "MARKER=$MARKER" \
    "$SIF" \
    bash -c '
        set -euo pipefail

        # Bootstrap uv once on NAS (~/.cache/uv goes to $UV_CACHE_DIR via env).
        # Uses NGC container'"'"'s python, installs uv to container-managed site.
        python -m pip install --quiet --user uv 2>&1 | tail -3
        export PATH=$HOME/.local/bin:$PATH

        # Persistent venv on NAS — created once, reused forever.
        if [ ! -d "$VENV_DIR" ]; then
            echo "[env] creating venv at $VENV_DIR"
            uv venv --python "$(which python)" "$VENV_DIR"
        fi

        # Skip the 15-minute install step if a previous run already
        # populated the venv. Marker version bumps on env-schema changes.
        if [ ! -f "$MARKER" ]; then
            echo "[env] installing MindEye paper deps into $VENV_DIR (first time)"
            UVPIP="uv pip install --python $VENV_DIR/bin/python --cache-dir $UV_CACHE_DIR --quiet"
            $UVPIP \
                "open_clip_torch" "accelerate==0.24.1" "omegaconf" "einops==0.8.1" \
                "transformers>=4.40" "timm" "nilearn" "nibabel" "imageio" \
                "webdataset" "dalle2-pytorch==1.15.6" "scikit-image" \
                "diffusers==0.23.0" "h5py" "scipy" 2>&1 | tail -5
            $UVPIP "git+https://github.com/openai/CLIP.git" 2>&1 | tail -3
            $UVPIP "/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/models/generative_models" 2>&1 | tail -3 || true
            touch "$MARKER"
            echo "[env] install complete — marker $MARKER touched; future jobs skip this step"
        else
            echo "[env] reusing venv at $VENV_DIR (marker $MARKER present)"
        fi

        cd /home/mhough/dev/hippy-feat
        "$VENV_DIR/bin/python" scripts/mindeye_retrieval_eval.py \
            --condition '"$CONDITION"' \
            --session '"$SESSION"' \
            --out-json /data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_'"$CONDITION"'.json
    '

echo "Done."
