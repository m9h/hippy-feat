#!/bin/bash
#SBATCH --job-name=fast-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

set -euo pipefail
mkdir -p /data/derivatives/rtmindeye_paper/logs

# Run inside the same MindEye container we use for score_full_metrics so the
# sgm + nilearn + open_clip + clip stack matches.
IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker load -i "$TAR"
fi

ZPOLICY=${ZPOLICY:-session_z}
CKPT_FOLD=${CKPT_FOLD:-0}
OUT_SUFFIX=${OUT_SUFFIX:-}

docker run --rm --gpus all \
    -v /home/mhough:/home/mhough \
    -v /data:/data \
    -w /home/mhough/dev/hippy-feat \
    -e ZPOLICY="$ZPOLICY" \
    -e CKPT_FOLD="$CKPT_FOLD" \
    -e OUT_SUFFIX="$OUT_SUFFIX" \
    "$IMAGE" \
    bash -c "
        rm -rf /usr/local/lib/python3.12/dist-packages/transformer_engine \
               /usr/local/lib/python3.12/dist-packages/transformer_engine-*.dist-info \
               2>/dev/null || true
        pip install --quiet pytorch_lightning omegaconf einops kornia open_clip_torch \
                            transformers diffusers fsspec xformers 2>&1 | tail -3
        export PYTHONPATH=/home/mhough/dev/hippy-feat:\$PYTHONPATH
        python -u scripts/train_fast_distill_from_slow.py
    "
