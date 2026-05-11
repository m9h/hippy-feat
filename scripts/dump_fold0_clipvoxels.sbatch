#!/bin/bash
#SBATCH --job-name=fold0-cv-dump
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

set -euo pipefail
mkdir -p /data/derivatives/rtmindeye_paper/logs

IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker load -i "$TAR"
fi

docker run --rm --gpus all \
    -v /home/mhough:/home/mhough \
    -v /data:/data \
    -w /home/mhough/dev/hippy-feat \
    "$IMAGE" \
    bash -c "
        rm -rf /usr/local/lib/python3.12/dist-packages/transformer_engine \
               /usr/local/lib/python3.12/dist-packages/transformer_engine-*.dist-info \
               2>/dev/null || true
        pip install --quiet pytorch_lightning omegaconf einops kornia open_clip_torch \
                            transformers diffusers fsspec xformers 2>&1 | tail -3
        export PYTHONPATH=/home/mhough/dev/hippy-feat:\$PYTHONPATH
        python -u scripts/dump_fold0_clipvoxels.py
    "
