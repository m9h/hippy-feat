#!/bin/bash
#SBATCH --job-name=task-6-extract-gt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# OpenCLIP GT extraction for training stimuli (ses-01 + ses-02). One-time
# cost ~15 min; populates the cache used by mindeye_retrieval_eval.py.

set -euo pipefail
IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar

mkdir -p /data/derivatives/rtmindeye_paper/logs

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    if [ -f "$TAR" ]; then
        echo "[setup] $IMAGE not in local cache — loading from $TAR"
        docker load -i "$TAR"
    else
        echo "[setup] $IMAGE missing AND $TAR not found. Run rebuild_mindeye_image.sbatch first."
        exit 1
    fi
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

        python scripts/task_6_extract_training_gt.py
    "
