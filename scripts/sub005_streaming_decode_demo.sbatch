#!/bin/bash
#SBATCH --job-name=sub005-streaming-demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

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
        # The container's python lacks our scripts/ deps but shares /home mount;
        # add scripts/ to PYTHONPATH and run.
        export PYTHONPATH=/home/mhough/dev/hippy-feat:\$PYTHONPATH
        python scripts/sub005_streaming_decode_demo.py --phase decode
    "
