#!/bin/bash
#SBATCH --job-name=mindeye-retrieval-docker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Run MindEye retrieval directly via docker (no apptainer SIF).
# The docker image mindeye-pytorch:26.03 was built once and cached at
# /data/mhough/docker (via overlay). Slurm allocates the GPU, docker
# uses --gpus all to pass it through nvidia-container-runtime.
#
# Why docker not apptainer: the apptainer SIF convert step has been
# the 24-hour blocker. Docker run skips it entirely; the image works
# (verified torch + diffusers + accelerate + open_clip + matmul on GB10).
#
# Runtime fix-up: NGC 26.03 ships transformer_engine which has an
# undefined-symbol mismatch with the container's libcublasLt 13.x.
# accelerate==0.24.1 imports TE at module load. Removing the TE package
# from the container's writable layer at start makes accelerate's
# importlib.util.find_spec("transformer_engine") return None, so
# accelerate skips the TE codepath entirely. We don't use TE for
# retrieval-only inference.

set -euo pipefail

CONDITION=${CONDITION:-A_fmriprep_glover}
SESSION=${SESSION:-ses-03}
IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
OUT_JSON=/data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_${CONDITION}.json

mkdir -p /data/derivatives/rtmindeye_paper/logs

# Self-heal: if the image got pruned from the local docker cache, reload
# from the NAS tarball before run.
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
        # Remove the broken transformer_engine package so accelerate skips
        # its TE wrappers (libcublasLt symbol mismatch in NGC 26.03).
        rm -rf /usr/local/lib/python3.12/dist-packages/transformer_engine \
               /usr/local/lib/python3.12/dist-packages/transformer_engine-*.dist-info \
               2>/dev/null || true

        python scripts/mindeye_retrieval_eval.py \
            --condition $CONDITION \
            --session $SESSION \
            --out-json $OUT_JSON
    "

echo "Done. JSON at $OUT_JSON"
