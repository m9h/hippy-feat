#!/bin/bash
#SBATCH --job-name=prereg-ret-canonical-ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Test (a): retrieval scoring with the sample=10 canonical checkpoint
# (which trains on ses-01..ses-03 and has test-set leakage on ses-03).
# Compares against our default ses-01-only checkpoint to isolate the
# checkpoint contribution to the 16-20pp first-rep gap vs paper Offline 76%.

set -euo pipefail
IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
mkdir -p /data/derivatives/rtmindeye_paper/logs
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker load -i "$TAR"
fi

CKPT=/data/derivatives/rtmindeye_paper/checkpoints/data_scaling_exp/concat_glmsingle/checkpoints/sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_avgrepeats_finalmask_epochs_150.pth

docker run --rm --gpus all \
    -v /home/mhough:/home/mhough \
    -v /data:/data \
    -w /home/mhough/dev/hippy-feat \
    "$IMAGE" \
    bash -c "
        rm -rf /usr/local/lib/python3.12/dist-packages/transformer_engine \
               /usr/local/lib/python3.12/dist-packages/transformer_engine-*.dist-info \
               2>/dev/null || true
        export PYTHONPATH=/home/mhough/dev/hippy-feat:\$PYTHONPATH
        python -u scripts/prereg_retrieval_pass.py \
            --checkpoint $CKPT
    "
