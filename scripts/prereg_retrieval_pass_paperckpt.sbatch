#!/bin/bash
#SBATCH --job-name=prereg-ret-paper-ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Paper-confirmed checkpoint (per Rishab on Discord): the actual ckpt that
# produced Table 1 numbers. Single-session ses-01 fine-tune, repeats_3split
# fold 10 (NOT fold 0 we'd been using), 150 epochs.

set -euo pipefail
IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
mkdir -p /data/derivatives/rtmindeye_paper/logs
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker load -i "$TAR"
fi

CKPT=/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth

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
