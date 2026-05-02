#!/bin/bash
#SBATCH --job-name=score-full-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Full Table-1-style scorer: 8 reconstruction metrics + 2 retrieval metrics,
# 5-seed averaging on the diffusion-prior + SDXL pipeline.
# Time budget: ~25 min/cell on GB10 GPU; 8 cells → ~3-4 hr.

set -euo pipefail
IMAGE=mindeye-pytorch:26.03
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
mkdir -p /data/derivatives/rtmindeye_paper/logs
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker load -i "$TAR"
fi

# Default: paper-faithful ses-01-only finalmask checkpoint, first-rep filter.
CKPT=${CKPT:-/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth}
CELLS=${CELLS:-Canonical_GLMsingle_ses-03 AR1freq_glover_rtm_glmdenoise_fracridge}
N_SEEDS=${N_SEEDS:-5}
FIRST_REP=${FIRST_REP:---first-rep}
OUT=${OUT:-/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/full_metrics.json}

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
        python -u scripts/score_full_metrics.py \
            --checkpoint $CKPT \
            --cells $CELLS \
            --n-seeds $N_SEEDS $FIRST_REP \
            --out $OUT
    "
