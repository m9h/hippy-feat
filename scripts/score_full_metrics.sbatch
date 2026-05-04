#!/bin/bash
#SBATCH --job-name=score-full-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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

# Paper-confirmed fold-10 ckpt (per Rishab's Discord clarification).
CKPT=${CKPT:-/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth}
# Default cells = 5 Table 1 tier rows we have βs for, all paper-faithful:
#   Probe_canonical_sessztrain_firstrep  → Offline 3T (single first-rep)
#   Probe_canonical_sessztrain_repavg    → Offline 3T (avg 3 reps)
#   Paper_RT_actual_delay0               → Fast RT
#   Paper_RT_actual_delay5               → Slow RT
#   Paper_RT_actual_delay63              → End-of-run RT
CELLS=${CELLS:-Probe_canonical_sessztrain_firstrep Probe_canonical_sessztrain_repavg Paper_RT_actual_delay0 Paper_RT_actual_delay5 Paper_RT_actual_delay63}
N_SEEDS=${N_SEEDS:-5}
# Probe_canonical_*_firstrep / _repavg cells are already n=50 unique images;
# RT_paper βs are 150 trials and need first-rep filtering. Default --first-rep
# is a no-op on the Probe cells, applies on the RT cells.
FIRST_REP=${FIRST_REP:---first-rep}
OUT=${OUT:-/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/full_metrics_fold10_table1.json}

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
        python -u scripts/score_full_metrics.py \
            --checkpoint $CKPT \
            --cells $CELLS \
            --n-seeds $N_SEEDS $FIRST_REP \
            --out $OUT
    "
