#!/bin/bash
#SBATCH --job-name=rt-paper-full-replica
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Cells 10-12 of TASK_2_1_PREREGISTRATION.md — faithful nilearn replicas.
# Slow: per-trial nilearn FirstLevelModel.fit on the whole brain.
# Expect ~30 min/cell on CPU. Three cells = up to 1.5 h total.

set -euo pipefail
cd /home/mhough/dev/hippy-feat
mkdir -p /data/derivatives/rtmindeye_paper/logs

CELLS=${CELLS:-RT_paper_replica_partial RT_paper_replica_full Offline_paper_replica_full}

.venv/bin/python scripts/rt_paper_full_replica.py \
    --cells $CELLS \
    --session ses-03 \
    --runs 1 2 3 4 5 6 7 8 9 10 11
