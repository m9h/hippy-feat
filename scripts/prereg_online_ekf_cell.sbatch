#!/bin/bash
#SBATCH --job-name=prereg-online-ekf-cell
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

set -euo pipefail
cd /home/mhough/dev/hippy-feat
mkdir -p /data/derivatives/rtmindeye_paper/logs

.venv/bin/python -u scripts/prereg_online_ekf_cell.py \
    --session ses-03 \
    --runs 1 2 3 4 5 6 7 8 9 10 11 \
    --post-stim-tr 8 \
    --out-cell-name EKF_session_online_glover_rtm
