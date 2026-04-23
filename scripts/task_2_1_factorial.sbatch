#!/bin/bash
#SBATCH --job-name=task-2-1-factorial
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Run the Task 2.1 factorial preprocessing (per-trial beta estimation).
# CPU partition: this is a one-time preprocessing pass, JAX falls back to
# CPU cleanly, and gpu partition has OverSubscribe=NO so it can't coexist
# with the checkpoint-download job holding 2 CPUs on cpu partition.
# CPU partition allows OverSubscribe=YES:4 so we fit alongside other jobs.
# Sized at 8 CPU / 48 GB to leave headroom for concurrent smri-fm work.

set -euo pipefail

cd /home/mhough/dev/hippy-feat
mkdir -p /data/derivatives/rtmindeye_paper/logs

.venv/bin/python scripts/task_2_1_factorial.py \
    --conditions ${FACTORIAL_CONDITIONS:-A_fmriprep_glover B_rtmotion_glmsingle} \
    --session ses-03 \
    --runs 1 2 3 4 5 6 7 8 9 10 11 \
    --out-dir /data/derivatives/rtmindeye_paper/task_2_1_betas

# Override with `FACTORIAL_CONDITIONS="A_fmriprep_glover" sbatch ...` to rerun
# just one condition. Condition B requires scripts/mcflirt_ses03.sbatch to have
# produced mc_boldres files at /data/3t/derivatives/motion_corrected_resampled/
# (raises FileNotFoundError per-run otherwise, which is recoverable).
