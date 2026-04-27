#!/bin/bash
#SBATCH --job-name=prereg-variant-sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Pre-registered 12-cell variant sweep (per TASK_2_1_PREREGISTRATION.md).
# Cells 1, 2, 4, 5, 6 are JAX-only and run in seconds-to-minutes each.
# Cells 7-12 add denoising / nilearn replicas — run in separate jobs.

set -euo pipefail
cd /home/mhough/dev/hippy-feat
mkdir -p /data/derivatives/rtmindeye_paper/logs

CELLS=${CELLS:-OLS_glover_rtm AR1freq_glover_rtm VariantG_glover_rtm VariantG_glover_rtm_prior AR1freq_glmsingleS1_rtm}

.venv/bin/python scripts/prereg_variant_sweep.py \
    --cells $CELLS \
    --session ses-03 \
    --runs 1 2 3 4 5 6 7 8 9 10 11
