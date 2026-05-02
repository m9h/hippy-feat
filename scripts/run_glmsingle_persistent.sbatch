#!/bin/bash
#SBATCH --job-name=glmsingle-persistent
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Test (b): real GLMsingle persistent global fit on rtmotion or fmriprep
# BOLD. Default arg picks rtmotion.

set -euo pipefail
cd /home/mhough/dev/hippy-feat
mkdir -p /data/derivatives/rtmindeye_paper/logs

BOLD=${BOLD:-rtmotion}
.venv/bin/python -u scripts/run_glmsingle_persistent.py --bold-source "$BOLD"
