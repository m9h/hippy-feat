#!/bin/bash
#SBATCH --job-name=peng-inventory
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/datasets/peng_2024_neurofeedback/logs/%x-%j.out

# Walks every Peng-2024 subject/session, reports per-run BOLD shapes,
# masks, behavioral trial counts. Includes a Variant G sanity check
# on the first subject's first feedback run — confirms the AR(1)
# Bayesian conjugate GLM produces sensible posterior variances on
# Peng's data structure (different mask format from sub-005).
#
# Output: /data/datasets/peng_2024_neurofeedback/inventory.json

set -euo pipefail
mkdir -p /data/datasets/peng_2024_neurofeedback/logs

cd /home/mhough/dev/hippy-feat
.venv/bin/python scripts/peng_inventory.py \
    --out /data/datasets/peng_2024_neurofeedback/inventory.json
