#!/bin/bash
#SBATCH --job-name=dl-fmriprep-mindeye
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Pull the paper's fMRIPrep'd sub-005 derivatives (41 files, ~8-10GB).
# This gives us the 'Offline 3T' branch of the Task 2.1 factorial for free,
# without needing to run fMRIPrep ourselves.

set -euo pipefail

OUT=/data/derivatives/rtmindeye_paper/fmriprep_mindeye
mkdir -p "$OUT" /data/derivatives/rtmindeye_paper/logs

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

hf download --repo-type dataset rishab-iyer1/fmriprep_mindeye \
    --local-dir "$OUT" \
    --token "$HF_TOKEN" \
    --max-workers 4

echo "Downloaded to $OUT:"
du -sh "$OUT"
find "$OUT" -name "*_bold.nii.gz" | wc -l
unset HF_TOKEN
