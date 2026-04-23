#!/bin/bash
#SBATCH --job-name=pull-pytorch-ngc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Pull the latest NVIDIA NGC PyTorch SIF for DGX Spark (arm64).
# Base for all MindEye inference jobs. Sized small — I/O-bound, not CPU.

set -euo pipefail

SIF=/data/derivatives/containers/pytorch_26.03.sif
mkdir -p "$(dirname "$SIF")" /data/derivatives/rtmindeye_paper/logs

if [ -f "$SIF" ]; then
    echo "Base PyTorch SIF already present at $SIF"
    ls -la "$SIF"
    exit 0
fi

echo "Pulling nvcr.io/nvidia/pytorch:26.03-py3 → $SIF"
apptainer pull "$SIF" docker://nvcr.io/nvidia/pytorch:26.03-py3
ls -la "$SIF"
