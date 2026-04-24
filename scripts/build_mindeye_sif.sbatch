#!/bin/bash
#SBATCH --job-name=build-mindeye-sif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/derivatives/containers/build_logs/%x-%j.out

# Build extended NGC PyTorch SIF with MindEye runtime deps baked in,
# via docker build + apptainer convert (same pattern as Raramuri SIF).
#
# Why docker not apptainer build --fakeroot:
#   apptainer 1.4.5 + NFS tmpdir + --fakeroot fails with "failed to
#   create staging file: no such file or directory" — known issue with
#   NFS user-namespace support. Docker uses overlayfs on /data and
#   doesn't have this problem (verified for raramuri-arm64.sif).

set -euo pipefail

REPO=/home/mhough/dev/hippy-feat
CTX=/data/derivatives/containers/build_ctx_mindeye
IMAGE_TAG=mindeye-pytorch:26.03
SIF=/data/derivatives/containers/mindeye_pytorch_26.03.sif
LOGS=/data/derivatives/containers/build_logs

mkdir -p "$LOGS" "$CTX" "$(dirname "$SIF")"

if [ -f "$SIF" ]; then
    echo "$SIF already exists — refusing to overwrite. rm to rebuild."
    ls -la "$SIF"
    exit 0
fi

# Build context must include Dockerfile + sgm source
echo "Staging build context at $CTX"
cp "$REPO/containers/Dockerfile.mindeye_pytorch" "$CTX/Dockerfile"
# Copy sgm source into the build context (only needs the sgm package,
# not the whole rtcloud-projects tree)
rm -rf "$CTX/mindeye_generative_models"
cp -r /data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/models/generative_models \
      "$CTX/mindeye_generative_models"
ls "$CTX"

date
echo "Building docker image: $IMAGE_TAG"
docker build --platform linux/arm64 -t "$IMAGE_TAG" "$CTX"
date
docker images "$IMAGE_TAG"

echo ""
echo "Converting docker → SIF at $SIF…"
# apptainer pull from docker-daemon avoids NFS fakeroot issues since
# the conversion reads docker's overlayfs (already on NAS) and writes
# the SIF directly. We point TMPDIR at NAS too for the squashfs build.
export APPTAINER_TMPDIR=/data/derivatives/apptainer_tmp
mkdir -p "$APPTAINER_TMPDIR"

apptainer build "$SIF" "docker-daemon://$IMAGE_TAG"
ls -la "$SIF"

echo ""
echo "Smoke test:"
apptainer exec "$SIF" python -c "
import torch, open_clip, diffusers, accelerate
from diffusers.models.vae import Decoder
print(f'torch       {torch.__version__}')
print(f'cuda        {torch.version.cuda}')
print(f'open_clip   {open_clip.__version__}')
print(f'diffusers   {diffusers.__version__}')
print(f'accelerate  {accelerate.__version__}')
print('Decoder import OK')
"
echo "Done."

# Optional: clean up build context if successful
rm -rf "$CTX/mindeye_generative_models"
