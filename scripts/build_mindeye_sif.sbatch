#!/bin/bash
#SBATCH --job-name=build-mindeye-sif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/derivatives/containers/build_logs/%x-%j.out

# Build extended NGC PyTorch SIF with MindEye runtime deps baked in.
# One-time ~30-60 min build. After this, every retrieval job uses this
# SIF directly — zero install phase, zero cascade-fixing.
#
# Background: the previous approach (NGC base + uv pip install at job
# start) ran into 2-hour install times because uv --target on NFS does
# plain copies of every wheel file (no hard-link). Three retrieval jobs
# in a row got killed by the 2 h walltime mid-install. Baking deps into
# a SIF eliminates that whole class of failure.
#
# All build temp dirs point at NAS to avoid the 97%-full local NVMe.

set -euo pipefail

DEF=/home/mhough/dev/hippy-feat/containers/mindeye_pytorch.def
OUT=/data/derivatives/containers/mindeye_pytorch_26.03.sif
LOGS=/data/derivatives/containers/build_logs
TMPROOT=/data/derivatives/apptainer_tmp

mkdir -p "$LOGS" "$TMPROOT" "$(dirname "$OUT")"

# Force every apptainer scratch + cache dir onto NAS — local nvme is 97% full
export APPTAINER_TMPDIR="$TMPROOT"
export APPTAINER_CACHEDIR="$TMPROOT/cache"
export TMPDIR="$TMPROOT"

if [ -f "$OUT" ]; then
    echo "$OUT already exists — refusing to overwrite. rm it first to rebuild."
    ls -la "$OUT"
    exit 0
fi

echo "Building $OUT from $DEF"
echo "Base: /data/derivatives/containers/pytorch_26.03.sif"
echo "Tmp:  $TMPROOT"
date

apptainer build --fakeroot "$OUT" "$DEF"

date
echo "Build complete:"
ls -la "$OUT"

echo "Smoke test:"
apptainer exec "$OUT" python -c "
import torch, open_clip, diffusers, accelerate
from diffusers.models.vae import Decoder
print(f'torch    {torch.__version__}')
print(f'CUDA     {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}')
print(f'open_clip {open_clip.__version__}')
print(f'diffusers {diffusers.__version__}')
print(f'accelerate {accelerate.__version__}')
print('Decoder imported OK from diffusers.models.vae')
print('GPU visible:', torch.cuda.is_available())
"

echo "Cleaning up scratch..."
rm -rf "$TMPROOT/cache" "$TMPROOT"/build-temp* 2>/dev/null || true
echo "Done."
