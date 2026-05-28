#!/bin/bash
#SBATCH --job-name=pytorch-arm64-sif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/containers/build_logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# Build the PyTorch NGC SIF for arm64 + CUDA, matching the 26.04 release
# already used by the JAX SIF. The 26.03 SIF predates the JAX 26.04 line
# and is kept only so older mindeye scripts don't break — new work should
# target 26.04.
#
# Conversion path: docker save → docker-archive:// → apptainer build.
# Same justification as build_jax_arm64_sif.sbatch — `docker-daemon://` and
# `docker://` deadlock on large multi-layer NGC images (apptainer issue
# #1019). The docker image is already locally pulled (11 GB content,
# 34.8 GB on disk) so step 1 is a no-op.

set -euo pipefail

SIF=/data/derivatives/containers/pytorch_26.04.sif
IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
# LOCAL NVMe scratch — /data is NFS and metadata-bound rm/tar is glacial.
TMPDIR_APP=/home/mhough/.cache/apptainer-tmp
TARBALL="$TMPDIR_APP/pytorch-26.04.tar"

mkdir -p "$(dirname "$SIF")" /data/derivatives/containers/build_logs "$TMPDIR_APP"

if [[ -f "$SIF" ]]; then
    echo "$SIF already exists. Delete it to rebuild."
    ls -la "$SIF"
    exit 0
fi

echo "Clearing apptainer scratch at $TMPDIR_APP …"
rm -rf "$TMPDIR_APP"/* 2>/dev/null || true

echo "Step 1/3: docker pull $IMAGE (likely no-op — already cached)"
time docker pull "$IMAGE"
docker images "$IMAGE"

echo
echo "Step 2/3: docker save → $TARBALL"
echo "  (sequential tar write of ~24 GB layer tree, local NVMe)"
time docker save "$IMAGE" -o "$TARBALL"
ls -lh "$TARBALL"

echo
echo "Step 3/3: apptainer build SIF from docker-archive"
echo "  (mksquashfs single-threaded but reading from seekable local file)"
export APPTAINER_TMPDIR="$TMPDIR_APP"
time apptainer build "$SIF" "docker-archive://$TARBALL"

rm -f "$TARBALL"
ls -la "$SIF"

echo
echo "Verifying torch inside the SIF (no GPU yet; GPU verify is a separate srun)…"
apptainer exec --no-init "$SIF" python -c \
    "import torch; print('torch', torch.__version__); print('cuda built:', torch.version.cuda)"

echo "Done."
