#!/bin/bash
#SBATCH --job-name=jax-arm64-sif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/derivatives/containers/build_logs/%x-%j.out
# No --time: single-user cluster, partition max is infinite.

# One-shot build of an arm64 + CUDA JAX SIF from NVIDIA NGC.
#
# Why this script exists: PyPI does not ship CUDA-enabled jaxlib for
# aarch64. The only sanctioned GPU-JAX path on GB10 is the NGC image,
# which we convert to a portable SIF for sbatch.
#
# Conversion path (docker → tar → SIF):
#   1. docker pull (parallel layer fetch, ~30s)
#   2. docker save → tar  (sequential write, ~5–10 min for 24 GB)
#   3. apptainer build SIF from docker-archive://tar  (~15–25 min mksquashfs)
#
# Why NOT `apptainer build SIF docker://...` or `docker-daemon://...`:
# both routes use apptainer's streaming OCI loader, which has a known
# goroutine-deadlock bug on large multi-layer images (see apptainer/apptainer
# issue #1019). Job 1370 hung 13 h on `docker-daemon://` with 0 CPU/0 I/O,
# all threads parked on `futex_do_wait`. The docker-archive path reads from
# a seekable file, skipping the broken streaming pipeline entirely.
#
# NGC JAX 26.04 = JAX 0.9.2 + CUDA 13.2.1 (latest as of 2026-05-24).
# Check https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/
# before bumping IMAGE.

set -euo pipefail

SIF=/data/derivatives/containers/jax-arm64-cu13.sif
IMAGE=nvcr.io/nvidia/jax:26.04-py3
# Apptainer scratch + intermediate tarball must live on LOCAL NVMe, not NFS.
# /data is NFS (NFS metadata ops kill apptainer + rm performance on the
# millions of small files in an OCI layer tree). Job 1371 wasted 31 min
# just on the pre-build `rm -rf` over NFS before we caught it.
TMPDIR_APP=/home/mhough/.cache/apptainer-tmp
TARBALL="$TMPDIR_APP/jax-26.04.tar"

mkdir -p "$(dirname "$SIF")" /data/derivatives/containers/build_logs "$TMPDIR_APP"

if [[ -f "$SIF" ]]; then
    echo "$SIF already exists. Delete it to rebuild."
    ls -la "$SIF"
    exit 0
fi

# Clear any stale partial state from the hung 1370 run.
echo "Clearing apptainer scratch at $TMPDIR_APP …"
rm -rf "$TMPDIR_APP"/* 2>/dev/null || true

echo "Step 1/3: docker pull $IMAGE"
echo "  (skipped if already in local docker images; parallel layer fetch otherwise)"
time docker pull "$IMAGE"
docker images "$IMAGE"

echo
echo "Step 2/3: docker save → $TARBALL"
echo "  (sequential tar write of ~24 GB layer tree)"
time docker save "$IMAGE" -o "$TARBALL"
ls -lh "$TARBALL"

echo
echo "Step 3/3: apptainer build SIF from docker-archive (no daemon socket)"
echo "  (mksquashfs single-threaded but reading from seekable file)"
export APPTAINER_TMPDIR="$TMPDIR_APP"
time apptainer build "$SIF" "docker-archive://$TARBALL"

# Clean up the 24 GB tarball — SIF is the artifact we keep.
rm -f "$TARBALL"

ls -la "$SIF"

echo
echo "Verifying GPU JAX inside the SIF…"
apptainer exec --nv "$SIF" python -c \
    "import jax; print('jax', jax.__version__); print('devices', jax.devices())"

echo "Done."
