#!/bin/bash
#SBATCH --job-name=rebuild-mindeye-img
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Rebuild the mindeye-pytorch:26.03 docker image AND save it as a tarball
# on /data so a subsequent `docker system prune` doesn't lose us the cache
# again. After save, any sbatch can `docker load -i $TAR` instead of
# rebuilding from scratch (which downloads ~15 GB of deps + builds the
# uv venv on top of NGC 26.03).
#
# Tarball lives at /data/derivatives/containers/mindeye-pytorch-26.03.tar
# (~10-15 GB compressed; /data has 38 TB free).

set -euo pipefail

CTX=/data/derivatives/containers/build_ctx_mindeye
TAR=/data/derivatives/containers/mindeye-pytorch-26.03.tar
IMAGE=mindeye-pytorch:26.03

mkdir -p /data/derivatives/rtmindeye_paper/logs

if [ ! -f "$CTX/Dockerfile" ]; then
    echo "[build] missing Dockerfile at $CTX — copying from repo"
    cp /home/mhough/dev/hippy-feat/containers/Dockerfile.mindeye_pytorch "$CTX/Dockerfile"
fi

echo "[build] starting docker build of $IMAGE from $CTX"
date -u +"%Y-%m-%dT%H:%M:%SZ"
docker build -t "$IMAGE" "$CTX"

echo "[build] image built. Saving to $TAR"
date -u +"%Y-%m-%dT%H:%M:%SZ"
docker save "$IMAGE" -o "$TAR"
echo "[build] tarball:"
ls -lh "$TAR"

# Quick verification — load it back to confirm the tar is valid
echo "[build] verifying tarball loads cleanly"
docker rmi "$IMAGE" || true
docker load -i "$TAR"
docker images | grep mindeye-pytorch || { echo "[build] FAIL: image missing after reload"; exit 1; }

echo "[build] done."
date -u +"%Y-%m-%dT%H:%M:%SZ"
