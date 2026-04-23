#!/bin/bash
#SBATCH --job-name=raramuri-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/derivatives/raramuri/logs/%x-%j.out

# Raramuri's published ghcr image is amd64 only; the Dockerfile comment
# in the repo explicitly says the arm64 build is meant to happen locally
# on GB10. So we git-clone, docker build on this arm64 host, and optionally
# convert to SIF. The base image (nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04)
# has a native arm64 manifest so buildx picks it up automatically.

set -euo pipefail

REPO_DIR=/data/derivatives/raramuri/src
IMAGE_TAG=raramuri:local-arm64
SIF=/data/derivatives/containers/raramuri-arm64.sif

mkdir -p "$(dirname "$REPO_DIR")" "$(dirname "$SIF")" /data/derivatives/raramuri/logs

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning Raramuri…"
    git clone --depth 1 https://github.com/chrisvoncsefalvay/raramuri.git "$REPO_DIR"
else
    echo "Updating Raramuri clone…"
    git -C "$REPO_DIR" fetch --depth 1 origin main
    git -C "$REPO_DIR" reset --hard origin/main
fi

cd "$REPO_DIR"

echo "Building arm64 image: $IMAGE_TAG"
docker build \
    --platform linux/arm64 \
    -t "$IMAGE_TAG" \
    -f docker/Dockerfile \
    .

echo "Docker image built:"
docker images "$IMAGE_TAG"

echo "Converting to SIF at $SIF…"
apptainer build "$SIF" "docker-daemon://$IMAGE_TAG"

ls -la "$SIF"
echo "Done."
