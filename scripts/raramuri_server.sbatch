#!/bin/bash
#SBATCH --job-name=raramuri-server
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/derivatives/raramuri/logs/%x-%j.out

# Long-running Raramuri hot server. Keeps all five models GPU-resident
# (PERSIST_EXTRACTOR_MODELS=1) so every /infer request is <2s after warm.
#
# Pre-req:  the 875/vbjax-analytica GPU job must have released the GPU.
#           HF_TOKEN must have Llama 3.2 3B access.
#
# Server listens on port $PORT; clients inside the node reach it at
# http://localhost:$PORT. Outside-node access needs Slurm port-forward or
# sshuttle — not needed for jaxoccoli (runs on the same node).

set -euo pipefail

PORT=${RARAMURI_PORT:-8765}
SIF=/data/derivatives/containers/raramuri-arm64.sif
MODELS_DIR=/data/derivatives/raramuri/models
TRIBE_CACHE=/data/derivatives/raramuri/tribe_cache
HF_CACHE=/data/derivatives/raramuri/hf_cache
HF_TOKEN_FILE=${HF_HOME:-$HOME/.cache/huggingface}/token

mkdir -p /data/derivatives/raramuri/logs "$MODELS_DIR" "$TRIBE_CACHE" "$HF_CACHE"

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found. Submit raramuri_build.sbatch first." >&2
    exit 1
fi

if [ ! -s "$HF_TOKEN_FILE" ]; then
    echo "ERROR: no HF token at $HF_TOKEN_FILE" >&2
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")

echo "Starting Raramuri server on port $PORT (node $(hostname))"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Image: $SIF"

apptainer run --cleanenv \
    --nv \
    --env HF_TOKEN="$HF_TOKEN" \
    --env RARAMURI_TRANSCRIPT_BACKEND=parakeet \
    --env RARAMURI_PERSIST_EXTRACTOR_MODELS=1 \
    --env RARAMURI_VJEPA_DTYPE=bf16 \
    --env RARAMURI_AUDIO_DTYPE=bf16 \
    --env RARAMURI_TEXT_DTYPE=bf16 \
    --env RARAMURI_PARALLEL_EXTRACTORS=2 \
    -B "$MODELS_DIR":/models \
    -B "$TRIBE_CACHE":/models/tribe \
    -B "$HF_CACHE":/models/hf \
    --entrypoint python \
    "$SIF" \
    /app/infer_server.py --port "$PORT" --host 0.0.0.0

unset HF_TOKEN
