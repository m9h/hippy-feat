#!/usr/bin/env python3
"""
MindEye Inference on Variant Betas

Requires NVIDIA NGC PyTorch container (no native PyTorch on DGX Spark).
Run with:
    docker run --gpus all -v /data:/data -v /home/mhough/dev:/workspace \
        nvcr.io/nvidia/pytorch:25.03-py3 \
        python3 /workspace/hippy-feat/scripts/run_mindeye_inference.py

Or interactively:
    docker run --gpus all -it -v /data:/data -v /home/mhough:/home/mhough \
        nvcr.io/nvidia/pytorch:25.03-py3 bash

Dependencies (install inside container):
    pip install accelerate omegaconf
    # MindEye generative_models needs to be available on path
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Check for PyTorch
try:
    import torch
    assert torch.cuda.is_available(), "CUDA required — run inside NGC container"
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0)}")
except (ImportError, AssertionError) as e:
    print(f"ERROR: {e}")
    print("This script requires GPU PyTorch. Run inside an NVIDIA NGC container:")
    print("  docker run --gpus all -v /data:/data nvcr.io/nvidia/pytorch:25.03-py3 \\")
    print("    python3 scripts/run_mindeye_inference.py")
    sys.exit(1)


# Paths
VARIANT_BASE = Path("/data/derivatives/mindeye_variants")
MODEL_PATH = Path("/data/3t/data/model/sub-005_all_task-C_bs24_MST_rishab_MSTsplit_union_mask_finetune_0.pth")
MINDEYE_SCRIPTS = Path("/home/mhough/.gemini/antigravity/playground/cobalt-cosmos/mindeye/scripts")
COMP_DIR = VARIANT_BASE / "comparison"


def load_model(device="cuda"):
    """Load frozen MindEye model checkpoint."""
    sys.path.insert(0, str(MINDEYE_SCRIPTS))
    sys.path.insert(0, str(MINDEYE_SCRIPTS.parent))

    # The model architecture is defined in mindeye's models.py
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint keys: {list(checkpoint.keys())[:10]}")
    return checkpoint


def run_inference_variant(model, variant_name: str, device="cuda") -> Dict:
    """Run MindEye inference on all trial betas for one variant."""
    beta_dir = VARIANT_BASE / f"variant_{variant_name}" / "betas"
    beta_files = sorted(beta_dir.glob("run-01_tr-*.npy"))

    results = {"variant": variant_name, "n_trials": len(beta_files), "clip_embeddings": []}

    for bf in beta_files:
        beta = np.load(bf)
        # Model expects (1, 1, 8627) tensor
        betas_tt = torch.tensor(beta[np.newaxis, np.newaxis, :], dtype=torch.float32).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            voxel_ridge = model.ridge(betas_tt[:, [-1]], 0)
            _, clip_voxels, _ = model.backbone(voxel_ridge)
            results["clip_embeddings"].append(clip_voxels.cpu().numpy())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["a_baseline", "c_pervoxel_hrf"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading MindEye model...")
    model = load_model(args.device)

    all_results = {}
    for vname in args.variants:
        print(f"\nRunning inference: {vname}")
        t0 = time.time()
        result = run_inference_variant(model, vname, args.device)
        elapsed = time.time() - t0
        print(f"  {result['n_trials']} trials in {elapsed:.1f}s")
        all_results[vname] = result

    # Save CLIP embeddings for comparison
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    for vname, result in all_results.items():
        if result["clip_embeddings"]:
            embeds = np.concatenate(result["clip_embeddings"], axis=0)
            np.save(COMP_DIR / f"clip_embeddings_{vname}.npy", embeds)
            print(f"Saved CLIP embeddings: {embeds.shape}")


if __name__ == "__main__":
    main()
