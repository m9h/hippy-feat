#!/usr/bin/env python3
"""Full Table-1-style metric scorer (paper Iyer et al. ICML 2026).

For each cell in CELLS, computes all 10 metrics from paper Table 1:
  Reconstruction (each averaged over 5 random diffusion-prior seeds):
    PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EfficientNet, SwAV
  Retrieval (deterministic):
    Image top-1, Brain top-1

Output: JSON per cell with per-metric values + 5-seed std + n_trials.

Designed to run inside the MindEye Docker container so the SDXL +
diffusion-prior + perceptual-feature-extractor stack is available.

Usage:
    python score_full_metrics.py --cells {cell_name1,cell_name2,...} \
        --checkpoint <path-to-finalmask-MindEye-ckpt> \
        --n-seeds 5 --first-rep --num-voxels 2792
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Path setup: paper's RT-cloud project supplies model classes + metric helpers
# ---------------------------------------------------------------------------
PAPER_REPO = Path("/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/scripts")
RT_MINDEYE_SRC = Path("/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/rt_mindEye2/src")
GENMODELS = Path("/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/models")
sys.path.insert(0, str(PAPER_REPO))
sys.path.insert(0, str(RT_MINDEYE_SRC))
sys.path.insert(0, str(GENMODELS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# xformers is installed in the container but GB10 has no working
# memory_efficient_attention kernel for the SDXL input shapes. Monkey-patch
# the function with a torch SDPA wrapper that handles xformers' (B, M, H, K)
# layout by transposing to (B, H, M, K) for SDPA then back.
import torch.nn.functional as _F
try:
    import xformers  # type: ignore
    import xformers.ops as _xops  # type: ignore

    def _sdpa_attention(q, k, v, attn_bias=None, p=0.0, scale=None, op=None):
        # xformers layout: (B, M, H, K). torch SDPA: (B, H, M, K).
        q4 = q.transpose(1, 2) if q.dim() == 4 else q
        k4 = k.transpose(1, 2) if k.dim() == 4 else k
        v4 = v.transpose(1, 2) if v.dim() == 4 else v
        out = _F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_bias,
                                                dropout_p=p, scale=scale)
        return out.transpose(1, 2).contiguous() if q.dim() == 4 else out

    _xops.memory_efficient_attention = _sdpa_attention
except ImportError:
    pass

import utils_mindeye  # noqa: E402
from models import BrainNetwork  # type: ignore  # noqa: E402

PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
PRE = PAPER_ROOT / "task_2_1_betas" / "prereg"
RT3T = PAPER_ROOT / "rt3t" / "data"
GT_CACHE = PAPER_ROOT / "task_2_1_betas" / "gt_cache"
STIMULI = RT3T / "all_stimuli" / "special515"
RT_ALL_DATA = Path("/data/rt_all_data")
DEFAULT_CKPT = (
    PAPER_ROOT
    / "checkpoints/data_scaling_exp/concat_glmsingle/checkpoints/"
    / "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_avgrepeats_finalmask_epochs_150.pth"
)


# ---------------------------------------------------------------------------
# Model setup — copies from rtcloud-projects mindeye.py:107-289
# ---------------------------------------------------------------------------
HIDDEN_DIM = 1024
N_BLOCKS = 4
SEQ_LEN = 1
CLIP_SEQ_DIM = 256
CLIP_EMB_DIM = 1664
IMSIZE = 224


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # noqa: D401
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features, seq_len):
        super().__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList(
            [nn.Linear(input_size, out_features) for input_size in input_sizes]
        )

    def forward(self, x, subj_idx):
        return torch.cat(
            [self.linears[subj_idx](x[:, seq]).unsqueeze(1) for seq in range(SEQ_LEN)],
            dim=1,
        )


def build_mindeye(num_voxels: int, device: str) -> nn.Module:
    """Construct the MindEye2 architecture used by paper RT-cloud script."""
    from models import PriorNetwork, BrainDiffusionPrior  # type: ignore

    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=HIDDEN_DIM, seq_len=SEQ_LEN)
    model.backbone = BrainNetwork(
        h=HIDDEN_DIM, in_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
        clip_size=CLIP_EMB_DIM, out_dim=CLIP_EMB_DIM * CLIP_SEQ_DIM,
        n_blocks=N_BLOCKS,
        blurry_recon=False,  # fold-10 paper ckpt has no low-level submodule
    )
    out_dim = CLIP_EMB_DIM
    depth, dim_head = 6, 52
    heads = CLIP_EMB_DIM // dim_head
    timesteps = 100
    prior_network = PriorNetwork(
        dim=out_dim, depth=depth, dim_head=dim_head, heads=heads,
        causal=False, num_tokens=CLIP_SEQ_DIM, learned_query_mode="pos_emb",
    )
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    model.to(device)
    return model


def load_diffusion_engine(device: str):
    import pickle
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(GENMODELS / "generative_models/configs/unclip6.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["model"]["params"]["sampler_config"]["params"]["num_steps"] = 38
    with open(RT_ALL_DATA / "diffusion_engine", "rb") as f:
        engine = pickle.load(f)
    engine.eval().requires_grad_(False).to(device)
    ckpt = torch.load(RT_ALL_DATA / "cache/unclip6_epoch0_step110000.ckpt",
                       map_location="cpu", weights_only=False)
    engine.load_state_dict(ckpt["state_dict"])
    batch = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
    }
    out = engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    return engine, vector_suffix


def load_clip_img_embedder(device: str):
    import pickle
    with open(RT_ALL_DATA / "clip_img_embedder", "rb") as f:
        emb = pickle.load(f)
    emb.to(device)
    return emb


# ---------------------------------------------------------------------------
# Cell I/O + first-rep filtering
# ---------------------------------------------------------------------------
def load_cell(cell: str) -> tuple[np.ndarray, np.ndarray]:
    betas = np.load(PRE / f"{cell}_ses-03_betas.npy")
    ids = np.load(PRE / f"{cell}_ses-03_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, ids


def cumulative_zscore(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[0]
    for i in range(n):
        if i < 2:
            mu = arr[:max(i, 1)].mean(0, keepdims=True) if i > 0 else 0.0
            sd = 1.0
        else:
            mu = arr[:i].mean(0, keepdims=True)
            sd = arr[:i].std(0, keepdims=True) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out


def filter_first_rep_special515(betas: np.ndarray, ids: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    seen = set()
    keep = np.zeros(len(ids), dtype=bool)
    for i, t in enumerate(ids):
        if t.startswith("all_stimuli/special515/") and t not in seen:
            keep[i] = True
            seen.add(t)
    return betas[keep], ids[keep]


def load_gt_images(image_names: np.ndarray, device: str) -> torch.Tensor:
    """Load + resize ground-truth special515 images as (N, 3, 224, 224)."""
    tx = transforms.Compose([
        transforms.Resize((IMSIZE, IMSIZE), antialias=True),
        transforms.ToTensor(),
    ])
    out = []
    for name in image_names:
        p = STIMULI / Path(name).name
        out.append(tx(Image.open(p).convert("RGB")))
    return torch.stack(out, 0).to(device)


# ---------------------------------------------------------------------------
# Reconstruction (5-seed averaging happens at the metric level)
# ---------------------------------------------------------------------------
def reconstruct_one_seed(model, diffusion_engine, vector_suffix,
                          backbone: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        prior_out = model.diffusion_prior.p_sample_loop(
            backbone.shape, text_cond=dict(text_embed=backbone),
            cond_scale=1.0, timesteps=20,
        )
        recons = []
        for i in range(len(backbone)):
            samp = utils_mindeye.unclip_recon(
                prior_out[[i]], diffusion_engine, vector_suffix,
                num_samples=1,
            )
            recons.append(samp.cpu())
    out = torch.vstack(recons)
    return transforms.Resize((IMSIZE, IMSIZE), antialias=True)(out)


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------
def score_cell(cell: str, model, clip_img_embedder,
                diffusion_engine, vector_suffix,
                n_seeds: int, first_rep: bool, device: str) -> dict:
    print(f"\n=== {cell} ===", flush=True)
    betas_all, ids_all = load_cell(cell)
    if not (cell.startswith("RT_paper_replica") or
            cell.startswith("Offline_paper_replica") or
            cell.startswith("RT_streaming_pst") or
            cell.startswith("Probe_canonical_") or
            cell.startswith("Canonical_GLMsingle")):
        betas_all = cumulative_zscore(betas_all)

    if first_rep:
        betas, ids = filter_first_rep_special515(betas_all, ids_all)
    else:
        spec_mask = np.array([t.startswith("all_stimuli/special515/")
                                for t in ids_all])
        betas, ids = betas_all[spec_mask], ids_all[spec_mask]
    n_trials = betas.shape[0]
    print(f"  {n_trials} test trials  (first_rep={first_rep})", flush=True)

    # Forward to backbone (deterministic)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        voxel = torch.from_numpy(betas).float().to(device).unsqueeze(1)
        ridge_out = model.ridge(voxel, 0)
        backbone, clip_voxels, _ = model.backbone(ridge_out)
    print(f"  backbone {backbone.shape}  clip_voxels {clip_voxels.shape}", flush=True)

    # GT images for retrieval + perceptual metrics
    gt_imgs = load_gt_images(ids, device)
    print(f"  GT images {gt_imgs.shape}", flush=True)

    # 5-seed averaged reconstructions — store all seeds as (n_seeds, N, 3, H, W)
    all_recons = []
    for s in range(n_seeds):
        torch.manual_seed(1234 + s)
        t0 = time.time()
        recon = reconstruct_one_seed(
            model, diffusion_engine, vector_suffix, backbone, device,
        )
        all_recons.append(recon)
        print(f"  seed {s + 1}/{n_seeds} done in {time.time() - t0:.1f}s",
              flush=True)
    all_recons = torch.stack(all_recons, 0)                          # (S, N, 3, H, W)

    # ---- metrics ----------------------------------------------------------
    from utils_mindeye import (
        calculate_pixcorr, calculate_ssim, calculate_alexnet,
        calculate_inception_v3, calculate_clip, calculate_efficientnet_b1,
        calculate_swav,
    )

    metrics = {
        "pixcorr": [], "ssim": [], "alexnet2": [], "alexnet5": [],
        "inception": [], "clip": [], "efficientnet": [], "swav": [],
    }
    for s in range(n_seeds):
        recons_s = all_recons[s].to(device).to(torch.float16)
        gt = gt_imgs.to(torch.float16)
        metrics["pixcorr"].append(float(calculate_pixcorr(recons_s, gt)))
        metrics["ssim"].append(float(calculate_ssim(recons_s, gt)))
        a2, a5 = calculate_alexnet(recons_s, gt)
        metrics["alexnet2"].append(float(a2))
        metrics["alexnet5"].append(float(a5))
        metrics["inception"].append(float(calculate_inception_v3(recons_s, gt)))
        metrics["clip"].append(float(calculate_clip(recons_s, gt)))
        metrics["efficientnet"].append(
            float(calculate_efficientnet_b1(recons_s, gt))
        )
        metrics["swav"].append(float(calculate_swav(recons_s, gt)))
        print(f"  seed {s + 1} metrics: pixcorr={metrics['pixcorr'][-1]:.3f}  "
              f"alex2={metrics['alexnet2'][-1]:.3f}  "
              f"clip={metrics['clip'][-1]:.3f}", flush=True)

    # Retrieval (deterministic — no seed averaging)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        gt_emb = clip_img_embedder(gt_imgs).float().cpu()             # (N, 256, 1664)
        gt_flat = nn.functional.normalize(gt_emb.reshape(len(gt_emb), -1), dim=-1)
        cv_flat = nn.functional.normalize(
            clip_voxels.cpu().reshape(len(clip_voxels), -1), dim=-1,
        )
        # Image retrieval: brain → image (find correct GT given brain pred)
        sim = (cv_flat @ gt_flat.T).numpy()                          # (N, N)
        fwd_top1 = float((np.argmax(sim, axis=1) ==
                           np.arange(len(sim))).mean())
        # Brain retrieval: image → brain
        sim_t = sim.T
        bwd_top1 = float((np.argmax(sim_t, axis=1) ==
                           np.arange(len(sim_t))).mean())

    summary = {
        "cell": cell,
        "n_trials": n_trials,
        "n_seeds": n_seeds,
        "first_rep": first_rep,
        "image_retrieval_top1": fwd_top1,
        "brain_retrieval_top1": bwd_top1,
    }
    for k, vals in metrics.items():
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
        summary[f"{k}_per_seed"] = vals
    print(f"  summary: img-ret={fwd_top1:.3f}  brain-ret={bwd_top1:.3f}  "
          f"pixcorr={summary['pixcorr_mean']:.3f}±{summary['pixcorr_std']:.3f}",
          flush=True)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--num-voxels", type=int, default=2792)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--first-rep", action="store_true",
                     help="filter to single-trial first-rep n=50 (paper Table 1 default)")
    ap.add_argument("--out", default=str(PRE / "full_metrics.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)
    print(f"checkpoint: {Path(args.checkpoint).name}", flush=True)
    print(f"num_voxels={args.num_voxels} n_seeds={args.n_seeds} "
          f"first_rep={args.first_rep}", flush=True)

    print("\n[1] building MindEye + loading checkpoint", flush=True)
    model = build_mindeye(num_voxels=args.num_voxels, device=device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)

    print("\n[2] loading diffusion engine + clip_img_embedder", flush=True)
    diffusion_engine, vector_suffix = load_diffusion_engine(device)
    clip_img_embedder = load_clip_img_embedder(device)

    print("\n[3] scoring cells", flush=True)
    out = []
    for cell in args.cells:
        try:
            r = score_cell(cell, model, clip_img_embedder, diffusion_engine,
                            vector_suffix, args.n_seeds, args.first_rep, device)
            out.append(r)
        except Exception as e:
            print(f"  ERROR on {cell}: {e}", flush=True)
            out.append({"cell": cell, "error": str(e)})

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
