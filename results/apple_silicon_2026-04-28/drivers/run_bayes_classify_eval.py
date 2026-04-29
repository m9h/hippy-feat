#!/usr/bin/env python3
"""Bayesian classification eval for Variant G cells.

For each special515 trial:
  1. MC sample K=100 β ~ N(β_mean, diag(β_var))
  2. Forward each sample: ridge → BrainNetwork → CLIP token embedding (256, 1664)
  3. Cosine sim to 50 GT image CLIP embeddings → 50-way distribution per sample
  4. Aggregate across K samples → empirical posterior over 50 images per trial

Reports:
  - Posterior top-1 / top-5 (point-estimate baseline at β_mean)
  - Confidence of modal class
  - Brier score
  - Selective accuracy curve at τ ∈ {0.3, 0.5, 0.7, 0.9}
  - Calibration curve (10 bins)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")

# Stubs (Decoder + sgm) — same as run_retrieval_local.py
import diffusers, diffusers.models  # noqa
vae_mod = types.ModuleType("diffusers.models.vae")
class _DecStub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
vae_mod.Decoder = _DecStub
sys.modules["diffusers.models.vae"] = vae_mod
diffusers.models.vae = vae_mod

gm = types.ModuleType("generative_models")
sgm = types.ModuleType("generative_models.sgm")
sgm_util = types.ModuleType("generative_models.sgm.util")
sgm_modules = types.ModuleType("generative_models.sgm.modules")
sgm_enc = types.ModuleType("generative_models.sgm.modules.encoders")
sgm_enc_mods = types.ModuleType("generative_models.sgm.modules.encoders.modules")
sgm_util.append_dims = lambda x, n: x
class _Stub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
sgm_enc_mods.FrozenOpenCLIPImageEmbedder = _Stub
sgm_enc_mods.FrozenOpenCLIPEmbedder2 = _Stub
for mod in [gm, sgm, sgm_util, sgm_modules, sgm_enc, sgm_enc_mods]:
    sys.modules[mod.__name__] = mod
sys.modules["sgm"] = sgm

sys.path.insert(0, str(RT_MINDEYE))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import mindeye_retrieval_eval as M
M.RTCLOUD_MINDEYE = RT_MINDEYE
from run_retrieval_local import compute_gt_mps


def _autocast_for(device: str):
    if device.startswith("mps"):
        return torch.amp.autocast("mps", dtype=torch.float16)
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cpu", dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="cell name with `_with_vars` suffix")
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--n_mc", type=int, default=100, help="MC posterior samples per trial")
    ap.add_argument("--checkpoint", default=str(
        LOCAL / "rt3t" / "data" / "model" /
        "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
    ))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-json", default=str(LOCAL / "task_2_1_betas" / "bayes_eval_results.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    device = args.device if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}  cell={args.cell}  n_mc={args.n_mc}")

    # 1. Load β_mean, β_var, trial_ids
    pre = LOCAL / "task_2_1_betas" / "prereg"
    betas = np.load(pre / f"{args.cell}_{args.session}_betas.npy")          # (N, V)
    vars_ = np.load(pre / f"{args.cell}_{args.session}_vars.npy")           # (N, V)
    ids = np.load(pre / f"{args.cell}_{args.session}_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    print(f"  loaded β {betas.shape}, var mean={vars_.mean():.4f}")

    # 2. Causal cumulative z-score (same fix as v2 retrieval; cell wasn't pre-z'd)
    n, V = betas.shape
    z_mean = np.zeros_like(betas, dtype=np.float32)
    cum = np.zeros(V, dtype=np.float64)
    cum_sq = np.zeros(V, dtype=np.float64)
    for i in range(n):
        if i == 0:
            z_mean[0] = 0.0
        else:
            mu = cum / i
            sd = np.sqrt(np.maximum(cum_sq / i - mu**2, 1e-12))
            z_mean[i] = ((betas[i] - mu) / (sd + 1e-8)).astype(np.float32)
        cum += betas[i]
        cum_sq += betas[i].astype(np.float64) ** 2
    # Variance also gets divided by sd² (delta method on linear standardization)
    # Apply same per-trial sd:
    z_var = np.zeros_like(vars_, dtype=np.float32)
    cum.fill(0.0); cum_sq.fill(0.0)
    for i in range(n):
        if i > 0:
            mu = cum / i
            sd = np.sqrt(np.maximum(cum_sq / i - mu**2, 1e-12)) + 1e-8
            z_var[i] = (vars_[i] / (sd**2)).astype(np.float32)
        cum += betas[i]
        cum_sq += betas[i].astype(np.float64) ** 2

    # 3. Filter to special515 test trials
    mask = np.array([t.startswith("all_stimuli/special515/") for t in ids])
    z_mean_t = z_mean[mask]
    z_var_t = z_var[mask]
    ids_t = ids[mask]
    unique_images = np.array(sorted(set(ids_t)))
    img_to_idx = {u: i for i, u in enumerate(unique_images)}
    trial_image_idx = np.array([img_to_idx[t] for t in ids_t])
    print(f"  {z_mean_t.shape[0]} test trials, {len(unique_images)} unique images")

    # 4. GT CLIP embeddings (cached)
    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in unique_images]
    cache_dir = LOCAL / "task_2_1_betas" / "gt_cache"
    gt_emb = compute_gt_mps(image_paths, device=device, cache_dir=cache_dir)
    print(f"  gt: {gt_emb.shape}")

    # 5. MindEye model + forward
    print("  loading MindEye...")
    model, ss, se = M.load_mindeye(Path(args.checkpoint),
                                    n_voxels=z_mean_t.shape[1], device=device)

    # Flatten gt for cosine
    gt_flat = gt_emb.reshape(gt_emb.shape[0], -1)
    gt_flat = gt_flat / (np.linalg.norm(gt_flat, axis=1, keepdims=True) + 1e-8)

    # 6. MC sample + forward per trial
    rng = np.random.default_rng(0)
    n_trials = z_mean_t.shape[0]
    n_imgs = len(unique_images)
    posteriors = np.zeros((n_trials, n_imgs), dtype=np.float32)  # empirical p(image | trial)
    pred_at_mean = np.zeros((n_trials, n_imgs), dtype=np.float32)  # baseline: forward β_mean

    sd_t = np.sqrt(np.maximum(z_var_t, 1e-10))                 # (N_test, V)

    print(f"  MC sampling {args.n_mc} draws × {n_trials} trials...")
    t0 = time.time()
    with torch.no_grad(), _autocast_for(device):
        # Baseline: just β_mean
        for i in range(n_trials):
            b = torch.from_numpy(z_mean_t[i:i+1].astype(np.float32)).to(device).unsqueeze(1)
            voxel_ridge = model.ridge(b, 0)
            backbone_out = model.backbone(voxel_ridge)
            cv = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            p = cv.float().cpu().numpy().reshape(-1)
            p = p / (np.linalg.norm(p) + 1e-8)
            sims = (gt_flat @ p)
            pred_at_mean[i] = sims

        # MC posterior
        for s in range(args.n_mc):
            eps = rng.standard_normal(size=z_mean_t.shape).astype(np.float32)
            beta_sample = z_mean_t + sd_t * eps                 # (N_test, V)
            for i in range(n_trials):
                b = torch.from_numpy(beta_sample[i:i+1]).to(device).unsqueeze(1)
                voxel_ridge = model.ridge(b, 0)
                backbone_out = model.backbone(voxel_ridge)
                cv = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
                p = cv.float().cpu().numpy().reshape(-1)
                p = p / (np.linalg.norm(p) + 1e-8)
                sims = (gt_flat @ p)
                # Mark which image is top-1 for this MC sample
                top = sims.argmax()
                posteriors[i, top] += 1.0
            if (s + 1) % 20 == 0:
                print(f"    MC {s+1}/{args.n_mc}  ({time.time()-t0:.1f}s elapsed)")
    posteriors /= args.n_mc
    print(f"  total MC time: {time.time()-t0:.1f}s")

    # 7. Metrics
    # Point-estimate (β_mean) top-1/top-5
    ranks_mean = (-pred_at_mean).argsort(axis=1)
    pe_top1 = float(np.mean(ranks_mean[:, 0] == trial_image_idx))
    pe_top5 = float(np.mean([trial_image_idx[i] in ranks_mean[i, :5] for i in range(n_trials)]))

    # Posterior mode
    post_top1_idx = posteriors.argmax(axis=1)
    post_top1_acc = float(np.mean(post_top1_idx == trial_image_idx))
    post_top1_conf = posteriors[np.arange(n_trials), post_top1_idx]    # confidence
    correct = (post_top1_idx == trial_image_idx)

    # Brier score over all classes
    one_hot = np.zeros_like(posteriors)
    one_hot[np.arange(n_trials), trial_image_idx] = 1.0
    brier = float(np.mean(np.sum((posteriors - one_hot) ** 2, axis=1)))

    # Selective accuracy at τ thresholds
    selective = {}
    for tau in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        keep = post_top1_conf >= tau
        if keep.sum() == 0:
            selective[f"tau_{tau}"] = {"coverage": 0.0, "accuracy": float("nan")}
        else:
            acc = float(correct[keep].mean())
            cov = float(keep.mean())
            selective[f"tau_{tau}"] = {"coverage": cov, "accuracy": acc}

    # Calibration: 10 equal-width bins on confidence
    bins = np.linspace(0, 1, 11)
    cal = []
    for k in range(10):
        lo, hi = bins[k], bins[k+1]
        mask = (post_top1_conf >= lo) & (post_top1_conf < hi if k < 9 else post_top1_conf <= hi)
        if mask.sum() == 0:
            cal.append({"bin": [float(lo), float(hi)], "n": 0,
                        "mean_conf": float("nan"), "empirical_acc": float("nan")})
        else:
            cal.append({"bin": [float(lo), float(hi)], "n": int(mask.sum()),
                        "mean_conf": float(post_top1_conf[mask].mean()),
                        "empirical_acc": float(correct[mask].mean())})

    # Expected calibration error (ECE)
    ece = 0.0
    for c in cal:
        if c["n"] > 0:
            ece += (c["n"] / n_trials) * abs(c["mean_conf"] - c["empirical_acc"])
    ece = float(ece)

    out = {
        "cell": args.cell, "n_trials": int(n_trials), "n_images": int(n_imgs),
        "n_mc": args.n_mc, "device": device,
        "point_estimate_top1": pe_top1,
        "point_estimate_top5": pe_top5,
        "posterior_mode_top1": post_top1_acc,
        "brier_score": brier,
        "ECE": ece,
        "selective_accuracy": selective,
        "calibration_bins": cal,
        "var_mean_z": float(z_var_t.mean()),
        "var_mean_raw": float(vars_.mean()),
    }
    op = Path(args.out_json)
    existing = json.loads(op.read_text()) if op.exists() else []
    if isinstance(existing, dict): existing = [existing]
    existing.append(out)
    op.write_text(json.dumps(existing, indent=2))

    print()
    print(f"  point-estimate top-1: {pe_top1:.4f}   top-5: {pe_top5:.4f}")
    print(f"  posterior-mode top-1: {post_top1_acc:.4f}")
    print(f"  Brier: {brier:.4f}   ECE: {ece:.4f}")
    print(f"  selective accuracy:")
    for tau, v in selective.items():
        print(f"    {tau}: cov={v['coverage']:.3f}  acc={v['accuracy']:.4f}")


if __name__ == "__main__":
    main()
