#!/usr/bin/env python3
"""Unified scorer: compute ALL neurofeedback-relevant metrics per cell.

For each cell + checkpoint:
  - top-1 / top-5 (50-way retrieval; paper "Image↑")
  - brain retrieval bwd_acc (50-way; paper "Brain↑")
  - 2-AFC pairwise accuracy (chance=50%)
  - merge/separate AUC + Cohen's d (closed-loop neurofeedback target)
  - Brier score on softmax-of-similarity (calibration)
  - ECE (expected calibration error)
  - selective accuracy at τ ∈ {0.5, 0.7, 0.9}
  - β-reliability across the 3 reps of each test image (rep-rep correlation)

Usage:
  python score_unified_metrics.py [--ckpt fold0|fold10] [--cells cellA,cellB,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")

import diffusers, diffusers.models  # noqa
vae_mod = types.ModuleType("diffusers.models.vae")
class _Stub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
vae_mod.Decoder = _Stub
sys.modules["diffusers.models.vae"] = vae_mod
diffusers.models.vae = vae_mod
gm = types.ModuleType("generative_models")
sgm = types.ModuleType("generative_models.sgm")
sgm_util = types.ModuleType("generative_models.sgm.util")
sgm_modules = types.ModuleType("generative_models.sgm.modules")
sgm_enc = types.ModuleType("generative_models.sgm.modules.encoders")
sgm_enc_mods = types.ModuleType("generative_models.sgm.modules.encoders.modules")
sgm_util.append_dims = lambda x, n: x
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

warnings.filterwarnings("ignore")

PREREG = LOCAL / "task_2_1_betas" / "prereg"
CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
CKPTS = {
    "fold0":  LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth",
    "fold10": LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth",
}
device = "mps" if torch.backends.mps.is_available() else "cpu"


# -------- Metrics ------------------------------------------------------------

def top_k_acc(sim: np.ndarray, gt_idx: np.ndarray, k: int) -> float:
    """sim[i, j] = similarity between query i and candidate j; gt_idx[i] = true j."""
    n = sim.shape[0]
    topk = np.argsort(sim, axis=1)[:, -k:]
    correct = np.array([gt_idx[i] in topk[i] for i in range(n)])
    return float(correct.mean())


def brain_retrieval_acc(sim: np.ndarray, k: int = 1) -> float:
    """Backward retrieval: GT[i] retrieves pred[i] as nearest. Use sim.T."""
    n = sim.shape[0]
    correct = np.array([np.argmax(sim[:, i]) == i for i in range(n)])
    return float(correct.mean())


def two_afc_acc(sim: np.ndarray) -> float:
    """For each query i, compare correct candidate i vs random distractor j;
    accuracy = fraction where sim[i,i] > sim[i,j]. Average over all i and j≠i.
    chance = 50%."""
    n = sim.shape[0]
    correct = 0; total = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if sim[i, i] > sim[i, j]: correct += 1
            total += 1
    return correct / total


def merge_separate(sim: np.ndarray):
    """For each pair (i,j): compute pairwise similarity. Same-image (i==j)
    distribution vs different-image (i≠j). Returns AUC + Cohen's d."""
    n = sim.shape[0]
    # 'merge' = diagonal entries (same query i, candidate i)
    merge = np.diag(sim)
    # 'separate' = off-diagonal
    mask = ~np.eye(n, dtype=bool)
    sep = sim[mask]
    # Cohen's d
    d = (merge.mean() - sep.mean()) / np.sqrt(0.5 * (merge.var() + sep.var()) + 1e-12)
    # ROC-AUC = Mann-Whitney U / (n_pos × n_neg) — rank-based, no sklearn needed
    s = np.concatenate([merge, sep])
    y = np.concatenate([np.ones(len(merge)), np.zeros(len(sep))])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = (y == 1).sum(); n_neg = (y == 0).sum()
    sum_pos_ranks = ranks[y == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), float(d)


def brier(softmax_probs: np.ndarray, gt_idx: np.ndarray) -> float:
    """Brier score over multiclass: mean (p_i - y_i)^2, with y one-hot."""
    n, c = softmax_probs.shape
    y = np.zeros_like(softmax_probs)
    y[np.arange(n), gt_idx] = 1.0
    return float(((softmax_probs - y) ** 2).sum(axis=1).mean())


def ece(softmax_probs: np.ndarray, gt_idx: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: average gap between max-prob and accuracy
    within bins of max-prob."""
    n = softmax_probs.shape[0]
    confidences = softmax_probs.max(axis=1)
    predictions = softmax_probs.argmax(axis=1)
    correct = (predictions == gt_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0: continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        err += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(err)


def selective_acc(softmax_probs: np.ndarray, gt_idx: np.ndarray, taus: list[float]):
    """Accuracy on the top-τ-fraction trials by max-confidence."""
    confidences = softmax_probs.max(axis=1)
    predictions = softmax_probs.argmax(axis=1)
    correct = (predictions == gt_idx).astype(float)
    out = {}
    for tau in taus:
        # τ here = confidence threshold, NOT fraction. Trials above threshold:
        mask = confidences >= tau
        if mask.sum() == 0:
            out[f"sel@τ≥{tau}"] = (None, 0)
        else:
            out[f"sel@τ≥{tau}"] = (float(correct[mask].mean()), int(mask.sum()))
    return out


def beta_reliability(per_trial_betas: np.ndarray, trial_ids: np.ndarray) -> float:
    """Pearson r between betas of repeated presentations of same image,
    averaged across images. Higher = more reliable single-trial signal."""
    by_img: dict[str, list[np.ndarray]] = {}
    for i, t in enumerate(trial_ids):
        ts = str(t)
        if ts.startswith("all_stimuli/special515/"):
            by_img.setdefault(ts, []).append(per_trial_betas[i])
    rs = []
    for img, reps in by_img.items():
        if len(reps) < 2: continue
        for i in range(len(reps)):
            for j in range(i+1, len(reps)):
                a, b = reps[i] - reps[i].mean(), reps[j] - reps[j].mean()
                r = (a * b).sum() / (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()) + 1e-12)
                rs.append(float(r))
    return float(np.mean(rs)) if rs else 0.0


# -------- Forward pass + scoring per cell -----------------------------------

def filter_first_rep(arr: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen, keep = set(), []
    for i, t in enumerate(ids):
        ts = str(t)
        if not ts.startswith("all_stimuli/special515/"): continue
        if ts in seen: continue
        seen.add(ts); keep.append(i)
    return arr[keep], np.asarray([str(ids[i]) for i in keep])


def forward(model, ss, se, betas):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            cv = model.backbone(voxel_ridge)
            cv = cv[1] if isinstance(cv, tuple) else cv
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, 0).reshape(-1, ss, se)


def score_cell(name: str, model, ss, se):
    """Return all metrics dict; or {'error': ...} if cell missing."""
    p = PREREG / f"{name}_ses-03_betas.npy"
    if not p.exists():
        return {"error": "missing"}
    raw_betas = np.load(p)
    raw_ids = np.load(PREREG / f"{name}_ses-03_trial_ids.npy")

    # First-rep filtered for retrieval-style metrics
    test_betas, test_ids = filter_first_rep(raw_betas, raw_ids)
    if len(test_ids) == 0:
        return {"error": "no special515 trials"}
    image_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)
    pred = forward(model, ss, se, test_betas)
    sim = M.cosine_sim_tokens(pred, gt)  # (n, n)
    n = sim.shape[0]
    gt_idx = np.arange(n)

    # Softmax over candidates for calibration metrics
    sim_t = sim - sim.max(axis=1, keepdims=True)
    probs = np.exp(sim_t * 30.0)  # temperature; 30 ~ typical for cosine sim
    probs = probs / probs.sum(axis=1, keepdims=True)

    metrics = {
        "n_test": n,
        "top1": top_k_acc(sim, gt_idx, 1),
        "top5": top_k_acc(sim, gt_idx, 5),
        "brain_retrieval_top1": brain_retrieval_acc(sim, 1),
        "two_afc": two_afc_acc(sim),
    }
    auc, cohend = merge_separate(sim)
    metrics["merge_separate_auc"] = auc
    metrics["cohens_d"] = cohend
    metrics["brier"] = brier(probs, gt_idx)
    metrics["ece"] = ece(probs, gt_idx, n_bins=10)
    sel = selective_acc(probs, gt_idx, [0.05, 0.1, 0.2, 0.5])
    for k, v in sel.items():
        metrics[k] = v
    metrics["beta_reliability_r"] = beta_reliability(raw_betas, raw_ids)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", choices=list(CKPTS), default="fold10")
    ap.add_argument("--cells", default=None,
                     help="comma-separated; default = all RT_paper_*_inclz cells")
    ap.add_argument("--out", default=str(LOCAL / "task_2_1_betas/unified_metrics.json"))
    args = ap.parse_args()

    print(f"loading checkpoint: {args.ckpt}")
    model, ss, se = M.load_mindeye(CKPTS[args.ckpt], n_voxels=2792, device=device)

    if args.cells:
        cells = args.cells.split(",")
    else:
        cells = sorted({p.stem.replace("_ses-03_betas", "") for p in PREREG.glob("RT_paper_*_inclz_ses-03_betas.npy")})
        cells += [
            "RT_paper_EoR_K10_CSFWM_inclz",
            "RT_paper_EoR_OLS_glover_inclz",
            "RT_paper_EoR_OLS_hrflib_inclz",
            "RT_paper_EoR_fmriprep_inclz",
            "RTmotion_GLMsingle_singleRep",
        ]

    print(f"\n{'cell':45s}  {'top1':>6s}  {'top5':>6s}  {'brain':>6s}  {'2AFC':>6s}  {'AUC':>6s}  {'d':>5s}  {'Brier':>6s}  {'ECE':>5s}  {'β-rel':>6s}")
    print("-" * 130)
    out = {"ckpt": args.ckpt, "cells": {}}
    for cell in cells:
        m = score_cell(cell, model, ss, se)
        if "error" in m:
            print(f"{cell:45s}  {m['error']}")
            continue
        out["cells"][cell] = m
        print(f"{cell:45s}  "
              f"{m['top1']*100:5.1f}%  {m['top5']*100:5.1f}%  "
              f"{m['brain_retrieval_top1']*100:5.1f}%  {m['two_afc']*100:5.1f}%  "
              f"{m['merge_separate_auc']:5.3f}  {m['cohens_d']:4.2f}  "
              f"{m['brier']:5.3f}  {m['ece']:4.2f}  "
              f"{m['beta_reliability_r']:5.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
