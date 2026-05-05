#!/usr/bin/env python3
"""Reproduce paper Table 1 Offline 3T 76% by averaging across 3 repeats per
test image — the policy used in
`PrincetonCompMemLab/mindeye_offline:avg_betas/recon_inference-multisession.ipynb`.

Key ckpt-name signal: `..._avgrepeats_finalmask_epochs_150` — the test set is
50 special515 images, each with βs averaged across 3 ses-03 repeats. We had
been scoring on first-rep only, which loses √3 SNR.

Pipeline (matches recon_inference-multisession.ipynb):
  1. Load ses-03 GLMsingle (TYPED_FITHRF_GLMDENOISE_RR.npz) — per-session run.
  2. Project to finalmask (19174) → relmask (2792).
  3. utils.filter_and_average_repeats: collapse repeats per image to one β
     (50 special515 → 50 averaged βs; trained-on images keep 1 each).
  4. Train-only z-score (mean/std from non-test rows).
  5. Forward fold-10 ckpt → CLIP-brain → cos-sim top-1 vs CLIP-image.
"""
from __future__ import annotations

import json
import sys
import types
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
LOCAL_DRIVERS = Path("/Users/mhough/Workspace/local_drivers")

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
sys.path.insert(0, str(LOCAL_DRIVERS))

import mindeye_retrieval_eval as M
M.RTCLOUD_MINDEYE = RT_MINDEYE
from run_retrieval_local import compute_gt_mps

warnings.filterwarnings("ignore")

SES03_NPZ = LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz"
SES03_BRAIN = LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/sub-005_ses-03_task-C_brain.nii.gz"
FINAL_MASK = LOCAL / "rt3t/data/sub-005_final_mask.nii.gz"
RELMASK = LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy"
EVENTS_DIR = LOCAL / "rt3t/data/events"
CKPT = Path(__import__("os").environ.get(
    "CKPT",
    str(LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"),
))
CACHE = LOCAL / "task_2_1_betas/gt_cache"
device = "mps" if torch.backends.mps.is_available() else "cpu"


def filter_and_average_repeats(vox: np.ndarray, names: np.ndarray):
    """Verbatim port of utils.filter_and_average_repeats from mindeye_offline."""
    assert len(vox) == len(names)
    repeats = {}
    for i, im in enumerate(names):
        repeats.setdefault(im, []).append(i)
    keep = np.ones(len(vox), dtype=bool)
    out = deepcopy(vox).astype(np.float32)
    for idxs in repeats.values():
        if len(idxs) > 1:
            out[idxs[0]] = vox[idxs].mean(axis=0)
            keep[idxs[1:]] = False
    return out[keep], np.where(keep)[0]


print("=== loading ses-03 GLMsingle ===", flush=True)
gs = np.load(SES03_NPZ, allow_pickle=True)
betasmd = gs["betasmd"].squeeze().astype(np.float32)  # (V, 693)
print(f"  betasmd: {betasmd.shape}, pcnum={int(gs['pcnum'])}", flush=True)

brain_3d = nib.load(SES03_BRAIN).get_fdata() > 0
brain_flat = brain_3d.flatten()
n_brain = int(brain_flat.sum())
assert n_brain == betasmd.shape[0], f"brain {n_brain} != betas {betasmd.shape[0]}"

# (V_brain, 693) → (506160, 693) full-vol → finalmask (19174) → relmask (2792)
betas_full = np.zeros((506160, betasmd.shape[1]), dtype=np.float32)
betas_full[brain_flat] = np.nan_to_num(betasmd, nan=0.0)
final_mask_flat = nib.load(FINAL_MASK).get_fdata().flatten() > 0
relmask = np.load(RELMASK)
betas_2792 = betas_full[final_mask_flat][relmask]                    # (2792, 693)
vox = betas_2792.T                                                    # (693, 2792)
print(f"  vox after finalmask+relmask: {vox.shape}", flush=True)

# Build vox_image_names for ses-03 (693 trials, in order)
names = []
for run in range(1, 12):
    ev = pd.read_csv(EVENTS_DIR / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    ev = ev[ev["image_name"].astype(str).ne("blank.jpg")
            & ev["image_name"].astype(str).ne("nan")
            & ev["image_name"].notna()]
    names.extend(ev["image_name"].astype(str).tolist())
names = np.asarray(names)
print(f"  vox_image_names: {len(names)}  (expect 693)", flush=True)
assert len(names) == vox.shape[0], f"names {len(names)} != vox rows {vox.shape[0]}"

# === Apply filter_and_average_repeats ===
vox_avg, kept = filter_and_average_repeats(vox, names)
names_avg = names[kept]
print(f"  after avg-repeats: vox {vox_avg.shape}, unique imgs {len(names_avg)}", flush=True)

# Identify test = 50 special515 with 3 repeats
from collections import Counter
counts = Counter(names)
test_mask = np.array([counts[n] == 3 and "special515" in n for n in names_avg])
train_mask = ~test_mask
n_test, n_train = int(test_mask.sum()), int(train_mask.sum())
print(f"  test (3-rep special515): {n_test}, train: {n_train}", flush=True)
assert n_test == 50, f"expected 50 test images, got {n_test}"

# === Train-only z-score ===
train_mean = vox_avg[train_mask].mean(axis=0)
train_std = vox_avg[train_mask].std(axis=0) + 1e-6
vox_z = (vox_avg - train_mean) / train_std
print(f"  z-scored (train-only stats): mean of vox_z[:,0] = {vox_z[:,0].mean():.4f}", flush=True)

test_betas = vox_z[test_mask]            # (50, 2792)
test_names = names_avg[test_mask]        # (50,)

# === Forward through fold-10 ckpt ===
print(f"\n=== loading ckpt: {CKPT.name} ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)

print("=== forward pass ===", flush=True)
clipvoxels_list = []
with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
    b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
    for i in range(0, b.shape[0], 8):
        vr = model.ridge(b[i:i+8], 0)
        out = model.backbone(vr)
        cv = out[1] if isinstance(out, tuple) else out
        clipvoxels_list.append(cv.float().cpu().numpy())
pred = np.concatenate(clipvoxels_list, 0).reshape(-1, ss, se)   # (50, 256, 1664)
print(f"  pred clipvoxels: {pred.shape}", flush=True)

# === Compute CLIP-image embeddings (ground truth) ===
print("=== computing CLIP-image embeddings ===", flush=True)
img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names]
gt = compute_gt_mps(img_paths, device=device, cache_dir=CACHE)         # (50, 256, 1664)
print(f"  gt: {gt.shape}", flush=True)

# === Cosine top-1 (matches eval cell 16: flatten + L2 + matmul) ===
def topk_acc(p, g, k=1):
    p_flat = p.reshape(p.shape[0], -1)
    g_flat = g.reshape(g.shape[0], -1)
    p_norm = p_flat / (np.linalg.norm(p_flat, axis=1, keepdims=True) + 1e-8)
    g_norm = g_flat / (np.linalg.norm(g_flat, axis=1, keepdims=True) + 1e-8)
    sim = p_norm @ g_norm.T
    labels = np.arange(p.shape[0])
    topk_idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in topk_idx[i] for i, lbl in enumerate(labels)]))

t1_fwd = topk_acc(pred, gt, k=1)        # brain → image (matches paper "fwd")
t1_bwd = topk_acc(gt, pred, k=1)        # image → brain
t5_fwd = topk_acc(pred, gt, k=5)

# 2-AFC for completeness
sim = (pred.reshape(50, -1) / (np.linalg.norm(pred.reshape(50, -1), axis=1, keepdims=True) + 1e-8)) @ \
      (gt.reshape(50, -1) / (np.linalg.norm(gt.reshape(50, -1), axis=1, keepdims=True) + 1e-8)).T
correct, total = 0, 0
for i in range(50):
    for j in range(50):
        if i == j: continue
        if sim[i, i] > sim[i, j]: correct += 1
        total += 1
two_afc = correct / total

print("\n========== RESULTS ==========")
print(f"  TOP-1 fwd (brain→image): {t1_fwd*100:.2f}%   [paper target: 76%]")
print(f"  TOP-1 bwd (image→brain): {t1_bwd*100:.2f}%")
print(f"  TOP-5 fwd:               {t5_fwd*100:.2f}%")
print(f"  2-AFC:                   {two_afc*100:.2f}%")

result = {
    "method": "ses-03 GLMsingle, finalmask+relmask, filter_and_average_repeats, train-only z, fold-10",
    "n_test": 50,
    "top1_fwd": t1_fwd,
    "top1_bwd": t1_bwd,
    "top5_fwd": t5_fwd,
    "two_afc": two_afc,
    "paper_target_top1": 0.76,
}
out = LOCAL / "task_2_1_betas/avg_repeats_offline_score.json"
out.write_text(json.dumps(result, indent=2))
print(f"\n  saved {out.name}")
