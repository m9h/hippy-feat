#!/usr/bin/env python3
"""Score retrieval on Princeton's CANONICAL GLMsingle TYPED_FITHRF_GLMDENOISE_RR
betas (sub-005 ses-03), using the same MindEye finalmask checkpoint we've
been using. Tells us if the 76%/77% Offline anchor reproduces on the actual
canonical betas (not our reverse-engineered Glover+AR1 cell 12).
"""
from __future__ import annotations

import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import pandas as pd

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")

# stubs (Decoder + sgm)
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

warnings.filterwarnings("ignore")

# ---- load canonical GLMsingle output ----------------------------------------

GLM_DIR = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C"
NPZ = GLM_DIR / "TYPED_FITHRF_GLMDENOISE_RR.npz"
CANONICAL_BRAIN = GLM_DIR / "sub-005_ses-03_task-C_brain.nii.gz"

print(f"loading canonical GLMsingle output: {NPZ}")
gs = np.load(NPZ, allow_pickle=True)
betas_flat = gs["betasmd"].reshape(-1, gs["betasmd"].shape[-1])    # (183408, 693)
print(f"  betas: {betas_flat.shape}, nan count: {np.isnan(betas_flat).sum()}")

# Reshape to 3D
canon_brain_img = nib.load(CANONICAL_BRAIN)
canon_brain_mask_3d = canon_brain_img.get_fdata() > 0               # (76,90,74) bool
canon_brain_flat = canon_brain_mask_3d.flatten()                    # (506160,)
n_canon_brain = canon_brain_flat.sum()
assert n_canon_brain == betas_flat.shape[0], \
    f"canonical brain mask {n_canon_brain} != betas first dim {betas_flat.shape[0]}"

# Build full 3D beta volume: (76,90,74, 693), zero outside canonical brain
n_trials = betas_flat.shape[1]
betas_4d_flat = np.zeros((506160, n_trials), dtype=np.float32)
betas_4d_flat[canon_brain_flat] = np.nan_to_num(betas_flat, nan=0.0)

# Apply OUR finalmask + relmask
our_brain = nib.load(LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz")
our_brain_flat = (our_brain.get_fdata() > 0).flatten()              # (506160,) 19174 True
rel = np.load(LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy")  # (19174,) 2792 True

# Project canonical betas through our masks: full -> our finalmask -> relmask
betas_finalmask = betas_4d_flat[our_brain_flat]                     # (19174, 693)
betas_2792 = betas_finalmask[rel]                                    # (2792, 693)
betas_per_trial = betas_2792.T                                       # (693, 2792)
print(f"  canonical betas projected to (n_trials, 2792 voxels): {betas_per_trial.shape}")

# ---- match trials to events (drop blanks) ------------------------------------

events_dir = LOCAL / "rt3t" / "data" / "events"
all_image_names = []
for run in range(1, 12):
    df = pd.read_csv(events_dir / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    df = df[df["image_name"].astype(str) != "blank.jpg"]            # match canonical 693
    all_image_names.extend(df["image_name"].astype(str).tolist())

assert len(all_image_names) == 693, f"got {len(all_image_names)} images, expected 693"
trial_ids = np.asarray(all_image_names)
print(f"  matched {len(trial_ids)} non-blank trials to canonical {betas_per_trial.shape[0]}")

# ---- apply paper-style cumulative z-score + repeat-averaging ----------------

# Per mindeye.py:770-784 — the paper's actual formula
print("\n=== apply paper cum-z + repeat-avg ===")
arr = betas_per_trial.astype(np.float32)
n, V = arr.shape

# Group by image, find repeats
img_groups: dict[str, list[int]] = {}
for i, t in enumerate(trial_ids):
    img_groups.setdefault(t, []).append(i)
print(f"  unique images: {len(img_groups)}")

# Paper's formula: re-z-score using LATEST stats (z_mean/z_std over all betas)
# Then for each image's repeats, average z'd values
z_mean = arr.mean(axis=0, keepdims=True)
z_std = arr.std(axis=0, keepdims=True) + 1e-6
z = (arr - z_mean) / z_std

# Filter to special515 trials
spec_mask = np.array([n.startswith("all_stimuli/special515/") for n in trial_ids])
spec_z = z[spec_mask]
spec_ids = trial_ids[spec_mask]

# Repeat-average per image
unique_images = sorted(set(spec_ids))
post_betas = []
for img in unique_images:
    idxs = [i for i, t in enumerate(spec_ids) if t == img]
    post_betas.append(spec_z[idxs].mean(axis=0))
post_betas = np.stack(post_betas, axis=0).astype(np.float32)
post_ids = np.asarray(unique_images)
print(f"  post repeat-avg: {post_betas.shape}")

# ---- retrieval scoring -------------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n=== retrieval scoring on canonical Offline βs (device={device}) ===")

cache_dir = LOCAL / "task_2_1_betas" / "gt_cache"
image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
               for n in unique_images]
gt_emb = compute_gt_mps(image_paths, device=device, cache_dir=cache_dir)
print(f"  gt: {gt_emb.shape}")

CKPT = LOCAL / "rt3t" / "data" / "model" / "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)

import time as _t
t0 = _t.time()
with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
    b = torch.from_numpy(post_betas.astype(np.float32)).to(device).unsqueeze(1)
    out = []
    for i in range(b.shape[0]):
        voxel_ridge = model.ridge(b[i:i+1], 0)
        backbone_out = model.backbone(voxel_ridge)
        clip_voxels = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
        out.append(clip_voxels.float().cpu().numpy())
pred_emb = np.concatenate(out, axis=0).reshape(-1, ss, se)
print(f"  pred: {pred_emb.shape}  ({_t.time()-t0:.1f}s)")

img_to_idx = {img: i for i, img in enumerate(unique_images)}
trial_idx = np.array([img_to_idx[t] for t in post_ids])
sim = M.cosine_sim_tokens(pred_emb, gt_emb)
top1 = M.top_k_retrieval(sim, trial_idx, k=1)
top5 = M.top_k_retrieval(sim, trial_idx, k=5)

# Pairwise AUC
Bn = post_betas / (np.linalg.norm(post_betas, axis=1, keepdims=True) + 1e-8)
D = 1.0 - Bn @ Bn.T
# Without repeat-avg, AUC requires same-image pairs; after repeat-avg there's only 1 trial per image
# So we report only retrieval for the post-repeat-avg cell

print(f"\n=== CANONICAL OFFLINE RETRIEVAL RESULT ===")
print(f"  betas: canonical GLMsingle TYPED_FITHRF_GLMDENOISE_RR (sub-005 ses-03)")
print(f"  test trials: {post_betas.shape[0]} unique special515 images (post repeat-avg)")
print(f"  TOP-1: {top1*100:.2f}%   TOP-5: {top5*100:.2f}%")

# Also save raw canonical betas (no z, no avg) for further analysis
np.save(LOCAL / "task_2_1_betas" / "prereg" / "Canonical_GLMsingle_OfflineFull_ses-03_betas.npy", post_betas)
np.save(LOCAL / "task_2_1_betas" / "prereg" / "Canonical_GLMsingle_OfflineFull_ses-03_trial_ids.npy", post_ids)

# Save result to retrieval JSON
import json
out_path = LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"
results = json.loads(out_path.read_text())
results.append({
    "condition": "Canonical_GLMsingle_OfflineFull",
    "session": "ses-03",
    "n_test_trials": int(post_betas.shape[0]),
    "n_unique_images": int(len(unique_images)),
    "top1_image_retrieval": float(top1),
    "top5_image_retrieval": float(top5),
    "notes": "Princeton canonical TYPED_FITHRF_GLMDENOISE_RR betas + paper cum-z + repeat-avg",
})
out_path.write_text(json.dumps(results, indent=2))
print(f"\n  saved to {out_path}")
