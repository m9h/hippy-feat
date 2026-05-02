#!/usr/bin/env python3
"""Score canonical and rtmotion GLMsingle outputs under all three
z-score policies × first-rep/rep-avg, to identify which matches the
paper Offline 3T anchors (76% single-rep top-1, 90% avg-3-reps top-1).

Per paper main.tex line 212 (§2.5.1 Offline Preprocessing):
  "Betas were z-scored voxelwise using the training images from the
   entire session."

So the paper-correct z policy for the Offline anchor is:
  - μ, σ computed over TRAINING-image betas only (non-special515 trials)
  - Applied to all betas
  - Non-causal in session position, but no test-set leakage

Three z policies tested:
  session_all     : μ, σ over ALL 693 non-blank trials (current Mac scorer; LEAKY)
  session_train   : μ, σ over training-image trials only (~543; PAPER POLICY)
  causal_cumz     : μ, σ over arr[:i] past-only (DGX scorer; conservative)

Two retrieval modes:
  first-rep       : keep first occurrence of each special515 image (50 trials)
  rep-avg         : average across 3 reps per image (50 averaged betas)
"""
from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")

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

CKPT = LOCAL / "rt3t" / "data" / "model" / "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
RESULTS_PATH = LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"
device = "mps" if torch.backends.mps.is_available() else "cpu"

CANONICAL_NPZ = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C" / "TYPED_FITHRF_GLMDENOISE_RR.npz"
CANONICAL_BRAIN = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C" / "sub-005_ses-03_task-C_brain.nii.gz"
RTMOTION_NPY = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C_RTMOTION" / "TYPED_FITHRF_GLMDENOISE_RR.npy"


def load_canonical_to_relmask() -> tuple[np.ndarray, np.ndarray]:
    """Returns (T, 2792) raw betas + chronological trial_ids matching betasmd order."""
    z = np.load(CANONICAL_NPZ, allow_pickle=True)
    betas_full = z["betasmd"].squeeze().astype(np.float32)
    canon_brain = nib.load(CANONICAL_BRAIN).get_fdata() > 0
    final_mask = nib.load(LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy")
    me_positions = np.where(final_mask.flatten())[0][relmask]
    canon_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_brain_idx[me_positions]
    betas_me = betas_full[me_in_canon, :].astype(np.float32)             # (2792, T)
    ids = []
    for run in range(1, 12):
        e = pd.read_csv(LOCAL / "rt3t" / "data" / "events" / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv",
                         sep="\t")
        e = e[e["image_name"].astype(str) != "blank.jpg"]
        ids += e["image_name"].astype(str).tolist()
    return betas_me.T.astype(np.float32), np.asarray(ids)                # (T=693, 2792), (T,)


def load_rtmotion_to_relmask() -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(RTMOTION_NPY, allow_pickle=True)
    gs = arr.item() if arr.dtype == object and arr.ndim == 0 else arr
    betasmd = gs["betasmd"]                                               # (X,Y,Z,T)
    flat = betasmd.reshape(-1, betasmd.shape[-1])
    flat = np.nan_to_num(flat, nan=0.0)
    final = (nib.load(LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz").get_fdata() > 0).flatten()
    rel = np.load(LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy")
    betas_2792 = flat[final][rel].T.astype(np.float32)                    # (T=693, 2792)
    ids_path = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C_RTMOTION" / "trial_ids_chronological.npy"
    ids = np.load(ids_path, allow_pickle=True)
    return betas_2792, np.asarray(ids)


# ----- z-score policies -------------------------------------------------------

def z_session_all(arr: np.ndarray, ids: np.ndarray) -> np.ndarray:
    mu = arr.mean(0, keepdims=True); sd = arr.std(0, keepdims=True) + 1e-6
    return ((arr - mu) / sd).astype(np.float32)


def z_session_train(arr: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Paper §2.5.1: μ, σ over training images (non-special515) only."""
    is_train = np.array([not str(t).startswith("all_stimuli/special515/") for t in ids])
    train = arr[is_train]
    mu = train.mean(0, keepdims=True); sd = train.std(0, keepdims=True) + 1e-6
    return ((arr - mu) / sd).astype(np.float32)


def z_causal_cumz(arr: np.ndarray, ids: np.ndarray) -> np.ndarray:
    n, V = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        if i == 0:
            out[i] = arr[i]
        elif i == 1:
            out[i] = arr[i] - arr[0]
        else:
            mu = arr[:i].mean(0); sd = arr[:i].std(0) + 1e-6
            out[i] = (arr[i] - mu) / sd
    return out


Z_POLICIES = {
    "session_all":   z_session_all,
    "session_train": z_session_train,
    "causal_cumz":   z_causal_cumz,
}


def filter_first_rep(arr: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen, keep = set(), []
    for i, t in enumerate(ids):
        ts = str(t)
        if not ts.startswith("all_stimuli/special515/"): continue
        if ts in seen: continue
        seen.add(ts); keep.append(i)
    return arr[keep], np.asarray([str(ids[i]) for i in keep])


def repeat_avg(arr: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    is_spec = np.array([str(t).startswith("all_stimuli/special515/") for t in ids])
    sa, si = arr[is_spec], ids[is_spec]
    unique = sorted(set(str(t) for t in si))
    out = []
    for img in unique:
        idxs = [i for i, t in enumerate(si) if str(t) == img]
        out.append(sa[idxs].mean(0))
    return np.stack(out, 0).astype(np.float32), np.asarray(unique)


# ----- score model ------------------------------------------------------------

model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)


def score(betas, ids, label):
    paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(str(n)).name
             for n in ids]
    gt = compute_gt_mps(paths, device=device, cache_dir=CACHE)
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            cv = model.backbone(voxel_ridge)
            cv = cv[1] if isinstance(cv, tuple) else cv
            out.append(cv.float().cpu().numpy())
    pred = np.concatenate(out, 0).reshape(-1, ss, se)
    img_to_idx = {str(im): i for i, im in enumerate(ids)}
    trial_idx = np.array([img_to_idx[str(t)] for t in ids])
    sim = M.cosine_sim_tokens(pred, gt)
    t1 = float(M.top_k_retrieval(sim, trial_idx, k=1))
    t5 = float(M.top_k_retrieval(sim, trial_idx, k=5))
    return t1, t5


# ----- run the matrix ---------------------------------------------------------

print("="*100)
print(f"{'BOLD source':16s}  {'z policy':16s}  {'mode':10s}  {'top1':>7s}  {'top5':>7s}  paper anchor")
print("="*100)

results_summary = []

for source_name, loader, anchors in [
    ("canonical(fmriprep)", load_canonical_to_relmask,
        {"first-rep": (0.76, "Offline 3T 1st-rep"), "repeat-avg": (0.90, "Offline 3T avg-3-rep")}),
    ("rtmotion",            load_rtmotion_to_relmask,
        {"first-rep": (0.76, "Offline 3T 1st-rep (cf paper)"), "repeat-avg": (0.90, "Offline 3T avg-3-rep (cf paper)")}),
]:
    raw, ids = loader()
    print(f"\n--- {source_name}  raw shape: {raw.shape} ---")
    for z_name, z_fn in Z_POLICIES.items():
        zarr = z_fn(raw, ids)
        for mode_name, mode_fn in [("first-rep", filter_first_rep), ("repeat-avg", repeat_avg)]:
            betas, kept_ids = mode_fn(zarr, ids)
            anchor, anchor_label = anchors[mode_name]
            t1, t5 = score(betas, kept_ids, f"{source_name}/{z_name}/{mode_name}")
            print(f"{source_name:16s}  {z_name:16s}  {mode_name:10s}  "
                  f"{t1*100:6.2f}%  {t5*100:6.2f}%  "
                  f"(paper {anchor_label} {anchor*100:.0f}%, Δ={(t1-anchor)*100:+.1f}pp)")
            results_summary.append({
                "bold": source_name, "z_policy": z_name, "mode": mode_name,
                "top1": t1, "top5": t5,
                "paper_anchor": anchor, "anchor_label": anchor_label,
                "delta_pp": (t1 - anchor) * 100,
            })

# Save matrix
print("\n=== summary ===")
out_path = LOCAL / "task_2_1_betas" / "z_policy_matrix.json"
with open(out_path, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"saved {out_path.name}")
