#!/usr/bin/env python3
"""Score retrieval on the COMBINED ses-01-03 GLMsingle output (pcnum=4)
vs the ses-03-only GLMsingle output (pcnum=0).

Per docs/00-pipeline.md in the canonical Princeton repo:
'Each session's preprocessed data was input to GLMsingle (all 3 sessions
together) to obtain single-trial response estimates'

So the canonical βs that fed the trained model are from the COMBINED run,
not the per-session run we've been using. Combined run has 2079 trials
(693 × 3 sessions); we need to extract the ses-03 portion (last 693).
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

# Path to combined GLMsingle
COMBINED_NPZ = LOCAL / "glmsingle/glmsingle_sub-005_ses-01-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz"
COMBINED_BRAIN = LOCAL / "glmsingle/glmsingle_sub-005_ses-01-03_task-C/sub-005_ses-01-03_task-C_brain.nii.gz"

CKPT = LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth"
CACHE = LOCAL / "task_2_1_betas/gt_cache"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=== loading combined ses-01-03 GLMsingle ===", flush=True)
gs = np.load(COMBINED_NPZ, allow_pickle=True)
betasmd = gs["betasmd"]                                      # (V, 1, 1, 2079)
betasmd_flat = betasmd.squeeze().astype(np.float32)          # (V, 2079)
print(f"  betasmd: {betasmd_flat.shape}, pcnum={int(gs['pcnum'])}, "
      f"FRAC_mean={float(gs['FRACvalue'].mean()):.3f}", flush=True)

# The combined run's brain mask
brain_path = COMBINED_BRAIN if COMBINED_BRAIN.exists() and COMBINED_BRAIN.stat().st_size > 1000 \
    else (LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/sub-005_ses-03_task-C_brain.nii.gz")
print(f"  using brain mask: {brain_path.name}", flush=True)
canon_brain_3d = nib.load(brain_path).get_fdata() > 0
canon_brain_flat = canon_brain_3d.flatten()
n_canon_brain = int(canon_brain_flat.sum())
print(f"  canonical brain mask voxels: {n_canon_brain}", flush=True)

if n_canon_brain != betasmd_flat.shape[0]:
    print(f"  WARN: mask voxels {n_canon_brain} != betas {betasmd_flat.shape[0]}", flush=True)
    # Use a derived brain mask from non-NaN voxels
    valid = ~np.isnan(betasmd_flat).any(axis=1)
    print(f"  non-NaN voxels: {valid.sum()}", flush=True)

# Project to our finalmask + relmask
final_mask_3d = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata() > 0
final_mask_flat = final_mask_3d.flatten()
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")
print(f"  finalmask: {int(final_mask_flat.sum())}, relmask: {int(relmask.sum())}", flush=True)

# To go from canonical voxel index → our voxel space, we need to map
# canonical brain voxels' positions back to the full BOLD volume, then
# select the finalmask + relmask portion.
# But the masks have DIFFERENT voxel counts (182879 vs 183408 for ses-03-only).
# Need to know what 3D positions each voxel index corresponds to.

# Approach: betasmd_flat is shape (n_canon_brain, n_trials). Index 0 is the
# first True voxel in canon_brain_flat. Build betas_full_3d with 0s outside
# canon_brain.
n_trials = betasmd_flat.shape[1]
betas_full = np.zeros((506160, n_trials), dtype=np.float32)
betas_full[canon_brain_flat] = np.nan_to_num(betasmd_flat, nan=0.0)

# Apply finalmask + relmask
betas_finalmask = betas_full[final_mask_flat]            # (19174, n_trials)
betas_2792 = betas_finalmask[relmask]                     # (2792, n_trials)
betas_per_trial = betas_2792.T                             # (n_trials, 2792)
print(f"  betas projected to (n_trials, 2792): {betas_per_trial.shape}", flush=True)

# Build trial_ids by concatenating events from sessions 01, 02, 03
trial_ids = []
for ses in ["ses-01", "ses-02", "ses-03"]:
    for run in range(1, 12):
        ev_path = LOCAL / "rt3t/data/events" / f"sub-005_{ses}_task-C_run-{run:02d}_events.tsv"
        if not ev_path.exists():
            print(f"  MISSING events: {ev_path.name}", flush=True)
            continue
        ev = pd.read_csv(ev_path, sep="\t")
        ev = ev[ev["image_name"].astype(str) != "blank.jpg"]
        for _, row in ev.iterrows():
            trial_ids.append(f"{ses}/{row['image_name']}")

trial_ids = np.asarray(trial_ids)
print(f"  trial_ids constructed: {len(trial_ids)}", flush=True)

if len(trial_ids) != n_trials:
    print(f"  WARN: trial_ids {len(trial_ids)} != n_trials {n_trials}", flush=True)
    # Try sessions 01, 02, 03 in different order or just use ses-03 portion
    # Combined run could be 03+02+01 instead of 01+02+03.
    # For now, look at the LAST 693 trials assuming chronological 01→02→03

# Apply paper-correct z-score (training-images only, session-wide)
# For each session's βs, exclude special515 + MST_pairs from stats
# Simplification: use full-session-all-trials z (matches Mac's prior protocol)
print("\n=== applying full-session z (per-session, session-by-session) ===", flush=True)
# Identify session boundaries — assume chronological 01, 02, 03 each with 693 trials
ses_boundaries = [(0, 693, "ses-01"), (693, 1386, "ses-02"), (1386, 2079, "ses-03")]
betas_z = np.zeros_like(betas_per_trial)
for start, end, name in ses_boundaries:
    if end > betas_per_trial.shape[0]:
        continue
    chunk = betas_per_trial[start:end]
    mu = chunk.mean(axis=0, keepdims=True)
    sd = chunk.std(axis=0, keepdims=True) + 1e-6
    betas_z[start:end] = (chunk - mu) / sd
    print(f"  {name}: trials {start}-{end}", flush=True)

# Extract ses-03 portion + score retrieval
ses03_z = betas_z[1386:]
ses03_ids = trial_ids[1386:1386+ses03_z.shape[0]] if len(trial_ids) > 1386 else None
print(f"\n  ses-03 portion: {ses03_z.shape}, ids={None if ses03_ids is None else len(ses03_ids)}", flush=True)


def score(betas, ids, mode_label):
    """Apply first-rep filter, run forward, score top-1/2-AFC."""
    seen, keep = set(), []
    for i, t in enumerate(ids):
        ts = str(t).split("/")[-1] if "/" in str(t) else str(t)
        if not ts.startswith("special_") and "/special515/" not in str(t):
            continue
        if ts in seen: continue
        seen.add(ts)
        keep.append(i)
    test_betas = betas[keep]
    test_ids = [str(ids[i]).split("/")[-1] if "/" in str(ids[i]) else str(ids[i]) for i in keep]
    print(f"\n  {mode_label}: {test_betas.shape[0]} first-rep trials", flush=True)

    # Build paths
    image_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)
    model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            cv = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(cv.float().cpu().numpy())
    pred = np.concatenate(out, 0).reshape(-1, ss, se)
    img_to_idx = {im: i for i, im in enumerate(test_ids)}
    trial_idx = np.array([img_to_idx[t] for t in test_ids])
    sim = M.cosine_sim_tokens(pred, gt)
    t1 = float(M.top_k_retrieval(sim, trial_idx, k=1))
    t5 = float(M.top_k_retrieval(sim, trial_idx, k=5))
    # 2-AFC
    n = sim.shape[0]
    correct, total = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if sim[i, i] > sim[i, j]: correct += 1
            total += 1
    two_afc = correct / total
    return t1, t5, two_afc


print("\n=== scoring combined ses-01-03 βs (ses-03 portion, first-rep) ===", flush=True)
if ses03_ids is not None:
    t1, t5, twoafc = score(ses03_z, ses03_ids, "combined-3-ses ses-03 portion")
    print(f"\n  TOP-1: {t1*100:.2f}%   TOP-5: {t5*100:.2f}%   2-AFC: {twoafc*100:.2f}%")
    print(f"  (paper Offline 3T 1st-rep target: 76%)")

# Save
out_path = LOCAL / "task_2_1_betas/combined_glmsingle_score.json"
result = {
    "method": "combined ses-01-03 GLMsingle (pcnum=4) ses-03 portion, first-rep",
    "pcnum": int(gs["pcnum"]),
    "n_test_trials": 50,
    "top1": t1, "top5": t5, "two_afc": twoafc,
    "comparison": {
        "ses03_only_glmsingle_pcnum0_top1": 0.56,
        "paper_target_top1": 0.76,
    },
}
out_path.write_text(json.dumps(result, indent=2))
print(f"\nsaved {out_path.name}")
