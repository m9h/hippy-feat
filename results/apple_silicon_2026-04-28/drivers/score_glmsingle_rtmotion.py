#!/usr/bin/env python3
"""Score the rtmotion-GLMsingle output two ways:

1. Single-rep filter (apples-to-apples vs EoR baseline 56% + paper EoR 66%):
   keep first occurrence of each special515 image, apply paper full-session
   cum-z, score retrieval.

2. Repeat-avg (apples-to-apples vs Offline anchor 76%): same paper cum-z,
   then average across the 3 reps of each special515 image, score.

Both modes use the new GLMsingle TYPED_FITHRF_GLMDENOISE_RR.npz fit on
rtmotion BOLD instead of fMRIPrep BOLD.
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

GLM_DIR = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C_RTMOTION"
NPZ = GLM_DIR / "TYPED_FITHRF_GLMDENOISE_RR.npy"  # GLMsingle modern format: pickled dict in .npy
PREREG = LOCAL / "task_2_1_betas" / "prereg"
RESULTS_PATH = LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"
CKPT = LOCAL / "rt3t" / "data" / "model" / "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"loading rtmotion-GLMsingle output: {NPZ}")
gs_arr = np.load(NPZ, allow_pickle=True)
gs = gs_arr.item() if gs_arr.dtype == object and gs_arr.ndim == 0 else gs_arr
beta_arr = gs["betasmd"]
print(f"  betasmd shape: {beta_arr.shape}, pcnum (GLMdenoise K): {gs.get('pcnum', 'n/a')}, "
      f"meanFRACvalue: {gs.get('FRACvalue').mean():.3f}" if "FRACvalue" in gs else "")

# Project to (n_trials, V) via our finalmask + relmask
if beta_arr.ndim == 4:
    X, Y, Z, T = beta_arr.shape
    flat = beta_arr.reshape(X * Y * Z, T)                    # (V_all, T)
else:
    flat = beta_arr.reshape(-1, beta_arr.shape[-1])
n_trials = flat.shape[1]
print(f"  flattened: {flat.shape}, n_trials={n_trials}, NaN count={np.isnan(flat).sum()}")
flat = np.nan_to_num(flat, nan=0.0)

# Apply our masks
our_brain = nib.load(LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz")
our_brain_flat = (our_brain.get_fdata() > 0).flatten()        # (506160,) 19174 True
rel = np.load(LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy")  # (19174,) 2792 True

if flat.shape[0] != our_brain_flat.shape[0]:
    raise SystemExit(f"voxel count mismatch: betas {flat.shape[0]} vs flat_brain {our_brain_flat.shape[0]}")

betas_brain = flat[our_brain_flat]                            # (19174, T)
betas_2792 = betas_brain[rel].T                                # (T, 2792)
print(f"  projected to (n_trials, 2792): {betas_2792.shape}")

# Trial IDs
trial_ids_path = GLM_DIR / "trial_ids_chronological.npy"
if trial_ids_path.exists():
    trial_ids = np.load(trial_ids_path, allow_pickle=True)
else:
    # rebuild from events.tsv
    print("  trial_ids_chronological.npy missing — rebuilding from events")
    events_dir = LOCAL / "rt3t" / "data" / "events"
    all_image_names = []
    for run in range(1, 12):
        df = pd.read_csv(events_dir / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
        df = df[df["image_name"].astype(str) != "blank.jpg"]
        all_image_names.extend(df["image_name"].astype(str).tolist())
    trial_ids = np.asarray(all_image_names)

assert len(trial_ids) == betas_2792.shape[0], \
    f"trial_ids {len(trial_ids)} != betas {betas_2792.shape[0]}"

# Paper cum-z (full-session — matches canonical scorer)
arr = betas_2792.astype(np.float32)
mu = arr.mean(axis=0, keepdims=True)
sd = arr.std(axis=0, keepdims=True) + 1e-6
z = (arr - mu) / sd

# Filter to special515
is_spec = np.array([str(t).startswith("all_stimuli/special515/") for t in trial_ids])
spec_z = z[is_spec]
spec_ids = trial_ids[is_spec]
print(f"  special515 trials: {spec_z.shape}, {len(set(spec_ids))} unique")

# Mode 1: SINGLE-REP filter
seen = set()
single_betas, single_ids = [], []
for i, t in enumerate(spec_ids):
    ts = str(t)
    if ts in seen:
        continue
    seen.add(ts)
    single_betas.append(spec_z[i])
    single_ids.append(ts)
single_betas = np.stack(single_betas, axis=0).astype(np.float32)
print(f"\n  single-rep: {single_betas.shape}")

# Mode 2: REPEAT-AVG
unique_images = sorted(set(spec_ids))
avg_betas = []
for img in unique_images:
    idxs = [i for i, t in enumerate(spec_ids) if str(t) == img]
    avg_betas.append(spec_z[idxs].mean(axis=0))
avg_betas = np.stack(avg_betas, axis=0).astype(np.float32)
avg_ids = list(unique_images)
print(f"  repeat-avg: {avg_betas.shape}")

# Score both
model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)


def fwd(test_betas):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            cv = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1, ss, se)


def score(test_betas, test_ids, label, paper_anchor, baseline, baseline_label):
    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)
    pred = fwd(test_betas)
    img_to_idx = {im: i for i, im in enumerate(test_ids)}
    trial_idx = np.array([img_to_idx[t] for t in test_ids])
    sim = M.cosine_sim_tokens(pred, gt)
    t1 = float(M.top_k_retrieval(sim, trial_idx, k=1))
    t5 = float(M.top_k_retrieval(sim, trial_idx, k=5))
    print(f"  {label:18s}  TOP-1: {t1*100:6.2f}%  TOP-5: {t5*100:6.2f}%  "
          f"(paper {paper_anchor*100:.0f}%, Δ={t1-paper_anchor:+.3f}; "
          f"vs {baseline_label} {baseline*100:.0f}%, Δ={t1-baseline:+.3f})")
    return t1, t5


print("\n=== rtmotion + GLMsingle (TYPED_FITHRF_GLMDENOISE_RR) ===")
t1_single, t5_single = score(single_betas, single_ids,
                              "single-rep", 0.66, 0.56, "EoR_baseline")
t1_avg, t5_avg = score(avg_betas, avg_ids,
                        "repeat-avg-3", 0.76, 0.56, "EoR_baseline")

# Save and persist
cell_single = "RTmotion_GLMsingle_singleRep"
cell_avg = "RTmotion_GLMsingle_repAvg"
np.save(PREREG / f"{cell_single}_ses-03_betas.npy", single_betas)
np.save(PREREG / f"{cell_single}_ses-03_trial_ids.npy", np.asarray(single_ids))
np.save(PREREG / f"{cell_avg}_ses-03_betas.npy", avg_betas)
np.save(PREREG / f"{cell_avg}_ses-03_trial_ids.npy", np.asarray(avg_ids))

results = json.loads(RESULTS_PATH.read_text())
results = [r for r in results if r.get("condition") not in (cell_single, cell_avg)]
results.append({
    "condition": cell_single, "session": "ses-03",
    "n_test_trials": int(single_betas.shape[0]), "n_unique_images": 50,
    "top1_image_retrieval": t1_single, "top5_image_retrieval": t5_single,
    "paper_anchor_top1": 0.66,
    "delta_vs_paper_pp": (t1_single - 0.66) * 100,
    "delta_vs_baseline_pp": (t1_single - 0.56) * 100,
    "notes": "rtmotion BOLD + actual GLMsingle TYPED_FITHRF_GLMDENOISE_RR + full-session z + first-rep filter",
})
results.append({
    "condition": cell_avg, "session": "ses-03",
    "n_test_trials": int(avg_betas.shape[0]), "n_unique_images": 50,
    "top1_image_retrieval": t1_avg, "top5_image_retrieval": t5_avg,
    "paper_anchor_top1": 0.76,
    "delta_vs_paper_pp": (t1_avg - 0.76) * 100,
    "notes": "rtmotion BOLD + actual GLMsingle TYPED_FITHRF_GLMDENOISE_RR + full-session z + 3-rep avg",
})
RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f"\nappended to {RESULTS_PATH.name}")
