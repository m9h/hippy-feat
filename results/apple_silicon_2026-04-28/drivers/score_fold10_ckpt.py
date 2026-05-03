#!/usr/bin/env python3
"""Re-score key cells against the paper-confirmed fold-10 checkpoint
(per Rishab's Discord clarification — DGX commit 1445966 / d38a0d0).

DGX rescored numbers we're aiming to reproduce:
  Offline 3T avg-3-rep (paper 90%): 88% on fold-10 (vs 74% on fold-0)
  EoR RT first-rep    (paper 66%): 64% on fold-10 (vs 58% on fold-0)
  Slow RT first-rep   (paper 58%): 66% on fold-10 (vs 58% on fold-0)
  Offline 3T 1st-rep  (paper 76%): 62% on fold-10 (vs 56% on fold-0)

We also score our Round 4 rtmotion+GLMsingle output and our RT_paper_*_inclz cells.
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

CKPT_FOLD10 = LOCAL / "rt3t" / "data" / "model" / "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150" / "last.pth"
PREREG = LOCAL / "task_2_1_betas" / "prereg"
CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
RESULTS_PATH = LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"
device = "mps" if torch.backends.mps.is_available() else "cpu"

CANONICAL_NPZ = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C" / "TYPED_FITHRF_GLMDENOISE_RR.npz"
CANONICAL_BRAIN = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C" / "sub-005_ses-03_task-C_brain.nii.gz"
RTMOTION_NPY = LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C_RTMOTION" / "TYPED_FITHRF_GLMDENOISE_RR.npy"


def load_canonical_to_relmask():
    z = np.load(CANONICAL_NPZ, allow_pickle=True)
    betas_full = z["betasmd"].squeeze().astype(np.float32)
    canon_brain = nib.load(CANONICAL_BRAIN).get_fdata() > 0
    final_mask = nib.load(LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy")
    me_positions = np.where(final_mask.flatten())[0][relmask]
    canon_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_brain_idx[me_positions]
    betas_me = betas_full[me_in_canon, :].astype(np.float32)
    ids = []
    for run in range(1, 12):
        e = pd.read_csv(LOCAL / "rt3t" / "data" / "events" / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv",
                         sep="\t")
        e = e[e["image_name"].astype(str) != "blank.jpg"]
        ids += e["image_name"].astype(str).tolist()
    return betas_me.T.astype(np.float32), np.asarray(ids)


def load_rtmotion_to_relmask():
    arr = np.load(RTMOTION_NPY, allow_pickle=True)
    gs = arr.item() if arr.dtype == object and arr.ndim == 0 else arr
    betasmd = gs["betasmd"]
    flat = np.nan_to_num(betasmd.reshape(-1, betasmd.shape[-1]), nan=0.0)
    final = (nib.load(LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz").get_fdata() > 0).flatten()
    rel = np.load(LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy")
    betas_2792 = flat[final][rel].T.astype(np.float32)
    ids = np.load(LOCAL / "glmsingle" / "glmsingle_sub-005_ses-03_task-C_RTMOTION" / "trial_ids_chronological.npy", allow_pickle=True)
    return betas_2792, np.asarray(ids)


def session_train_z(arr, ids):
    is_train = np.array([not str(t).startswith("all_stimuli/special515/") for t in ids])
    train = arr[is_train]
    mu = train.mean(0, keepdims=True); sd = train.std(0, keepdims=True) + 1e-6
    return ((arr - mu) / sd).astype(np.float32)


def first_rep_filter(arr, ids):
    seen, keep = set(), []
    for i, t in enumerate(ids):
        ts = str(t)
        if not ts.startswith("all_stimuli/special515/"): continue
        if ts in seen: continue
        seen.add(ts); keep.append(i)
    return arr[keep], np.asarray([str(ids[i]) for i in keep])


def repeat_avg(arr, ids):
    is_spec = np.array([str(t).startswith("all_stimuli/special515/") for t in ids])
    sa, si = arr[is_spec], ids[is_spec]
    unique = sorted(set(str(t) for t in si))
    out = []
    for img in unique:
        idxs = [i for i, t in enumerate(si) if str(t) == img]
        out.append(sa[idxs].mean(0))
    return np.stack(out, 0).astype(np.float32), np.asarray(unique)


print(f"loading fold-10 checkpoint: {CKPT_FOLD10}")
model, ss, se = M.load_mindeye(Path(CKPT_FOLD10), n_voxels=2792, device=device)


def score(betas, ids, label):
    paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(str(n)).name for n in ids]
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


# Score canonical fmriprep+GLMsingle (Offline anchor) and rtmotion+GLMsingle (Round 4)
print(f"\n{'cell':50s}  {'top1':>7s}  {'top5':>7s}  paper_anchor  Δ")
print("-" * 95)

results = []

for source, loader in [
    ("canonical_fmriprep_glmsingle", load_canonical_to_relmask),
    ("rtmotion_glmsingle",           load_rtmotion_to_relmask),
]:
    raw, ids = loader()
    z = session_train_z(raw, ids)  # paper §2.5.1 policy
    for mode_name, mode_fn, anchor in [("first-rep", first_rep_filter, 0.76),
                                         ("rep-avg", repeat_avg, 0.90)]:
        b, kept = mode_fn(z, ids)
        t1, t5 = score(b, kept, f"{source}/{mode_name}")
        label = f"fold10_{source}_{mode_name.replace('-','_')}"
        print(f"{label:50s}  {t1*100:6.2f}%  {t5*100:6.2f}%  paper {anchor*100:.0f}%   "
              f"Δ={(t1-anchor)*100:+.1f}pp")
        results.append({"cell": label, "top1": t1, "top5": t5,
                         "paper_anchor": anchor, "delta_pp": (t1-anchor)*100})

# Score our existing RT_paper_*_inclz cells with fold-10
RT_CELLS = [
    ("RT_paper_Fast_pst5_inclz",         0.36),
    ("RT_paper_Slow_pst20_inclz",        0.58),
    ("RT_paper_Slow_pst25_inclz",        0.58),
    ("RT_paper_EndOfRun_pst_None_inclz", 0.66),
]
for cell, anchor in RT_CELLS:
    if not (PREREG / f"{cell}_ses-03_betas.npy").exists():
        print(f"{cell:50s}  SKIP")
        continue
    raw_betas = np.load(PREREG / f"{cell}_ses-03_betas.npy")
    raw_ids = np.load(PREREG / f"{cell}_ses-03_trial_ids.npy")
    b, kept = first_rep_filter(raw_betas, raw_ids)
    t1, t5 = score(b, kept, cell)
    print(f"fold10_{cell:43s}  {t1*100:6.2f}%  {t5*100:6.2f}%  paper {anchor*100:.0f}%   "
          f"Δ={(t1-anchor)*100:+.1f}pp")
    results.append({"cell": f"fold10_{cell}", "top1": t1, "top5": t5,
                     "paper_anchor": anchor, "delta_pp": (t1-anchor)*100})

with open(LOCAL / "task_2_1_betas" / "fold10_rescore.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {LOCAL / 'task_2_1_betas' / 'fold10_rescore.json'}")
