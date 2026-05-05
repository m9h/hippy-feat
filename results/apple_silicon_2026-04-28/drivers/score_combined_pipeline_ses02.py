#!/usr/bin/env python3
"""Deploy combined pipeline on ses-02 (rtQA flagged 3 bad runs: 5, 8, 11).

Tests whether the deployment pipeline degrades on a session with mild
within-run signal-quality drops vs the cleaner ses-01.

Also breaks down retrieval performance by which run the test trials fall in,
to validate whether the rtQA flag (per-run tSNR < session mean − 5pp) actually
predicts retrieval failure.
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np, torch, torch.nn as nn, pandas as pd

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

CKPT = LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"
PREREG = LOCAL / "task_2_1_betas/prereg"
EVENTS = LOCAL / "rt3t/data/events"
SES = "ses-02"
TR = 1.5


class PerVoxelScalar(nn.Module):
    def __init__(self, n_vox=2792):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vox))
        self.bias = nn.Parameter(torch.zeros(n_vox))
    def forward(self, x):
        return x * self.gain + self.bias


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def fwd_eval(model, ss, se, x, batch=8):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = x.unsqueeze(1)
        for i in range(0, b.shape[0], batch):
            vr = model.ridge(b[i:i+batch], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float())
    return torch.cat(out, 0).reshape(-1, ss, se)


print("=== loading fold-0 ckpt ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)

refiner = PerVoxelScalar(n_vox=2792).to(device)
refiner.load_state_dict(torch.load(LOCAL/"task_2_1_betas/fast_refiner_state.pth", map_location=device, weights_only=True))
refiner.eval()


# --- Load ses-02 data + identify which run each trial is in ---
fast_b = np.load(PREREG / f"RT_paper_Fast_pst5_fmriprep_inclz_{SES}_betas.npy")
fast_ids = np.load(PREREG / f"RT_paper_Fast_pst5_fmriprep_inclz_{SES}_trial_ids.npy")
fast_ids = np.asarray([str(t) for t in fast_ids])
slow_b = np.load(PREREG / f"RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz_distill_fmriprep_{SES}_betas.npy")

# Build trial→run mapping
trial_runs = []
for run in range(1, 12):
    ev = pd.read_csv(EVENTS / f"sub-005_{SES}_task-C_run-{run:02d}_events.tsv", sep="\t")
    n_trials_run = len(ev)
    trial_runs.extend([run] * n_trials_run)
trial_runs = np.asarray(trial_runs)
assert len(trial_runs) == len(fast_ids), f"trial_runs {len(trial_runs)} != fast_ids {len(fast_ids)}"

# Identify 50 special515 with 3 reps + first-rep indices
counts = Counter(fast_ids)
test_imgs = {n for n, c in counts.items() if c == 3 and "special515" in n}
print(f"  test images: {len(test_imgs)}", flush=True)

# Group trials per image
by_img = defaultdict(list)
for i, n in enumerate(fast_ids):
    if n in test_imgs:
        by_img[n].append(i)


# Per-rep scoring: compute test set + each rep separately + flag bad runs
def score_per_rep(beta_arr, label):
    # Take first rep of each test image
    first_rep_idx = []
    for n, idxs in by_img.items():
        first_rep_idx.append(min(idxs))
    first_rep_idx = sorted(first_rep_idx)
    names = [fast_ids[i] for i in first_rep_idx]
    runs = [trial_runs[i] for i in first_rep_idx]
    paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in names]
    gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
    x = torch.from_numpy(beta_arr[first_rep_idx].astype(np.float32)).to(device)
    p = fwd_eval(model, ss, se, x, batch=8).cpu().numpy()
    img_acc = topk(p, gt, 1); bra_acc = topk(gt, p, 1)
    print(f"\n  {label}: subset0 (50 first-rep): Image={img_acc*100:5.1f}%  Brain={bra_acc*100:5.1f}%", flush=True)

    # Per-trial correctness for run breakdown
    pf = p.reshape(p.shape[0], -1); gf = np.asarray(gt).reshape(50, -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(50)
    correct = sim.argmax(axis=1) == labels
    bad_runs = {5, 8, 11}
    in_bad = np.array([r in bad_runs for r in runs])
    print(f"    trials in bad runs (5,8,11): {in_bad.sum()}/50, accuracy = {correct[in_bad].mean()*100:5.1f}%")
    print(f"    trials in good runs        : {(~in_bad).sum()}/50, accuracy = {correct[~in_bad].mean()*100:5.1f}%")
    return img_acc, bra_acc


def score_per_rep_refined(beta_arr, label):
    first_rep_idx = sorted(min(idxs) for idxs in by_img.values())
    names = [fast_ids[i] for i in first_rep_idx]
    paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in names]
    gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
    x = torch.from_numpy(beta_arr[first_rep_idx].astype(np.float32)).to(device)
    with torch.no_grad():
        x_ref = refiner(x)
    p = fwd_eval(model, ss, se, x_ref, batch=8).cpu().numpy()
    img_acc = topk(p, gt, 1); bra_acc = topk(gt, p, 1)
    print(f"  {label}: Image={img_acc*100:5.1f}%  Brain={bra_acc*100:5.1f}%", flush=True)
    return img_acc, bra_acc


print(f"\n=== ses-02 deployment test (rtQA flagged 3 runs: 5, 8, 11) ===", flush=True)
score_per_rep(fast_b, "Fast (no refiner)")
score_per_rep_refined(fast_b, "Fast (refined, ses-03-trained refiner)")
score_per_rep(slow_b, "Streaming Slow GLM (no refiner)")
score_per_rep_refined(slow_b, "Streaming Slow GLM (refined sanity)")

print("\n=== ses-01 reference (cleaner session) ===")
print("  Fast no refiner: 38/36 single-rep, 54/56 avg-of-2, 66/68 avg-of-3")
print("  Streaming Slow:  54/36 single-rep, 64/68 avg-of-2, 78/82 avg-of-3")
