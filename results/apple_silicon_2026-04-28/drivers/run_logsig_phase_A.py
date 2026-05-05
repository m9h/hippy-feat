#!/usr/bin/env python3
"""Phase A — zero-training log-signature feature replacement / mixing test.

For each ses-03 trial, per voxel, compute over post-stim BOLD window:
  - increment   = ΔBOLD = BOLD[onset+pst] - BOLD[onset]
  - levy_area   = ½ ∫(t dBOLD - BOLD dt)  [Lévy area against time]

Test 3 input configurations through fold-0 fwd:
  1. increment alone (sanity = boxcar-HRF-equivalent β)
  2. levy_area alone (does Lévy area carry signal?)
  3. β + α·levy_area for α grid (does mixing help?)

Compare to existing β baseline:
  Fast (pst=5):  rtmotion β subset0 = 36%
  Slow (pst=20): rtmotion β subset0 = 44%

Decision: if max-α config beats baseline by >4pp on either tier, proceed to Phase B.
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np, nibabel as nib, pandas as pd, torch, torch.nn as nn

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
SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
MC_DIR = LOCAL / "motion_corrected_resampled"
EVENTS_DIR = LOCAL / "rt3t/data/events"

# === Load brain mask + relmask ===
final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")

# === Load rtmotion BOLD per run, mask to 2792 voxels ===
print("=== loading rtmotion BOLD per run + masking to 2792 vox ===", flush=True)
bold_per_run = []
for run in RUNS:
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    frames = [nib.load(v).get_fdata().flatten()[final_mask][relmask].astype(np.float32) for v in vols]
    bold_per_run.append(np.stack(frames, axis=0))    # (T_run, 2792)
    print(f"  run-{run:02d}: BOLD ({bold_per_run[-1].shape[0]}, 2792)", flush=True)


# === Compute log-sig features per trial per voxel ===
def compute_features(bold_run, onset_TR, pst):
    """Return (increment, levy_area) for each voxel, given BOLD[onset:onset+pst+1, V]."""
    end = min(onset_TR + pst + 1, bold_run.shape[0])
    window = bold_run[onset_TR:end]                 # (W, V)
    if window.shape[0] < 2:
        V = bold_run.shape[1]
        return np.zeros(V, dtype=np.float32), np.zeros(V, dtype=np.float32)
    increment = window[-1] - window[0]              # (V,)
    # Lévy area: ½ Σ_i [(i + 0.5)(x_{i+1} - x_i) - (x_i + x_{i+1})/2]
    diffs = window[1:] - window[:-1]                # (W-1, V)
    sums  = window[:-1] + window[1:]                # (W-1, V)
    t_mid = (np.arange(window.shape[0] - 1, dtype=np.float32) + 0.5)[:, None]
    levy = 0.5 * (t_mid * diffs - 0.5 * sums).sum(axis=0)
    return increment.astype(np.float32), levy.astype(np.float32)


def features_for_pst(pst):
    """Iterate all trials (including blanks) to match β-cell indexing of 770."""
    print(f"\n=== computing features for pst={pst} ===", flush=True)
    incr_per_trial, levy_per_trial, names_per_trial = [], [], []
    for run_idx, run in enumerate(RUNS):
        ev = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv", sep="\t")
        run_start = float(ev.iloc[0]["onset"])
        for _, row in ev.iterrows():
            name = row["image_name"]
            if pd.isna(name):
                name = "nan"
            try:
                onset_TR = int(round((float(row["onset"]) - run_start) / TR))
                inc, lev = compute_features(bold_per_run[run_idx], onset_TR, pst)
            except Exception:
                V = bold_per_run[run_idx].shape[1]
                inc = np.zeros(V, dtype=np.float32); lev = np.zeros(V, dtype=np.float32)
            incr_per_trial.append(inc)
            levy_per_trial.append(lev)
            names_per_trial.append(str(name))
    return (np.stack(incr_per_trial), np.stack(levy_per_trial), np.asarray(names_per_trial))


# === Load existing β baselines for matched cells ===
PREREG = LOCAL / "task_2_1_betas/prereg"
def load_betas_for(prefix):
    b = np.load(PREREG / f"{prefix}_ses-03_betas.npy")
    ids = np.load(PREREG / f"{prefix}_ses-03_trial_ids.npy")
    return b, np.asarray([str(t) for t in ids])


# === Inclusive cum-z over a (n_trials, V) array ===
def inclusive_cumz(arr):
    n, V = arr.shape
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i+1].mean(axis=0)
        sd = arr[:i+1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


# === First-rep filter for special515 ===
def filter_first_rep(arr, names):
    seen, keep = set(), []
    for i, n in enumerate(names):
        if "special515" not in n: continue
        if n in seen: continue
        seen.add(n)
        keep.append(i)
    keep = np.array(keep)
    return arr[keep], [names[i] for i in keep]


print("\n=== loading fold-0 ckpt ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)


def fwd(arr):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(arr.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(0, b.shape[0], 8):
            vr = model.ridge(b[i:i+8], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, 0).reshape(-1, ss, se)


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


configs = [
    ("Fast", 5,  "RT_paper_Fast_pst5_inclz", 0.36),
    ("Slow", 20, "RT_paper_Slow_pst20_inclz", 0.58),
]

results = {}

for tier, pst, beta_cell, paper_anchor in configs:
    incr_all, levy_all, names_all = features_for_pst(pst)
    # cum-z each feature stream independently
    incr_z = inclusive_cumz(incr_all)
    levy_z = inclusive_cumz(levy_all)

    # Get matching β baseline (already cum-z'd & paper-pipeline-extracted)
    beta_z, beta_names = load_betas_for(beta_cell)
    assert len(beta_names) == len(names_all), \
        f"trial count mismatch: β {len(beta_names)} vs ours {len(names_all)}"

    # First-rep filter (using same names ordering as β, which we trust)
    incr_z_test, _ = filter_first_rep(incr_z, names_all)
    levy_z_test, _ = filter_first_rep(levy_z, names_all)
    beta_z_test, test_names = filter_first_rep(beta_z, beta_names)
    assert len(test_names) == 50

    # GT
    img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names]
    gt = compute_gt_mps(img_paths, device=device, cache_dir=LOCAL / "task_2_1_betas/gt_cache")

    print(f"\n=== {tier} (pst={pst}) ===", flush=True)
    tier_results = {}

    # 1. β alone (baseline)
    p = fwd(beta_z_test)
    img = topk(p, gt, 1); bra = topk(gt, p, 1)
    tier_results["beta_alone"] = {"image": img, "brain": bra}
    print(f"  β alone:                Image={img*100:5.1f}%  Brain={bra*100:5.1f}%  (paper {tier} {paper_anchor*100:.0f}%)")

    # 2. increment alone
    p = fwd(incr_z_test)
    img = topk(p, gt, 1); bra = topk(gt, p, 1)
    tier_results["increment_alone"] = {"image": img, "brain": bra}
    print(f"  increment alone:        Image={img*100:5.1f}%  Brain={bra*100:5.1f}%")

    # 3. Lévy alone
    p = fwd(levy_z_test)
    img = topk(p, gt, 1); bra = topk(gt, p, 1)
    tier_results["levy_alone"] = {"image": img, "brain": bra}
    print(f"  Lévy alone:             Image={img*100:5.1f}%  Brain={bra*100:5.1f}%")

    # 4. β + α·Lévy mixing grid
    print("  --- β + α·Lévy mixing ---")
    mix = {}
    for alpha in [-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0]:
        x = beta_z_test + alpha * levy_z_test
        p = fwd(x)
        img = topk(p, gt, 1); bra = topk(gt, p, 1)
        mix[f"{alpha:+.1f}"] = {"image": img, "brain": bra}
        print(f"    α={alpha:+5.1f}:  Image={img*100:5.1f}%  Brain={bra*100:5.1f}%")
    tier_results["beta_plus_alpha_levy"] = mix

    # 5. β + α·increment mixing grid (sanity, since we have inc separately)
    print("  --- β + α·increment mixing ---")
    mix2 = {}
    for alpha in [-0.5, -0.1, 0.0, 0.1, 0.5]:
        x = beta_z_test + alpha * incr_z_test
        p = fwd(x)
        img = topk(p, gt, 1); bra = topk(gt, p, 1)
        mix2[f"{alpha:+.1f}"] = {"image": img, "brain": bra}
        print(f"    α={alpha:+5.1f}:  Image={img*100:5.1f}%  Brain={bra*100:5.1f}%")
    tier_results["beta_plus_alpha_increment"] = mix2

    results[tier] = tier_results

(LOCAL / "task_2_1_betas/logsig_phase_A.json").write_text(json.dumps(results, indent=2))
print("\nsaved logsig_phase_A.json", flush=True)

# Decision
print("\n========== DECISION ==========")
for tier in ("Fast", "Slow"):
    base = results[tier]["beta_alone"]["image"]
    best_alpha = max(results[tier]["beta_plus_alpha_levy"].values(), key=lambda r: r["image"])
    best_inc = max(results[tier]["beta_plus_alpha_increment"].values(), key=lambda r: r["image"])
    levy_alone = results[tier]["levy_alone"]["image"]
    increment_alone = results[tier]["increment_alone"]["image"]
    delta_levy_mix = (best_alpha["image"] - base) * 100
    delta_inc_mix = (best_inc["image"] - base) * 100
    print(f"  {tier}: β={base*100:.1f}%, best β+α·Lévy={best_alpha['image']*100:.1f}% (Δ {delta_levy_mix:+.1f}pp), "
          f"best β+α·increment={best_inc['image']*100:.1f}% (Δ {delta_inc_mix:+.1f}pp), "
          f"Lévy-alone={levy_alone*100:.1f}%, increment-alone={increment_alone*100:.1f}%")

print("\nProceed to Phase B if any Δ > +4pp.")
