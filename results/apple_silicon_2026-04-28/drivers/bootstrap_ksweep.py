"""Bootstrap 2-AFC + AUC + Cohen's d on K-sweep cells to verify K=7 is
statistically separable from neighbors."""
from pathlib import Path
import sys, types, warnings
import numpy as np
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

CKPT = LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth"
CACHE = LOCAL / "task_2_1_betas/gt_cache"
PREREG = LOCAL / "task_2_1_betas/prereg"
device = "mps" if torch.backends.mps.is_available() else "cpu"
N_BOOT = 5000


def filter_first_rep(arr, ids):
    seen, keep = set(), []
    for i, t in enumerate(ids):
        ts = str(t)
        if not ts.startswith("all_stimuli/special515/"): continue
        if ts in seen: continue
        seen.add(ts); keep.append(i)
    return arr[keep], np.asarray([str(ids[i]) for i in keep])


model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)


def two_afc(sim, idx):
    sub = sim[np.ix_(idx, idx)]
    n = len(idx)
    correct = 0; total = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if sub[i, i] > sub[i, j]: correct += 1
            total += 1
    return correct / total


def auc_d(sim, idx):
    sub = sim[np.ix_(idx, idx)]
    n = len(idx)
    merge = np.diag(sub)
    sep = sub[~np.eye(n, dtype=bool)]
    d = (merge.mean() - sep.mean()) / np.sqrt(0.5 * (merge.var() + sep.var()) + 1e-12)
    s = np.concatenate([merge, sep])
    y = np.concatenate([np.ones(len(merge)), np.zeros(len(sep))])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = (y == 1).sum(); n_neg = (y == 0).sum()
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), float(d)


def get_sim(cell):
    raw = np.load(PREREG / f"{cell}_ses-03_betas.npy")
    ids = np.load(PREREG / f"{cell}_ses-03_trial_ids.npy")
    test, kept = filter_first_rep(raw, ids)
    paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(str(n)).name for n in kept]
    gt = compute_gt_mps(paths, device=device, cache_dir=CACHE)
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            cv = model.backbone(voxel_ridge)
            cv = cv[1] if isinstance(cv, tuple) else cv
            out.append(cv.float().cpu().numpy())
    pred = np.concatenate(out, 0).reshape(-1, ss, se)
    return M.cosine_sim_tokens(pred, gt)


CELLS = [f"RT_paper_EoR_K{k}_CSFWM_inclz" for k in [0, 3, 5, 7, 10, 15, 20]]
CELLS.append("RTmotion_GLMsingle_singleRep")
KMAP = {"K0": 0, "K3": 3, "K5": 5, "K7": 7, "K10": 10, "K15": 15, "K20": 20, "GLMsingle": "GLM"}

print(f"\n{'cell':45s}  {'2-AFC':>15s}  {'AUC':>15s}  {'Cohen d':>13s}")
print("-" * 95)
results = []
sims_cache = {}
for cell in CELLS:
    if not (PREREG / f"{cell}_ses-03_betas.npy").exists():
        print(f"  SKIP {cell}")
        continue
    sim = get_sim(cell)
    sims_cache[cell] = sim
    n = sim.shape[0]
    rng = np.random.default_rng(42)
    afc_b, auc_b, d_b = [], [], []
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        afc_b.append(two_afc(sim, idx))
        a, dd = auc_d(sim, idx)
        auc_b.append(a); d_b.append(dd)
    afc_b = np.array(afc_b); auc_b = np.array(auc_b); d_b = np.array(d_b)
    afc_lo, afc_hi = np.percentile(afc_b, [2.5, 97.5])
    auc_lo, auc_hi = np.percentile(auc_b, [2.5, 97.5])
    d_lo, d_hi = np.percentile(d_b, [2.5, 97.5])
    print(f"{cell:45s}  {afc_b.mean()*100:5.2f}% [{afc_lo*100:.1f}-{afc_hi*100:.1f}]  "
          f"{auc_b.mean():.3f} [{auc_lo:.3f}-{auc_hi:.3f}]  "
          f"{d_b.mean():4.2f} [{d_lo:4.2f}-{d_hi:4.2f}]")
    results.append({"cell": cell,
                     "afc_mean": float(afc_b.mean()), "afc_ci": [float(afc_lo), float(afc_hi)],
                     "auc_mean": float(auc_b.mean()), "auc_ci": [float(auc_lo), float(auc_hi)],
                     "d_mean": float(d_b.mean()), "d_ci": [float(d_lo), float(d_hi)]})

# Pairwise: K=7 vs neighbors
print("\n=== K=7 vs neighbors: Δ 2-AFC bootstrap CI ===")
ref_sim = sims_cache.get("RT_paper_EoR_K7_CSFWM_inclz")
ref_n = ref_sim.shape[0]
for cell in CELLS:
    if cell == "RT_paper_EoR_K7_CSFWM_inclz" or cell not in sims_cache:
        continue
    other_sim = sims_cache[cell]
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.choice(ref_n, size=ref_n, replace=True)
        diffs.append(two_afc(ref_sim, idx) - two_afc(other_sim, idx))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    sig = "*sig*" if (lo > 0 or hi < 0) else "ns"
    print(f"  K=7 vs {cell:40s}  Δ={diffs.mean()*100:+5.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}]  {sig}")

import json
out = {"n_boot": N_BOOT, "results": results}
out_path = LOCAL / "task_2_1_betas/k_sweep_bootstrap.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nsaved {out_path.name}")
