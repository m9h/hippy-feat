#!/usr/bin/env python3
"""Score the Slow tier diagnostic cells with single-rep filter.

Cells:
  RT_paper_Slow_pst25_inclz       — wider window: 25 TRs ≈ 37.5s
  RT_paper_Slow_pst30_inclz       — wider window: 30 TRs ≈ 45s
  RT_paper_Slow_pst20_canonrez    — pst=20 + full-session re-z (canonical-mindeye-style)
"""
from __future__ import annotations

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

PREREG = LOCAL / "task_2_1_betas" / "prereg"
CKPT = LOCAL / "rt3t" / "data" / "model" / "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
RESULTS_PATH = LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"
device = "mps" if torch.backends.mps.is_available() else "cpu"

CELLS = [
    "RT_paper_Slow_pst25_inclz",
    "RT_paper_Slow_pst30_inclz",
    "RT_paper_Slow_pst20_canonrez",
]
PAPER_ANCHOR = 0.58
SLOW_BASELINE = 0.44
SES = "ses-03"


def filter_first_rep(betas, trial_ids):
    keep_idx, seen = [], set()
    for i, t in enumerate(trial_ids):
        ts = str(t)
        if not ts.startswith("all_stimuli/special515/"):
            continue
        if ts in seen:
            continue
        seen.add(ts)
        keep_idx.append(i)
    return betas[keep_idx], [str(trial_ids[i]) for i in keep_idx]


model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)


def score(name):
    raw_betas = np.load(PREREG / f"{name}_{SES}_betas.npy")
    raw_ids = np.load(PREREG / f"{name}_{SES}_trial_ids.npy")
    test_betas, test_ids = filter_first_rep(raw_betas, raw_ids)
    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            cv = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(cv.float().cpu().numpy())
    pred = np.concatenate(out, axis=0).reshape(-1, ss, se)
    img_to_idx = {im: i for i, im in enumerate(test_ids)}
    trial_idx = np.array([img_to_idx[t] for t in test_ids])
    sim = M.cosine_sim_tokens(pred, gt)
    top1 = M.top_k_retrieval(sim, trial_idx, k=1)
    top5 = M.top_k_retrieval(sim, trial_idx, k=5)
    return float(top1), float(top5), int(test_betas.shape[0])


print(f"\n{'cell':45s}  {'top1':>7s}  {'top5':>7s}  {'paper':>7s}  {'Δpaper':>7s}  {'ΔSlowBase':>11s}")
results = json.loads(RESULTS_PATH.read_text())
for name in CELLS:
    if not (PREREG / f"{name}_{SES}_betas.npy").exists():
        print(f"{name:45s}  SKIP (betas not on disk)")
        continue
    t1, t5, n = score(name)
    print(f"{name:45s}  "
          f"{t1*100:6.2f}%  {t5*100:6.2f}%  "
          f"{PAPER_ANCHOR*100:6.0f}%  "
          f"{(t1-PAPER_ANCHOR)*100:+6.1f}pp  "
          f"{(t1-SLOW_BASELINE)*100:+10.1f}pp")
    res = {
        "condition": f"{name}_singleRep",
        "session": SES, "n_test_trials": n, "n_unique_images": 50,
        "top1_image_retrieval": t1, "top5_image_retrieval": t5,
        "paper_anchor_top1": PAPER_ANCHOR,
        "delta_vs_paper_pp": (t1 - PAPER_ANCHOR) * 100,
        "delta_vs_slow_baseline_pp": (t1 - SLOW_BASELINE) * 100,
    }
    results = [r for r in results if r.get("condition") != res["condition"]]
    results.append(res)

RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f"\nappended to {RESULTS_PATH.name}")
