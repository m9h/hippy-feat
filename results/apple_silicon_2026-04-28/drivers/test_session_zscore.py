#!/usr/bin/env python3
"""Test full-session (non-causal) z-score variant.

Hypothesis: paper's 'single-rep first-presentation' might be evaluated AFTER
the session ends, with stats computed over all 770 trials (non-causal).
This would differ from our causal cum-z especially for early-session test trials.

We have raw beta_history saved in beta_history.npy if we re-run cells with
that flag. But for now, we can approximate by re-z-scoring the saved (already
causally-z'd) betas with full-session stats — this is approximately equivalent
to applying full-session z to the raw betas, modulo the small inclusive vs
exclusive difference.

Better: re-run the cells but save raw beta_history before any z. Use a
simpler version that exits before z. Then this script applies multiple z
variants and scores.
"""
from __future__ import annotations

import json
import sys
import time
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
device = "mps" if torch.backends.mps.is_available() else "cpu"


def filter_first_rep_idx(trial_ids):
    keep_idx, seen = [], set()
    for i, t in enumerate(trial_ids):
        ts = str(t)
        if not ts.startswith("all_stimuli/special515/"):
            continue
        if ts in seen:
            continue
        seen.add(ts)
        keep_idx.append(i)
    return keep_idx


def fwd(test_betas, model, ss, se):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            cv = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1, ss, se)


# Approximation: take the saved (already cum-z'd) betas, take only test trials,
# and re-z-score those test trials with FULL-SESSION stats (computed on all
# trials). This is *almost* equivalent to applying global z to raw betas,
# modulo small differences in scale.
# The truly correct test would re-run cells without any z and apply global z
# from raw — but this approximation should be close enough to detect
# meaningful effects.

CELLS = [
    ("RT_paper_Fast_pst5_inclz",          0.36, "Fast"),
    ("RT_paper_Slow_pst20_inclz",         0.58, "Slow"),
    ("RT_paper_EndOfRun_pst_None_inclz",  0.66, "End-of-run"),
]

model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)

print(f"\n{'Tier':12s} {'Variant':30s} {'top-1':>6s} {'paper':>6s}")
for name, anchor, tier in CELLS:
    raw_betas = np.load(PREREG / f"{name}_ses-03_betas.npy")  # already cum-z'd
    raw_ids = np.load(PREREG / f"{name}_ses-03_trial_ids.npy")
    keep = filter_first_rep_idx(raw_ids)
    test_ids = [str(raw_ids[i]) for i in keep]

    # Variant A: as-is (causal cum-z, INCLUSIVE)
    test_betas_A = raw_betas[keep]
    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)
    pred_A = fwd(test_betas_A, model, ss, se)
    sim_A = M.cosine_sim_tokens(pred_A, gt)
    top1_A = M.top_k_retrieval(sim_A, np.arange(len(keep)), k=1)

    # Variant B: also re-z-score test betas with stats over the FULL 770-trial session
    # The saved betas are already cum-z'd. Re-z'ing them with their session-wide
    # mean/std is approximately equivalent to using full-session stats on raw betas.
    sess_mu = raw_betas.mean(axis=0, keepdims=True)
    sess_sd = raw_betas.std(axis=0, keepdims=True) + 1e-6
    test_betas_B = (raw_betas[keep] - sess_mu) / sess_sd
    pred_B = fwd(test_betas_B, model, ss, se)
    sim_B = M.cosine_sim_tokens(pred_B, gt)
    top1_B = M.top_k_retrieval(sim_B, np.arange(len(keep)), k=1)

    # Variant C: NO z-score at all — undo cum-z by applying inverse session
    # statistics (multiplied by saved std). This is wrong but informative.
    # Skip; can't easily invert without raw history.

    print(f"{tier:12s} {'causal cum-z (saved)':30s} {top1_A*100:5.1f}% {anchor*100:5.0f}%")
    print(f"{tier:12s} {'+full-session re-z':30s} {top1_B*100:5.1f}% {anchor*100:5.0f}%")
