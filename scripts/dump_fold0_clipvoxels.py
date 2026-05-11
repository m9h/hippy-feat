#!/usr/bin/env python3
"""Dump fold-0 forward outputs for the 50 first-rep test trials.

For each of the three β cells we care about (delay0 Fast, _kbm Fast, _kbm Slow),
load → forward through fold-0 (ridge + backbone) → extract `clip_voxels`.
Save as a single .npz with both `clip_voxels` and trial_ids so the Apple
agent can do element-wise compare against the Mac fold-0 forward outputs.

Per Apple agent reply 2026-05-10 follow-up #3: if Image=54 matches but
Brain=48 vs Mac's 58, the perturbation is in either the βs themselves or
in mindeye_retrieval_eval. Dumping clip_voxels both sides isolates which.
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

LOCAL = Path("/data/derivatives/rtmindeye_paper")
PREREG = LOCAL / "task_2_1_betas" / "prereg"
STIMULI = LOCAL / "rt3t" / "data" / "all_stimuli" / "special515"
RT_MINDEYE = LOCAL / "repos" / "rtcloud-projects" / "mindeye" / "rt_mindEye2" / "src"
sys.path.insert(0, str(RT_MINDEYE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mindeye_retrieval_eval as M

warnings.filterwarnings("ignore")

CKPT = Path(
    "/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
)
CELLS = [
    "Paper_RT_actual_delay0",
    "RT_paper_RLS_Fast_pst5_K7CSFWM_HP_e1_raw_kbm",
    "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw_kbm",
]
OUT = LOCAL / "task_2_1_betas" / "fold0_clipvoxels_parity_dgx.npz"


def first_rep_idx(ids: np.ndarray) -> np.ndarray:
    counts = Counter(ids)
    test = {n for n in ids if counts[n] == 3 and "special515" in n}
    seen, out = set(), []
    for i, n in enumerate(ids):
        if n in test and n not in seen:
            seen.add(n)
            out.append(i)
    return np.array(sorted(out))


def fwd(model, ss, se, betas, device, batch=8):
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        x = torch.from_numpy(betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(0, x.shape[0], batch):
            vr = model.ridge(x[i:i + batch], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, 0).reshape(-1, ss, se)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

print("loading fold-0 ckpt ...", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  ss={ss}  se={se}", flush=True)

out_dict = {"ckpt": str(CKPT), "ss": ss, "se": se}
for cell in CELLS:
    bf = PREREG / f"{cell}_ses-03_betas.npy"
    idf = PREREG / f"{cell}_ses-03_trial_ids.npy"
    if not bf.exists():
        print(f"  SKIP {cell}: no βs at {bf.name}", flush=True)
        continue
    betas = np.load(bf)
    ids = np.asarray([str(t) for t in np.load(idf, allow_pickle=True)])
    idx = first_rep_idx(ids)
    if len(idx) != 50:
        print(f"  SKIP {cell}: first_rep_idx len={len(idx)} ≠ 50", flush=True)
        continue
    cv = fwd(model, ss, se, betas[idx], device)
    out_dict[f"{cell}__clip_voxels"] = cv
    out_dict[f"{cell}__trial_ids"] = ids[idx]
    print(f"  {cell}: clip_voxels={cv.shape}  ids[:3]={ids[idx][:3]}",
          flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out_dict)
print(f"\nwrote {OUT}", flush=True)
