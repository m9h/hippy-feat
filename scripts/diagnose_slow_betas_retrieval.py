#!/usr/bin/env python3
"""Retrieval-only diagnostic for the streaming-RLS Slow β chance-level finding.

Job 1099 hit chance on `RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03`
through fold-0 (4% Image / 2% Brain), but apple-silicon claims 54%/58% on
the same filename. Before triaging with the Apple agent, fan out across
plausible (β source, ckpt, voxel-permutation) combinations and report which
ones are non-chance.

No SDXL — just `β → ridge → backbone → clip_voxels → cosine_sim(GT)`. Each
config is ~1 sec on GPU. Total runtime ~2 min including model loads.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

LOCAL = Path("/data/derivatives/rtmindeye_paper")
PREREG = LOCAL / "task_2_1_betas" / "prereg"
GT_CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
STIMULI = LOCAL / "rt3t" / "data" / "all_stimuli" / "special515"
RT3T = LOCAL / "rt3t" / "data"
RT_MINDEYE = LOCAL / "repos" / "rtcloud-projects" / "mindeye" / "rt_mindEye2" / "src"
sys.path.insert(0, str(RT_MINDEYE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mindeye_retrieval_eval as M

warnings.filterwarnings("ignore")

CKPTS = {
    "fold-0":  Path("/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"),
    "fold-10": Path("/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth"),
}

# Cells to test. Stem is what's before `_ses-03_betas.npy`.
CELLS = [
    # apple-silicon teacher target
    "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz",
    # streaming RLS without inclusive z, in case z-policy is the issue
    "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw",
    # paper-canonical Slow tier (Rishab's pre-saved delay=5)
    "Paper_RT_actual_delay5",
    # apple-silicon teacher should also work on EoR
    "RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz",
    # paper-canonical EoR tier
    "Paper_RT_actual_delay63",
    # Fast sanity (we know this one matches paper at 36-38%)
    "Paper_RT_actual_delay0",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== diagnose_slow_betas_retrieval.py  device={device} ===\n", flush=True)


def topk(p: np.ndarray, g: np.ndarray, k: int = 1) -> float:
    pf = p.reshape(p.shape[0], -1)
    gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def session_zscore(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-8
    return ((arr - mu) / sd).astype(np.float32)


def causal_cum_zscore(arr: np.ndarray) -> np.ndarray:
    """Strict causal: trial i uses stats from 0..i-1 only.
    Same as score_full_metrics.cumulative_zscore."""
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        if i < 2:
            mu = arr[:max(i, 1)].mean(0, keepdims=True) if i > 0 else 0.0
            sd = 1.0
        else:
            mu = arr[:i].mean(0, keepdims=True)
            sd = arr[:i].std(0, keepdims=True) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out


def fwd_clip_voxels(model, ss: int, se: int, x: torch.Tensor,
                     batch: int = 8) -> np.ndarray:
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        b = x.unsqueeze(1)
        for i in range(0, b.shape[0], batch):
            vr = model.ridge(b[i:i + batch], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float())
    return torch.cat(out, 0).reshape(-1, ss, se).cpu().numpy()


def first_rep_test_split(ids: np.ndarray) -> np.ndarray:
    counts = Counter(ids)
    test_imgs_set = {n for n in ids if counts[n] == 3 and "special515" in n}
    seen = set()
    out = []
    for i, n in enumerate(ids):
        if n in test_imgs_set and n not in seen:
            seen.add(n)
            out.append(i)
    return np.array(sorted(out))


def load_gt(image_paths: list[Path]) -> np.ndarray:
    keys = [GT_CACHE / f"{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}.npy"
             for p in image_paths]
    return np.stack([np.load(k) for k in keys])


# ---------------------------------------------------------------------------
# Per-cell diagnostic
# ---------------------------------------------------------------------------
def evaluate(model, ss: int, se: int, name: str, betas: np.ndarray,
              ids: np.ndarray, stem_label: str) -> dict:
    test_idx = first_rep_test_split(ids)
    if len(test_idx) != 50:
        return {"cell": name, "stem_label": stem_label, "skipped": True,
                "n_test": int(len(test_idx))}
    test_ids = ids[test_idx]
    test_paths = [STIMULI / Path(n).name for n in test_ids]
    gt = load_gt(test_paths)
    x = torch.from_numpy(betas[test_idx].astype(np.float32)).to(device)
    pred = fwd_clip_voxels(model, ss, se, x, batch=8)
    return {"cell": name, "stem_label": stem_label,
            "n_test": int(len(test_idx)),
            "image_top1": topk(pred, gt, 1),
            "brain_top1": topk(gt, pred, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
results = []
for ckpt_label, ckpt_path in CKPTS.items():
    print(f"\n=== {ckpt_label}: {ckpt_path.name} ===", flush=True)
    t0 = time.time()
    model, ss, se = M.load_mindeye(ckpt_path, n_voxels=2792, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    for cell in CELLS:
        beta_path = PREREG / f"{cell}_ses-03_betas.npy"
        ids_path = PREREG / f"{cell}_ses-03_trial_ids.npy"
        if not beta_path.exists():
            print(f"  {cell}: SKIP (no βs)", flush=True)
            continue
        betas = np.load(beta_path)
        ids = np.load(ids_path, allow_pickle=True)
        ids = np.asarray([str(t) for t in ids])

        # Native (whatever z-policy was baked into the file)
        r = evaluate(model, ss, se, f"{ckpt_label} | {cell} | native",
                      betas, ids, "native")
        results.append({"ckpt": ckpt_label, **r})
        print(f"  {cell:<55s} native:  "
              f"Image={r.get('image_top1', 0)*100:5.1f}%  "
              f"Brain={r.get('brain_top1', 0)*100:5.1f}%", flush=True)

        # If raw cell, additionally try session_zscore + causal_cum_zscore
        if cell.endswith("_raw") or cell.startswith("Paper_RT_actual"):
            for zname, zfn in [("session_z", session_zscore),
                                ("causal_cz", causal_cum_zscore)]:
                z_betas = zfn(betas)
                r = evaluate(model, ss, se, f"{ckpt_label} | {cell} | {zname}",
                              z_betas, ids, zname)
                results.append({"ckpt": ckpt_label, **r})
                print(f"  {cell:<55s} {zname:>9s}:  "
                      f"Image={r.get('image_top1', 0)*100:5.1f}%  "
                      f"Brain={r.get('brain_top1', 0)*100:5.1f}%", flush=True)

# ---------------------------------------------------------------------------
out = LOCAL / "task_2_1_betas" / "diagnose_slow_betas_retrieval.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}", flush=True)
