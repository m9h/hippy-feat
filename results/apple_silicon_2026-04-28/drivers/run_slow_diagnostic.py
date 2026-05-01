#!/usr/bin/env python3
"""Slow tier diagnostic — wider pst sweep + canonical re-z.

Slow is the only paper anchor with a statistically significant gap
(p=0.033, paper 58% outside our [30, 58] CI). Prior pst sweep over 18-22
plateaued at 44-46%. This driver tests three hypotheses:

1. **Wider pst (25, 30)**: paper's reported 29.45s Slow stim delay may
   include HRF-peak alignment (HRF peaks ~5 TR after onset, plus the 20-TR
   window = ~25 TR from stim onset). pst=25 or pst=30 might recover 58%.

2. **Canonical re-z of older betas**: per the handoff memory, canonical
   mindeye.py re-z's previously-saved betas with the latest trial's stats
   when handling repeats. Our inclusive cum-z z's each beta once with
   stats including up-to-and-current. With single-rep filter at scoring
   the difference should be minor, but worth testing.

Save raw 770-trial betas; score with single-rep filter at evaluation time.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t" / "data"
R.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
R.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"


def _local_load_mc(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc


def inclusive_cumz(beta_history, image_history, do_repeat_avg):
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    if not do_repeat_avg:
        return z, list(image_history)
    seen = {}
    out_b, out_i = [], []
    for i, img in enumerate(image_history):
        seen.setdefault(img, []).append(i)
        if len(seen[img]) == 1:
            out_b.append(z[i]); out_i.append(img)
        else:
            avg = z[seen[img]].mean(axis=0)
            first_pos = next(j for j, l in enumerate(out_i) if l == img)
            out_b[first_pos] = avg
    return np.stack(out_b, axis=0), out_i


def canonical_rez(beta_history, image_history, do_repeat_avg):
    """Canonical mindeye.py re-z behavior: at each trial i, re-z ALL
    previously-saved (β_history) with stats over arr[:i+1]. The output for
    trial i uses the LATEST stats; earlier trials' published z-scores are
    overwritten with the current trial's stats applied to their raw βs.

    For single-rep evaluation the final β at each trial position is the
    raw β z-scored with stats over the FULL session up to and including
    its position — equivalent to inclusive cum-z. The difference vs
    inclusive cum-z only matters if you save intermediate (early-session)
    z-values; final-state is the same.

    HOWEVER: there is one behavior difference. Canonical re-z's older
    trials in PLACE so that when a repeat appears, the prior-rep β has
    been overwritten with current stats. With single-rep filter at
    scoring time, we keep the FIRST occurrence — meaning we keep the
    LATEST available z-version of that first occurrence at the time of
    saving. To replicate this we apply inclusive cum-z but using
    FULL-SESSION stats to every trial (n=770), not progressive stats.
    """
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-6
    z = (arr - mu) / sd
    return z, list(image_history)


SESSION = "ses-03"
RUNS = list(range(1, 12))

CONFIGS = [
    ("RT_paper_Slow_pst25_inclz", 25, "inclusive"),
    ("RT_paper_Slow_pst30_inclz", 30, "inclusive"),
    ("RT_paper_Slow_pst20_canonrez", 20, "canonical_rez"),
]

R.OUT_DIR.mkdir(parents=True, exist_ok=True)

for cell_name, pst, z_mode in CONFIGS:
    if z_mode == "inclusive":
        R.cumulative_zscore_with_optional_repeat_avg = inclusive_cumz
    else:
        R.cumulative_zscore_with_optional_repeat_avg = canonical_rez

    print(f"\n=== {cell_name}  (pst={pst} TRs ≈ {pst*1.5:.1f}s, z_mode={z_mode}) ===")
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell_name,
        bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS,
        do_repeat_avg=False,
        streaming_post_stim_TRs=pst,
    )
    config["cum_z_formula"] = z_mode
    config["streaming_post_stim_TRs"] = pst
    np.save(R.OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell_name}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
