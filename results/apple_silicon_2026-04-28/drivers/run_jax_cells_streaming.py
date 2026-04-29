#!/usr/bin/env python3
"""Streaming-cropped versions of JAX cells 1, 2, 4 (OLS / AR1freq / VariantG).

For each probe trial, crop the timeseries and onsets to decode_TR = onset_TR
+ post_stim_TRs (paper RT pipeline: only data accumulated up to that TR is
visible). Compare against the existing full-run versions to read off the
windowing contribution to the Offline-vs-RT gap — the actual core question
of Task 2.1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
from prereg_variant_sweep import _glm_jax, load_mask, load_rtmotion, load_events

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
P.PAPER_ROOT = LOCAL
P.RT3T = LOCAL / "rt3t" / "data"
P.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye" / "data_sub-005"
                   / "bids" / "derivatives" / "fmriprep" / "sub-005")
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"
P.HRF_INDICES_PATH = str(LOCAL / "rt3t" / "data" / "avg_hrfs_s1_s2_full.npy")
P.HRF_LIB_PATH = str(LOCAL / "rt3t" / "data" / "getcanonicalhrflibrary.tsv")

POST_STIM_TRS = 8       # match the streaming pst=8 anchor that landed at 68%
SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5

# Cells we want streaming versions of (the GLM-choice axis, on rtmotion BOLD)
CELLS = [
    ("OLS_glover_rtm_streaming_pst8", "ols"),
    ("AR1freq_glover_rtm_streaming_pst8", "ar1_freq"),
    ("VariantG_glover_rtm_streaming_pst8", "variant_g"),
]


def run_streaming_cell(cell_name: str, mode: str) -> None:
    flat_brain, rel = load_mask()
    all_betas, trial_ids = [], []

    for run in RUNS:
        ts = load_rtmotion(SESSION, run, flat_brain, rel)   # (V, T)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        for trial_i in range(len(onsets)):
            onset_TR = int(round(onsets[trial_i] / TR))
            decode_TR = min(onset_TR + POST_STIM_TRS, n_trs_run - 1)
            n_eff = decode_TR + 1
            if n_eff < 8:    # not enough data; skip (mirrors nilearn's behavior)
                continue
            # Crop BOLD to [..., :n_eff]; onsets stay (build_design_matrix uses n_trs to clip)
            ts_crop = ts[:, :n_eff]
            beta, _ = _glm_jax(ts_crop, onsets, trial_i, TR, n_eff,
                                mode=mode, max_trs=200)
            all_betas.append(np.asarray(beta, dtype=np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(P.OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    config = {"cell": cell_name, "mode": mode, "session": SESSION, "runs": RUNS,
              "tr": TR, "streaming_post_stim_TRs": POST_STIM_TRS,
              "bold_source": "rtmotion", "hrf_strategy": "glover_delta_onset"}
    with open(P.OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell_name}: betas {betas.shape}")


if __name__ == "__main__":
    for name, mode in CELLS:
        print(f"\n=== {name} ({mode}, pst={POST_STIM_TRS}) ===")
        t0 = time.time()
        run_streaming_cell(name, mode)
        print(f"  elapsed: {time.time() - t0:.1f}s")
