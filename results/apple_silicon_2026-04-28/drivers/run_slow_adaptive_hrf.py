#!/usr/bin/env python3
"""Slow tier with adaptive per-trial decode_TR from `tr_label_hrf`.

Hypothesis: paper's reported Slow stim delay 29.45±2.63s reflects
per-trial timing where decode_TR adapts to that trial's HRF peak (the TR
where `tr_label_hrf` first matches the trial's image_name in the
canonical published `tr_labels.csv`). Fixed-pst from stim onset can't
match because trial-to-trial timing in the HRF peak varies (paper Table 4
±2.63s SD ≈ ±1.75 TR which fixed pst can't produce).

Adaptive rule:
    decode_TR = hrf_peak_TR_for_trial + N_post_peak

with N_post_peak ∈ {15, 20} swept. N=15 (post-peak) corresponds roughly
to a ~22-25 TR window from stim onset (HRF peaks at 4-5 TR + 15-20
post-peak TR), in the range where pst=25 was the best fixed setting.

The published `tr_labels.csv` files are at:
    rt3t/data/events/sub-005_ses-XX_task-C_run-XX_tr_labels.csv

Each has columns [tr_list, tr_trial_name, tr_label_hrf]. tr_label_hrf has
the trial's image_name at the row where the HRF should peak for that
trial (4-7 TRs after stim onset). This is what canonical mindeye.py:665
reads for the Fast tier.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    return z, list(image_history)


def load_tr_labels(session: str, run: int) -> pd.DataFrame:
    p = R.EVENTS_DIR / f"sub-005_{session}_task-C_run-{run:02d}_tr_labels.csv"
    return pd.read_csv(p)


def hrf_peak_TR_per_trial(events: pd.DataFrame, tr_labels: pd.DataFrame) -> list[int | None]:
    """For each trial in events, find the first TR where tr_label_hrf matches
    the trial's image_name. Returns one int per trial; None if not found."""
    out = []
    for _, ev in events.iterrows():
        img = str(ev.get("image_name", ""))
        # Match exact filename in tr_label_hrf column
        matches = tr_labels[tr_labels["tr_label_hrf"] == img]
        if len(matches) == 0:
            out.append(None)
            continue
        out.append(int(matches.iloc[0]["tr_list"]))
    return out


def run_adaptive_slow(cell_name: str, n_post_peak: int):
    flat_brain, rel = R.load_brain_paper_mask()
    tr = 1.5
    mask_img = nib.load(R.BRAIN_MASK)
    beta_history = []
    image_history = []
    n_skipped = 0
    config = {
        "cell": cell_name, "session": "ses-03", "tr": tr,
        "decode_rule": f"adaptive: decode_TR = hrf_peak_TR + {n_post_peak}",
        "n_post_peak": n_post_peak,
        "cum_z_formula": "inclusive (arr[:i+1])",
        "bold_source": "rtmotion",
        "tr_labels_source": "rt3t/data/events/*_tr_labels.csv  tr_label_hrf",
    }

    for run in R.runs_to_iter:
        try:
            bold_4d = R.load_rtmotion_4d("ses-03", run)
        except FileNotFoundError as e:
            print(f"  SKIP run-{run:02d}: {e}")
            continue
        events_path = R.EVENTS_DIR / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv"
        events = pd.read_csv(events_path, sep="\t")
        tr_labels = load_tr_labels("ses-03", run)
        hrf_peaks = hrf_peak_TR_per_trial(events, tr_labels)
        mc_params = R.load_mc_params("ses-03", run)
        n_trs_run = bold_4d.shape[3]

        for trial_i in range(len(events)):
            t0 = time.time()
            peak = hrf_peaks[trial_i]
            if peak is None:
                # blank.jpg or untracked trial: use simple onset+5 fallback
                onset_sec = float(events.iloc[trial_i]["onset"]) - float(events.iloc[0]["onset"])
                onset_TR = int(round(onset_sec / tr))
                decode_TR = min(onset_TR + 5, n_trs_run - 1)
            else:
                decode_TR = min(peak + n_post_peak, n_trs_run - 1)

            beta_vol = R.fit_lss_nilearn(
                bold_4d, events, trial_i, mc_params,
                tr=tr, mask_img=mask_img, streaming_decode_TR=decode_TR,
            )
            if beta_vol is None:
                n_skipped += 1
                continue
            beta_masked = beta_vol.flatten()[flat_brain][rel]
            beta_history.append(beta_masked.astype(np.float32))
            image_history.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))
            tag = f"peak={peak}+{n_post_peak}={decode_TR}" if peak is not None else f"fallback_TR={decode_TR}"
            if trial_i % 10 == 0:
                print(f"  {cell_name} run-{run:02d} trial {trial_i:3d} {tag} ({time.time()-t0:.2f}s)")

    print(f"  total skipped: {n_skipped}")
    z, ids = inclusive_cumz(beta_history, image_history, do_repeat_avg=False)
    np.save(R.OUT_DIR / f"{cell_name}_ses-03_betas.npy", z)
    np.save(R.OUT_DIR / f"{cell_name}_ses-03_trial_ids.npy", np.asarray(ids))
    config["n_trials_post"] = int(z.shape[0])
    with open(R.OUT_DIR / f"{cell_name}_ses-03_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell_name}: betas {z.shape}")


# expose the iterable used in run_adaptive_slow
R.runs_to_iter = list(range(1, 12))

CELLS = [
    ("RT_paper_Slow_adaptive_n15_inclz", 15),
    ("RT_paper_Slow_adaptive_n20_inclz", 20),
]
for name, n in CELLS:
    print(f"\n=== {name}  (adaptive: hrf_peak_TR + {n} post-peak TRs) ===")
    t0 = time.time()
    run_adaptive_slow(name, n)
    print(f"  total time: {time.time()-t0:.1f}s")
