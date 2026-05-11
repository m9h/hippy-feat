#!/usr/bin/env python3
"""Per-trial nilearn LSS extraction at pst=5 (Fast tier), all 770 events.

Replicates the paper's Fast tier β extraction more faithfully than our
streaming-RLS GLM does. Matches mindeye.py:745 — `FirstLevelModel(t_r=1.5,
slice_time_ref=0, hrf_model='glover', drift_model='cosine', drift_order=1,
high_pass=0.01, noise_model='ar1', signal_scaling=False)` with `mc_params`
(6 motion regressors from MCFLIRT) as confounds. For each trial, fits one
LSS with that trial as probe + all OTHER trials in the cropped window as
reference. BOLD window: `[0 : onset_TR + 5 + 1]`.

Keeps blank.jpg rows in the iteration (770 trials) to match apple-silicon's
v1 distillation recipe (their reply A2). Each blank gets a β fit anyway —
the resulting β is mostly noise (no real stimulus) but enters the train
pool just like Mac's pipeline.

Output: `RT_paper_Fast_pst5_lss_kbm_ses-03_betas.npy` shape (770, 2792).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

PAPER = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER / "rt3t" / "data"
EVENTS_DIR = RT3T / "events"
FMRIPREP = PAPER / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005"
PAR_DIR = PAPER / "cesar_local_derivatives/local_derivatives_ses_3/motion_corrected"
PREREG = PAPER / "task_2_1_betas" / "prereg"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
PST = 5
N_TR_PER_RUN = 192


def load_finalmask_idx() -> np.ndarray:
    """Indices into flat 3D BOLD that map to the 2792 MindEye finalmask voxels."""
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    return np.where(final_mask.flatten())[0][relmask]


def load_per_run_motion(run: int) -> np.ndarray:
    """Stack per-TR .par files into (T_run, 6) motion params."""
    pars = sorted(PAR_DIR.glob(
        f"sub-005_{SESSION}_task-C_run-{run:02d}_bold_*_mc.par"
    ))
    if not pars:
        return np.zeros((N_TR_PER_RUN, 6), dtype=np.float32)
    rows = [np.loadtxt(p) for p in pars]
    return np.stack(rows, axis=0).astype(np.float32)


def fit_lss_one(bold_4d: nib.Nifti1Image, events: pd.DataFrame,
                 probe_trial: int, mc_params: np.ndarray,
                 decode_TR: int) -> np.ndarray | None:
    """One LSS fit using nilearn — crops BOLD to [:decode_TR+1] and fits
    `FirstLevelModel(noise_model='ar1', hrf_model='glover', drift_model='cosine',
    drift_order=1, high_pass=0.01, signal_scaling=False)` with mc_params
    as confounds. Returns (X, Y, Z) volume of probe-contrast effect."""
    from nilearn.glm.first_level import FirstLevelModel

    base = events.copy()
    base["onset"] = base["onset"].astype(float) - base["onset"].iloc[0]

    decode_sec = decode_TR * TR
    base = base[base["onset"] <= decode_sec].reset_index(drop=True)
    if probe_trial >= len(base):
        return None

    bold_arr = bold_4d.get_fdata()[..., :decode_TR + 1]
    bold_used = nib.Nifti1Image(bold_arr, bold_4d.affine)
    mc_used = mc_params[:decode_TR + 1] if mc_params is not None else None

    base["trial_type"] = np.where(
        np.arange(len(base)) == probe_trial, "probe", "reference"
    )
    base["duration"] = 1.0

    confounds = (pd.DataFrame(mc_used,
                              columns=[f"mc_{i}" for i in range(mc_used.shape[1])])
                 if mc_used is not None else None)

    glm = FirstLevelModel(
        t_r=TR, slice_time_ref=0,
        hrf_model="glover",
        drift_model="cosine", drift_order=1, high_pass=0.01,
        signal_scaling=False, smoothing_fwhm=None,
        noise_model="ar1",
        n_jobs=1, verbose=0,
        memory_level=0, minimize_memory=True,
    )
    glm.fit(bold_used, events=base[["onset", "duration", "trial_type"]],
            confounds=confounds)
    return glm.compute_contrast("probe", output_type="effect_size").get_fdata()


def main():
    fmask = load_finalmask_idx()
    print(f"finalmask: {len(fmask)} voxels", flush=True)

    out_betas = []
    out_ids = []

    for run in RUNS:
        # Load BOLD + motion
        bold_path = (FMRIPREP / SESSION / "func"
                     / f"sub-005_{SESSION}_task-C_run-{run:02d}"
                     f"_space-T1w_desc-preproc_bold.nii.gz")
        bold_4d = nib.load(bold_path)
        events = pd.read_csv(EVENTS_DIR /
                             f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                             sep="\t")
        # Mac semantics: keep blank rows; turn NaN image_name into "blank.jpg"
        events["image_name"] = events["image_name"].fillna("blank.jpg")
        mc_params = load_per_run_motion(run)
        run_start = float(events["onset"].iloc[0])

        n_run_trials = len(events)
        print(f"\nrun-{run:02d}: {n_run_trials} events  "
              f"BOLD={bold_4d.shape}  mc={mc_params.shape}", flush=True)

        for trial_i in range(n_run_trials):
            t0 = time.time()
            onset_sec = float(events.iloc[trial_i]["onset"]) - run_start
            onset_TR = int(round(onset_sec / TR))
            decode_TR = min(onset_TR + PST, N_TR_PER_RUN - 1)

            beta_vol = fit_lss_one(bold_4d, events, trial_i, mc_params, decode_TR)
            if beta_vol is None:
                print(f"  run-{run:02d} trial {trial_i:3d}: SKIP "
                      f"(probe not visible at decode_TR={decode_TR})", flush=True)
                continue
            beta_masked = beta_vol.flatten()[fmask].astype(np.float32)
            out_betas.append(beta_masked)
            out_ids.append(str(events.iloc[trial_i]["image_name"]))
            if (trial_i + 1) % 10 == 0 or trial_i == 0:
                print(f"  run-{run:02d} trial {trial_i:3d}: "
                      f"decode_TR={decode_TR} ({time.time() - t0:.2f}s)",
                      flush=True)

    betas = np.stack(out_betas, axis=0)
    ids = np.asarray(out_ids)
    PREREG.mkdir(parents=True, exist_ok=True)
    cell = "RT_paper_Fast_pst5_lss_raw_kbm"
    np.save(PREREG / f"{cell}_{SESSION}_betas.npy", betas)
    np.save(PREREG / f"{cell}_{SESSION}_trial_ids.npy", ids)
    cfg = {"method": "per-trial nilearn LSS at pst=5 (Mac mindeye.py:745 verbatim)",
           "n_trials": int(betas.shape[0]),
           "session": SESSION,
           "keep_blanks": True,
           "pst": PST,
           "hrf": "glover",
           "noise_model": "ar1",
           "drift_model": "cosine",
           "drift_order": 1,
           "high_pass": 0.01,
           "z": "raw (apply at retrieval)"}
    with open(PREREG / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nsaved {cell}: betas {betas.shape}, ids {ids.shape}", flush=True)


if __name__ == "__main__":
    main()
