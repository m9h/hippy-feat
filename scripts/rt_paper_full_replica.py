#!/usr/bin/env python3
"""Cells 10–12 of the pre-registered variant sweep — faithful nilearn replicas.

Reproduces the paper's actual RT and Offline pipelines as closely as
possible, by calling nilearn's `FirstLevelModel` with the exact arguments
mindeye.py:745 uses, then layering the cumulative running z-score and
optional repeat-averaging that the paper does outside the GLM.

Cells:
    10. RT_paper_replica_partial      — nilearn AR(1) + cumulative z-score, NO repeat-avg
    11. RT_paper_replica_full         — nilearn AR(1) + cumulative z-score + repeat-avg (canonical)
    12. Offline_paper_replica_full    — same as 11 but on fmriprep BOLD instead of rtmotion

Per-trial fits are slow (nilearn FirstLevelModel.fit per probe trial).
Expect ~30 min/cell on CPU; this script is offline-only. Submitted via
`scripts/rt_paper_full_replica.sbatch`.
"""
from __future__ import annotations

import argparse
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


PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
FMRIPREP_ROOT = (PAPER_ROOT / "fmriprep_mindeye" / "data_sub-005"
                 / "bids" / "derivatives" / "fmriprep" / "sub-005")
EVENTS_DIR = RT3T / "events"
BRAIN_MASK = RT3T / "sub-005_final_mask.nii.gz"
RELMASK = RT3T / "sub-005_ses-01_task-C_relmask.npy"
MC_DIR = Path("/data/3t/derivatives/motion_corrected_resampled")
OUT_DIR = PAPER_ROOT / "task_2_1_betas" / "prereg"


def load_brain_paper_mask() -> tuple[np.ndarray, np.ndarray]:
    flat_brain = (nib.load(BRAIN_MASK).get_fdata() > 0).flatten()
    rel = np.load(RELMASK)
    return flat_brain, rel


def load_rtmotion_4d(session: str, run: int) -> nib.Nifti1Image:
    pattern = f"{session}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    if not vols:
        raise FileNotFoundError(f"no mc_boldres for {session} run-{run:02d}")
    frames = [nib.load(v).get_fdata().astype(np.float32) for v in vols]
    arr = np.stack(frames, axis=-1)                        # (X, Y, Z, T)
    affine = nib.load(vols[0]).affine
    return nib.Nifti1Image(arr, affine)


def load_fmriprep_4d(session: str, run: int) -> nib.Nifti1Image:
    p = (FMRIPREP_ROOT / session / "func"
         / f"sub-005_{session}_task-C_run-{run:02d}"
           f"_space-T1w_desc-preproc_bold.nii.gz")
    return nib.load(p)


def load_mc_params(session: str, run: int) -> np.ndarray | None:
    """MCFLIRT motion params (.par files); None if not on disk."""
    par_dir = Path("/data/3t/derivatives/motion_corrected_resampled_par")
    par_path = par_dir / f"{session}_run-{run:02d}.par"
    if par_path.exists():
        return np.loadtxt(par_path).astype(np.float32)
    return None


def fit_lss_nilearn(bold_4d: nib.Nifti1Image, events: pd.DataFrame,
                     probe_trial: int, mc_params: np.ndarray | None,
                     tr: float = 1.5, mask_img: nib.Nifti1Image | None = None,
                     streaming_decode_TR: int | None = None,
                     ) -> np.ndarray | None:
    """One LSS fit using nilearn — replicates mindeye.py:745.

    If `streaming_decode_TR` is given, crop BOLD to volumes [0..decode_TR]
    and events to `events.onset <= decode_TR * tr` BEFORE fitting. This
    matches Rishab's notebook cell 19: at every non-blank TR the GLM is
    refit on the BOLD-and-events accumulated up to that TR. Cumulative
    motion params are also cropped if provided. Returns None if the probe
    trial isn't yet visible at the cropped window.
    """
    from nilearn.glm.first_level import FirstLevelModel

    base = events.copy()
    base["onset"] = base["onset"].astype(float) - base["onset"].iloc[0]

    if streaming_decode_TR is not None:
        decode_sec = streaming_decode_TR * tr
        base = base[base["onset"] <= decode_sec].reset_index(drop=True)
        if probe_trial >= len(base):
            return None
        bold_arr = bold_4d.get_fdata()[..., :streaming_decode_TR + 1]
        bold_used = nib.Nifti1Image(bold_arr, bold_4d.affine)
        if mc_params is not None:
            mc_used = mc_params[:streaming_decode_TR + 1]
        else:
            mc_used = None
    else:
        bold_used = bold_4d
        mc_used = mc_params

    base["trial_type"] = np.where(
        np.arange(len(base)) == probe_trial, "probe", "reference"
    )
    base["duration"] = 1.0  # avoid nilearn's null-duration warning

    confounds = (pd.DataFrame(mc_used,
                              columns=[f"mc_{i}" for i in range(mc_used.shape[1])])
                 if mc_used is not None else None)

    glm = FirstLevelModel(
        t_r=tr, slice_time_ref=0,
        hrf_model="glover",
        drift_model="cosine", drift_order=1, high_pass=0.01,
        signal_scaling=False, smoothing_fwhm=None,
        noise_model="ar1",
        n_jobs=1, verbose=0,
        memory_level=0, minimize_memory=True,
        mask_img=mask_img if mask_img is not None else False,
    )
    glm.fit(run_imgs=bold_used, events=base, confounds=confounds)
    eff = glm.compute_contrast("probe", output_type="effect_size")
    return eff.get_fdata()                                 # (X, Y, Z) volume


def cumulative_zscore_with_optional_repeat_avg(
        beta_history: list[np.ndarray],
        image_history: list[str],
        do_repeat_avg: bool,
        ) -> tuple[np.ndarray, list[str]]:
    """Apply the paper's CAUSAL cumulative running z-score (and optional
    repeat-averaging) to the accumulated trial-betas.

    Replicates mindeye.py:770-784. Each trial i is z-scored using the
    statistics of trials 0..i-1 ONLY — strict causality, no test-set
    leakage.

    Earlier (buggy) version used `arr.mean(axis=0)` over ALL trials
    including future ones. Apple-Silicon side caught this: their v2
    retrieval eval moved cell 12 from 80% (leaky) to 76% (paper-exact).
    DGX cell 11 inflation by 8pp vs Mac was the same leak.

    Returns the post-processed (n_trials, V) beta matrix and the
    image label per (post-processed) trial — which may be shorter than
    the input if repeat_avg collapses repeats.
    """
    arr = np.stack(beta_history, axis=0).astype(np.float32)   # (n, V)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        if i < 2:
            # Too few past trials to z-score; just center on whatever's there
            if i == 0:
                z[i] = arr[i]                                # no past stats
            else:
                z[i] = arr[i] - arr[0]                       # center on first
        else:
            mu = arr[:i].mean(axis=0)
            sd = arr[:i].std(axis=0) + 1e-6
            z[i] = (arr[i] - mu) / sd

    if not do_repeat_avg:
        return z, list(image_history)

    seen: dict[str, list[int]] = {}
    out_betas, out_ids = [], []
    for i, img in enumerate(image_history):
        seen.setdefault(img, []).append(i)
        if len(seen[img]) == 1:
            out_betas.append(z[i])
            out_ids.append(img)
        else:
            avg = z[seen[img]].mean(axis=0)
            # Replace the first instance's entry with the running average
            first_pos = next(j for j, l in enumerate(out_ids) if l == img)
            out_betas[first_pos] = avg
    return np.stack(out_betas, axis=0), out_ids


def _ols_residuals_for_past_run(bold_4d: nib.Nifti1Image,
                                 events: pd.DataFrame,
                                 flat_brain: np.ndarray, rel: np.ndarray,
                                 tr: float) -> np.ndarray:
    """Cheap per-voxel OLS GLM with all events as task regressors + drift +
    intercept. Returns the masked residuals (V_brain, T_run) — pure noise
    after subtracting the canonical-Glover task fit. Used to build a
    task-orthogonal cross-run template."""
    arr_4d = bold_4d.get_fdata().astype(np.float32)
    X, Y, Z, T = arr_4d.shape
    Y_masked = arr_4d.reshape(-1, T)[flat_brain][rel]              # (V_brain, T)
    onsets_sec = events["onset"].astype(float).values - float(events["onset"].iloc[0])
    n_hrf_trs = int(np.ceil(32.0 / tr))
    from rt_glm_variants import make_glover_hrf
    hrf = make_glover_hrf(tr, n_hrf_trs)
    n_trials = len(onsets_sec)
    Xd = np.zeros((T, n_trials + 2), dtype=np.float32)
    for i, onset_sec in enumerate(onsets_sec):
        onset_TR = int(round(onset_sec / tr))
        probe = np.zeros(T, dtype=np.float32)
        if 0 <= onset_TR < T:
            probe[onset_TR] = 1.0
        Xd[:, i] = np.convolve(probe, hrf)[:T]
    Xd[:, -2] = np.cos(2 * np.pi * np.arange(T) / max(T - 1, 1))
    Xd[:, -1] = 1.0
    XtX_inv = np.linalg.inv(Xd.T @ Xd + 1e-4 * np.eye(Xd.shape[1])).astype(np.float32)
    beta = XtX_inv @ Xd.T @ Y_masked.T                              # (P, V)
    pred = Xd @ beta                                                # (T, V)
    resid = Y_masked.T - pred                                       # (T, V)
    return resid.astype(np.float32)


def run_cell(cell_name: str, bold_loader, session: str, runs: list[int],
              do_repeat_avg: bool,
              streaming_post_stim_TRs: int | None = None,
              cross_run_K: int | None = None,
              cross_run_residual_K: int | None = None,
              ) -> tuple[np.ndarray, list[str], dict]:
    """If `streaming_post_stim_TRs` is given, simulate paper RT pipeline:
    each trial's LSS GLM is fit on BOLD/events cropped to TR =
    onset_TR + post_stim. Otherwise fit on the full run (offline LSS).

    If `cross_run_K` is given, build a Regime C cross-run HOSVD template:
    for run r, take the top-K spatial PCs of concat(past runs 1..r-1)
    BOLD, project current run's BOLD onto them, and append the resulting
    K time series as additional nuisance regressors. Strictly causal —
    each run uses only earlier runs' data."""
    flat_brain, rel = load_brain_paper_mask()
    tr = 1.5
    mask_img = nib.load(BRAIN_MASK)

    beta_history: list[np.ndarray] = []
    image_history: list[str] = []
    config = {"cell": cell_name, "session": session, "runs": runs,
              "do_repeat_avg": do_repeat_avg, "tr": tr,
              "streaming_post_stim_TRs": streaming_post_stim_TRs,
              "cross_run_K": cross_run_K,
              "cross_run_residual_K": cross_run_residual_K,
              "nilearn_args": {
                  "hrf_model": "glover", "drift_model": "cosine",
                  "drift_order": 1, "high_pass": 0.01,
                  "noise_model": "ar1", "signal_scaling": False,
              }}

    # When using cross-run template, accumulate past-run BOLD as (V, T_r)
    # masked time series so that SVD is cheap. For the residual variant,
    # store residuals (T_r, V_brain) instead.
    past_runs_masked: list[np.ndarray] = []
    past_runs_residuals: list[np.ndarray] = []  # (T_r, V_brain) per past run

    for run_idx, run in enumerate(runs):
        try:
            bold_4d = bold_loader(session, run)
        except FileNotFoundError as e:
            print(f"  SKIP run-{run:02d}: {e}")
            continue
        events_path = EVENTS_DIR / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv"
        events = pd.read_csv(events_path, sep="\t")
        mc_params = load_mc_params(session, run)
        n_trs_run = bold_4d.shape[3] if hasattr(bold_4d, "shape") else None

        # Compute Regime C cross-run nuisance for this run before the
        # trial loop. Strictly causal: only past runs feed the template.
        cross_run_confounds = None
        if cross_run_K is not None and len(past_runs_masked) > 0:
            past_concat = np.concatenate(past_runs_masked, axis=1)  # (V_brain, T_past)
            past_centered = past_concat - past_concat.mean(axis=1, keepdims=True)
            U, _, _ = np.linalg.svd(past_centered.astype(np.float64),
                                      full_matrices=False)
            template = U[:, :cross_run_K].astype(np.float32)        # (V_brain, K)
            current_bold_4d = bold_4d.get_fdata().astype(np.float32)
            Xs, Ys, Zs, T_r = current_bold_4d.shape
            current_masked = current_bold_4d.reshape(-1, T_r)[flat_brain][rel]
            cross_run_confounds = (template.T @ current_masked).T   # (T_r, K)
            print(f"  [{cell_name}] run-{run:02d} cross-run RAW-BOLD template "
                  f"from {len(past_runs_masked)} past runs (V={template.shape[0]}, K={cross_run_K})")
        elif cross_run_residual_K is not None and len(past_runs_residuals) > 0:
            # Task-orthogonal: SVD of concatenated GLM residuals from past
            # runs. Spatial PCs (right singular vectors) capture noise
            # structure shared across sessions without the task subspace.
            past_resid = np.concatenate(past_runs_residuals, axis=0)   # (T_past, V_brain)
            past_centered = past_resid - past_resid.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(past_centered.astype(np.float64),
                                       full_matrices=False)
            template = Vt[:cross_run_residual_K].T.astype(np.float32)   # (V_brain, K)
            current_bold_4d = bold_4d.get_fdata().astype(np.float32)
            Xs, Ys, Zs, T_r = current_bold_4d.shape
            current_masked = current_bold_4d.reshape(-1, T_r)[flat_brain][rel]
            cross_run_confounds = (template.T @ current_masked).T       # (T_r, K)
            print(f"  [{cell_name}] run-{run:02d} cross-run RESIDUAL template "
                  f"from {len(past_runs_residuals)} past runs (V={template.shape[0]}, K={cross_run_residual_K})")

        # Stash this run's data for future runs' templates
        if cross_run_K is not None:
            current_bold_for_stash = bold_4d.get_fdata().astype(np.float32)
            Xs, Ys, Zs, T_r = current_bold_for_stash.shape
            past_runs_masked.append(
                current_bold_for_stash.reshape(-1, T_r)[flat_brain][rel]
            )
        if cross_run_residual_K is not None:
            past_runs_residuals.append(
                _ols_residuals_for_past_run(bold_4d, events, flat_brain, rel, tr)
            )

        # Combine motion + cross-run nuisance into a single confound table
        if cross_run_confounds is not None:
            if mc_params is not None:
                # Align T dimension (mc may be one row per TR; cross-run is T_r rows)
                T_mc = mc_params.shape[0]
                T_cr = cross_run_confounds.shape[0]
                T_use = min(T_mc, T_cr)
                combined = np.concatenate(
                    [mc_params[:T_use], cross_run_confounds[:T_use]], axis=1
                )
            else:
                combined = cross_run_confounds
            confounds_for_fit = combined
        else:
            confounds_for_fit = mc_params

        for trial_i in range(len(events)):
            t0 = time.time()
            decode_TR = None
            if streaming_post_stim_TRs is not None:
                onset_sec = float(events.iloc[trial_i]["onset"]) - float(events.iloc[0]["onset"])
                onset_TR = int(round(onset_sec / tr))
                decode_TR = onset_TR + streaming_post_stim_TRs
                if n_trs_run is not None:
                    decode_TR = min(decode_TR, n_trs_run - 1)
            beta_vol = fit_lss_nilearn(bold_4d, events, trial_i, confounds_for_fit,
                                        tr=tr, mask_img=mask_img,
                                        streaming_decode_TR=decode_TR)
            if beta_vol is None:
                print(f"  {cell_name} run-{run:02d} trial {trial_i:3d} "
                      f"SKIP (decode_TR not yet reached)")
                continue
            beta_masked = beta_vol.flatten()[flat_brain][rel]
            beta_history.append(beta_masked.astype(np.float32))
            image_history.append(str(events.iloc[trial_i].get("image_name",
                                                                str(trial_i))))
            tag = f"streaming(decode_TR={decode_TR})" if decode_TR is not None else "full-run"
            print(f"  {cell_name} run-{run:02d} trial {trial_i:3d} "
                  f"{tag} ({time.time() - t0:.2f}s)")

    betas, trial_ids = cumulative_zscore_with_optional_repeat_avg(
        beta_history, image_history, do_repeat_avg=do_repeat_avg,
    )
    config["n_trials_post"] = int(betas.shape[0])
    return betas, trial_ids, config


CELLS = {
    # Cells 10-12 — original full-run-BOLD anchors (Regime A in
    # TASK_2_1_AMENDMENT_2026-04-28.md). These mislabel as RT but actually
    # fit LSS on the full run; kept for the windowing decomposition.
    "RT_paper_replica_partial":   dict(loader=load_rtmotion_4d,
                                         do_repeat_avg=False,
                                         streaming_post_stim_TRs=None),
    "RT_paper_replica_full":      dict(loader=load_rtmotion_4d,
                                         do_repeat_avg=True,
                                         streaming_post_stim_TRs=None),
    "Offline_paper_replica_full": dict(loader=load_fmriprep_4d,
                                         do_repeat_avg=True,
                                         streaming_post_stim_TRs=None),
    # Regime B (within-run streaming) — the amendment's locked anchors.
    # Each per-trial GLM fits on BOLD/events cropped to onset_TR+pst.
    # pst=8 is the paper-RT replica per Mac's recovery of the 10pp gap.
    "RT_paper_replica_streaming_pst4_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=4),
    "RT_paper_replica_streaming_pst6_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=6),
    "RT_paper_replica_streaming_pst8_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=8),
    "RT_paper_replica_streaming_pst10_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=10),
    "RT_paper_replica_streaming_pst8_full":
        dict(loader=load_rtmotion_4d, do_repeat_avg=True,
             streaming_post_stim_TRs=8),
    # Regime C — cross-run HOSVD template confound (TASK_2_1_AMENDMENT
    # H3'). Each run's GLM gets K cross-run nuisance regressors built
    # causally from concatenated past runs' BOLD. The actual deliverable.
    "RT_streaming_pst8_HOSVD_K5_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=8, cross_run_K=5),
    "RT_streaming_pst8_HOSVD_K10_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=8, cross_run_K=10),
    "RT_streaming_pst8_HOSVD_K5_full":
        dict(loader=load_rtmotion_4d, do_repeat_avg=True,
             streaming_post_stim_TRs=8, cross_run_K=5),
    # H3'-corrected: task-residual HOSVD. Past-run BOLD has the GLM task
    # fit subtracted before SVD, so spatial PCs span session-shared noise
    # only — projecting them out shouldn't remove decode signal.
    "RT_streaming_pst8_ResidHOSVD_K5_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=8, cross_run_residual_K=5),
    "RT_streaming_pst8_ResidHOSVD_K10_partial":
        dict(loader=load_rtmotion_4d, do_repeat_avg=False,
             streaming_post_stim_TRs=8, cross_run_residual_K=10),
    "RT_streaming_pst8_ResidHOSVD_K5_full":
        dict(loader=load_rtmotion_4d, do_repeat_avg=True,
             streaming_post_stim_TRs=8, cross_run_residual_K=5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS.keys()))
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--runs", nargs="+", type=int, default=list(range(1, 12)))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cell in args.cells:
        if cell not in CELLS:
            print(f"  SKIP unknown cell {cell}")
            continue
        print(f"\n=== {cell} ===")
        cfg = CELLS[cell]
        try:
            betas, trial_ids, config = run_cell(
                cell, cfg["loader"], args.session, args.runs,
                do_repeat_avg=cfg["do_repeat_avg"],
                streaming_post_stim_TRs=cfg.get("streaming_post_stim_TRs"),
                cross_run_K=cfg.get("cross_run_K"),
                cross_run_residual_K=cfg.get("cross_run_residual_K"),
            )
            np.save(OUT_DIR / f"{cell}_{args.session}_betas.npy", betas)
            np.save(OUT_DIR / f"{cell}_{args.session}_trial_ids.npy",
                    np.asarray(trial_ids))
            with open(OUT_DIR / f"{cell}_{args.session}_config.json", "w") as f:
                json.dump(config, f, indent=2)
            print(f"  saved {cell}: betas {betas.shape}")
        except Exception as e:
            print(f"  FAILED {cell}: {e}")


if __name__ == "__main__":
    main()
