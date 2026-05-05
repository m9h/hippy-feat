#!/usr/bin/env python3
"""Streaming RLS GLM (Mac champion replication: results/apple_silicon_2026-04-28/STREAMING_RLS_GLM.md).

For each trial i in chronological order across the full ses-03 session:
  1. decode_TR_i = global_onset_TR_i + pst (Fast=5, Slow=20) or run-end (EoR)
  2. Build truncated design at decode_TR_i:
       - 11 per-run intercepts
       - 11 per-run cosine drifts (order 1)
       - 6 continuous motion regressors (MCFLIRT .par, stacked across runs)
       - 7 aCompCor PCs (CSF∪WM eroded×1, HP-filtered at 0.01 Hz, per-run, stacked)
       - i HRF-convolved Glover trial regressors (one per trial seen 1..i)
  3. Truncate BOLD: y[:decode_TR_i, :2792]
  4. Solve ridge: β = (XᵀX + λI)⁻¹ Xᵀy with λ = 1e-3 · tr(XᵀX) / K (or 1e-2 if n < K)
  5. Extract β_i from the i-th trial column

Apply inclusive cum-z to (770, 2792) β series, save as prereg cells:
  RT_paper_RLS_Fast_pst5_K7CSFWM_HP_e1_inclz
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz
  RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

PAPER = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER / "rt3t" / "data"
EVENTS_DIR = RT3T / "events"
FMRIPREP = (PAPER / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005")
ACOMPCOR_DIR = PAPER / "task_2_1_betas" / "acompcor"
PAR_DIR = PAPER / "cesar_local_derivatives/local_derivatives_ses_3/motion_corrected"
PREREG = PAPER / "task_2_1_betas" / "prereg"
SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
N_TR_PER_RUN = 192


def make_glover_hrf(tr: float, n_trs: int) -> np.ndarray:
    from nilearn.glm.first_level import glover_hrf
    return glover_hrf(tr, oversampling=1, time_length=n_trs * tr)[:n_trs]


def load_finalmask_idx() -> np.ndarray:
    """Indices into flat 3D BOLD that map to 2792 MindEye finalmask voxels."""
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    return np.where(final_mask.flatten())[0][relmask]


def load_per_run_motion(run: int) -> np.ndarray:
    """Stack per-TR .par files into (T_run, 6) motion params."""
    pars = sorted(PAR_DIR.glob(
        f"sub-005_{SESSION}_task-C_run-{run:02d}_bold_*_mc.par"
    ))
    if not pars:
        raise FileNotFoundError(f"no .par files for run {run}")
    rows = [np.loadtxt(p) for p in pars]
    return np.stack(rows, axis=0).astype(np.float32)                    # (T_run, 6)


def load_per_run_acompcor(run: int) -> np.ndarray:
    p = ACOMPCOR_DIR / f"sub-005_{SESSION}_run-{run:02d}_acompcor_K7_csfwm_hp_e1.npy"
    return np.load(p).astype(np.float32)                                 # (T_run, 7)


def load_bold_2792(run: int, fmask_idx: np.ndarray) -> np.ndarray:
    p = (FMRIPREP / SESSION / "func"
         / f"sub-005_{SESSION}_task-C_run-{run:02d}"
           f"_space-T1w_desc-preproc_bold.nii.gz")
    arr = nib.load(p).get_fdata().astype(np.float32)                     # (X, Y, Z, T)
    T = arr.shape[-1]
    return arr.reshape(-1, T)[fmask_idx].T.astype(np.float32)             # (T, 2792)


def build_trial_metadata() -> tuple[list[dict], np.ndarray]:
    """Per-trial: run, onset_within_run_TR, global_onset_TR, image_name.

    Mac uses 770 trials = all non-blank events across 11 runs (incl. unchosen NSD).
    """
    trials = []
    images = []
    for r_idx, run in enumerate(RUNS):
        e = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                          sep="\t")
        # events.tsv onsets are session-cumulative seconds — subtract run's
        # first event onset to get within-run seconds.
        run_t0 = float(e["onset"].iloc[0])
        e = e[e["image_name"] != "blank.jpg"].reset_index(drop=True)
        for _, row in e.iterrows():
            onset_tr = int(round((float(row["onset"]) - run_t0) / TR))
            onset_tr = min(onset_tr, N_TR_PER_RUN - 1)
            global_tr = r_idx * N_TR_PER_RUN + onset_tr
            trials.append({
                "run_idx": r_idx, "run": run,
                "onset_tr_within_run": onset_tr,
                "global_onset_tr": global_tr,
                "image_name": str(row["image_name"]),
            })
            images.append(str(row["image_name"]))
    return trials, np.asarray(images)


def make_concat_nuisance(motion: np.ndarray, acomp: np.ndarray
                         ) -> np.ndarray:
    """Stack per-run nuisance into one (T_total, 35) design block:
       11 intercepts | 11 drifts | 6 motion (already concat) | 7 aCompCor.
    """
    n_runs = len(RUNS)
    T_total = n_runs * N_TR_PER_RUN
    intercept = np.zeros((T_total, n_runs), dtype=np.float32)
    drift = np.zeros((T_total, n_runs), dtype=np.float32)
    for r in range(n_runs):
        s = r * N_TR_PER_RUN
        e = s + N_TR_PER_RUN
        intercept[s:e, r] = 1.0
        # cosine drift order 1: cos(pi * (2t+1) / (2*N))
        t = np.arange(N_TR_PER_RUN)
        drift[s:e, r] = np.cos(np.pi * (2 * t + 1) / (2 * N_TR_PER_RUN))
    return np.concatenate([intercept, drift, motion, acomp], axis=1)


def streaming_rls_betas(bold: np.ndarray, nuisance: np.ndarray,
                          trials: list[dict], pst: int | None,
                          hrf: np.ndarray, T_total: int,
                          ) -> np.ndarray:
    """Run the streaming RLS GLM. Returns (n_trials, 2792) β series.

    pst: if None, use run-end as decode point (EoR). Otherwise onset+pst (Fast=5, Slow=20).
    """
    V = bold.shape[1]
    n_trials = len(trials)
    n_nuis = nuisance.shape[1]
    out_betas = np.zeros((n_trials, V), dtype=np.float32)

    # Pre-build all trial regressors lazily (full T_total length each, will truncate later)
    # Each trial j contributes a regressor: probe at global_onset_tr_j convolved with HRF.
    n_hrf = len(hrf)
    trial_regs_full = np.zeros((T_total, n_trials), dtype=np.float32)
    for j, t in enumerate(trials):
        on = t["global_onset_tr"]
        if on < T_total:
            trial_regs_full[on, j] = 1.0
    # Convolve column-by-column
    for j in range(n_trials):
        trial_regs_full[:, j] = np.convolve(trial_regs_full[:, j], hrf)[:T_total]

    t0 = time.time()
    for i, trial in enumerate(trials):
        run_end_tr = (trial["run_idx"] + 1) * N_TR_PER_RUN
        if pst is None:
            decode_tr = run_end_tr - 1
        else:
            decode_tr = trial["global_onset_tr"] + pst
            decode_tr = min(decode_tr, T_total - 1)

        n_used = decode_tr + 1
        # Trial cols 0..i (inclusive)
        X_trials = trial_regs_full[:n_used, : i + 1]                     # (n_used, i+1)
        X_nuis = nuisance[:n_used]                                        # (n_used, 35)
        X = np.concatenate([X_nuis, X_trials], axis=1)                   # (n_used, 35+i+1)
        y = bold[:n_used]                                                 # (n_used, V)

        K = X.shape[1]
        XtX = X.T @ X                                                     # (K, K)
        # Ridge λ — minimal regularization just for numerical stability;
        # using stronger λ (like 1e-3·tr/K) crushes per-trial signal and
        # leaves all βs sharing a common ridge-shrunk mean pattern.
        tr_xtx = float(np.trace(XtX))
        if n_used < K:
            lam = 1e-3 * tr_xtx / K
        else:
            lam = 1e-8 * tr_xtx / K
        XtX += lam * np.eye(K, dtype=X.dtype)

        # Solve. Use cholesky for SPD ridge matrix.
        Xty = X.T @ y                                                     # (K, V)
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
        except np.linalg.LinAlgError:
            beta = np.linalg.solve(XtX, Xty)
        # Extract this trial's β: it's the last trial col → column n_nuis + i
        out_betas[i] = beta[n_nuis + i]

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    trial {i + 1:>3}/{n_trials}  X={X.shape}  "
                   f"({time.time() - t0:.0f}s elapsed)", flush=True)

    return out_betas


def causal_inclusive_zscore(arr: np.ndarray) -> np.ndarray:
    """Causal cum-z (exclusive of current trial): z_i uses [0..i-1] stats.

    Matches `import_canonical_glmsingle.causal_cumulative_zscore` and produces
    well-conditioned βs with across-trial std ~ 1. The 'inclusive' variant
    (using [0..i] stats) is unstable for early trials where variance estimates
    collapse to epsilon, neutralizing signal.
    """
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[0]
    for i in range(n):
        if i == 0:
            out[i] = arr[i]
        elif i == 1:
            out[i] = arr[i] - arr[0]
        else:
            mu = arr[:i].mean(0, keepdims=True)
            sd = arr[:i].std(0, keepdims=True) + 1e-6
            out[i] = (arr[i] - mu) / sd
    return out


def save_cell(name: str, betas: np.ndarray, ids: np.ndarray, cfg: dict) -> None:
    PREREG.mkdir(parents=True, exist_ok=True)
    np.save(PREREG / f"{name}_{SESSION}_betas.npy", betas)
    np.save(PREREG / f"{name}_{SESSION}_trial_ids.npy", ids)
    import json
    with open(PREREG / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: betas {betas.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["Fast", "Slow", "EoR"],
                     choices=["Fast", "Slow", "EoR"])
    args = ap.parse_args()

    print("[1] loading metadata + nuisance regressors")
    trials, image_ids = build_trial_metadata()
    n_trials = len(trials)
    print(f"  {n_trials} non-blank trials over {len(RUNS)} runs")

    fmask = load_finalmask_idx()
    print(f"  finalmask: {len(fmask)} voxels")

    motion_per_run = [load_per_run_motion(r) for r in RUNS]
    acomp_per_run = [load_per_run_acompcor(r) for r in RUNS]
    motion = np.concatenate(motion_per_run, axis=0).astype(np.float32)   # (T_total, 6)
    acomp = np.concatenate(acomp_per_run, axis=0).astype(np.float32)     # (T_total, 7)
    T_total = motion.shape[0]
    print(f"  motion: {motion.shape}  acompcor: {acomp.shape}  T_total={T_total}")

    print("[2] loading BOLD per run + concatenating")
    t0 = time.time()
    bold = np.concatenate(
        [load_bold_2792(r, fmask) for r in RUNS], axis=0
    ).astype(np.float32)                                                 # (T_total, 2792)
    print(f"  bold: {bold.shape}  loaded in {time.time() - t0:.0f}s")

    nuisance = make_concat_nuisance(motion, acomp)
    print(f"  nuisance: {nuisance.shape} (11 intercept + 11 drift + 6 motion + 7 acomp = 35)")

    n_hrf = int(np.ceil(32.0 / TR))
    hrf = make_glover_hrf(TR, n_hrf)

    pst_map = {"Fast": 5, "Slow": 20, "EoR": None}
    for tier in args.tiers:
        pst = pst_map[tier]
        print(f"\n[3] Streaming RLS GLM tier={tier}  pst={pst}")
        betas = streaming_rls_betas(bold, nuisance, trials, pst, hrf, T_total)
        # Save RAW βs — let retrieval pass apply its standard cum-z
        # (matches the canonical pipeline's `cumulative_zscore` utility).
        cell_name = (f"RT_paper_RLS_{tier}"
                     f"{'_pst5' if tier == 'Fast' else '_pst20' if tier == 'Slow' else ''}"
                     f"_K7CSFWM_HP_e1_raw")
        save_cell(cell_name, betas, image_ids,
                   {"tier": tier, "pst": pst, "n_trials": n_trials,
                    "nuisance_shape": list(nuisance.shape),
                    "hrf": "glover", "z": "raw (apply at retrieval)"})


if __name__ == "__main__":
    main()
