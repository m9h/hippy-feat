#!/usr/bin/env python3
"""LSS baseline matching streaming RLS GLM nuisance.

For each trial i, fit per-trial LSS (probe vs all-others contrast) on the same
truncated BOLD window used by the streaming RLS GLM, with the IDENTICAL 35
nuisance regressors (11 per-run intercepts + 11 per-run cosine drifts +
6 motion + 7 aCompCor K=7 from CSF∪WM eroded×1, HP-filter 0.01 Hz).

Difference vs streaming RLS GLM: design has only 2 trial cols (probe trial
+ sum-of-other-trials lump), instead of i+1 individual trial regressors.

Saves cells `RT_paper_LSS_K7CSFWM_HP_e1_{Fast_pst5,Slow_pst20,EoR}` so the
RLS-vs-LSS delta isolates streaming-vs-per-trial as the only variable.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse loaders + design helpers from the RLS driver
sys.path.insert(0, str(Path(__file__).parent))
from run_streaming_rls_glm import (  # type: ignore  # noqa: E402
    PREREG, RUNS, SESSION, TR, N_TR_PER_RUN,
    build_trial_metadata, causal_inclusive_zscore, load_finalmask_idx,
    load_per_run_motion, load_per_run_acompcor, load_bold_2792,
    make_concat_nuisance, make_glover_hrf, save_cell,
)


def lss_betas(bold: np.ndarray, nuisance: np.ndarray,
               trials: list[dict], pst: int | None,
               hrf: np.ndarray, T_total: int) -> np.ndarray:
    """Per-trial LSS: design = (probe, lump_other_trials, nuisance...)."""
    V = bold.shape[1]
    n_trials = len(trials)
    n_nuis = nuisance.shape[1]
    out = np.zeros((n_trials, V), dtype=np.float32)

    # Pre-build all trial impulse regressors (HRF-conv'd at global onset)
    n_hrf = len(hrf)
    trial_regs_full = np.zeros((T_total, n_trials), dtype=np.float32)
    for j, t in enumerate(trials):
        on = t["global_onset_tr"]
        if on < T_total:
            trial_regs_full[on, j] = 1.0
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
        # Probe = trial i regressor only; lump = sum of all other trials' regressors
        probe = trial_regs_full[:n_used, i: i + 1]
        if n_trials > 1:
            others = trial_regs_full[:n_used, :].sum(axis=1, keepdims=True) - probe
        else:
            others = np.zeros((n_used, 1), dtype=np.float32)
        X_nuis = nuisance[:n_used]
        X = np.concatenate([probe, others, X_nuis], axis=1)                  # (n_used, 2 + 35)
        y = bold[:n_used]
        K = X.shape[1]
        XtX = X.T @ X
        tr_xtx = float(np.trace(XtX))
        # Light ridge for numerical stability (LSS is well-determined here)
        lam = 1e-8 * tr_xtx / K
        XtX += lam * np.eye(K, dtype=X.dtype)
        Xty = X.T @ y
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
        except np.linalg.LinAlgError:
            beta = np.linalg.solve(XtX, Xty)
        # β for the probe column is at index 0
        out[i] = beta[0]

        if (i + 1) % 100 == 0 or i == 0:
            print(f"    trial {i + 1:>3}/{n_trials}  X={X.shape}  "
                   f"({time.time() - t0:.0f}s elapsed)", flush=True)

    return out


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
    motion_per_run = [load_per_run_motion(r) for r in RUNS]
    acomp_per_run = [load_per_run_acompcor(r) for r in RUNS]
    motion = np.concatenate(motion_per_run, axis=0).astype(np.float32)
    acomp = np.concatenate(acomp_per_run, axis=0).astype(np.float32)
    T_total = motion.shape[0]

    print("[2] loading BOLD per run + concatenating")
    t0 = time.time()
    bold = np.concatenate(
        [load_bold_2792(r, fmask) for r in RUNS], axis=0
    ).astype(np.float32)
    print(f"  bold: {bold.shape} in {time.time() - t0:.0f}s")

    nuisance = make_concat_nuisance(motion, acomp)
    n_hrf = int(np.ceil(32.0 / TR))
    hrf = make_glover_hrf(TR, n_hrf)

    pst_map = {"Fast": 5, "Slow": 20, "EoR": None}
    for tier in args.tiers:
        pst = pst_map[tier]
        print(f"\n[3] LSS-with-aCompCor tier={tier}  pst={pst}")
        betas = lss_betas(bold, nuisance, trials, pst, hrf, T_total)
        cell_name = (f"RT_paper_LSS_{tier}"
                     f"{'_pst5' if tier == 'Fast' else '_pst20' if tier == 'Slow' else ''}"
                     f"_K7CSFWM_HP_e1_raw")
        save_cell(cell_name, betas, image_ids,
                   {"tier": tier, "pst": pst, "n_trials": n_trials,
                    "nuisance_shape": list(nuisance.shape),
                    "hrf": "glover", "z": "raw (apply at retrieval)",
                    "design": "LSS (probe + lump others) + 35 nuisance"})


if __name__ == "__main__":
    main()
