#!/usr/bin/env python3
"""Cell 16 — TRULY online EKF: streaming Kalman state accumulates across
trials, not reset per trial. The previous EKF cell (13) re-instantiated
the state every trial, which defeats the entire point of having a
streaming filter.

Architecture:

    1. Concatenate all 11 runs of ses-03 BOLD into one (V, T_total) array
    2. Build a session-level design matrix where EVERY trial has its own
       probe regressor: shape (T_total, n_trials + drift)
    3. Initialize streaming_kalman_ar1 ONCE
    4. Walk through every TR sequentially, updating the filter
    5. At each trial's "decode time" (onset_TR + post_stim_window_TRs),
       snapshot state.beta_mean[probe_col_for_that_trial]

Late trials therefore see all TRs from t=0 up to their decode time —
strictly more evidence than early trials. If the streaming Kalman is
exploiting accumulated within-session data, the per-trial β quality
should improve monotonically with trial position. The earlier
β-reliability bucketing analysis (early vs mid vs late thirds of
ses-03) showed a +25-90% relative gain across the session purely from
cumulative z-score adaptation; if the online EKF works correctly, it
should add ~similar or larger gain at the GLM level.

Memory note: V × P × P covariance is the tight constraint. For V=2792
and P=770 (all trials as probe columns + drift) this is ~6.6 GB float32
— too big. We use a DIAGONAL-only covariance variant: V × P float32
= ~8.6 MB. Loses cross-coefficient correlations in the posterior; for
LSS designs where the probe regressors are nearly orthogonal anyway,
this is a reasonable approximation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prereg_variant_sweep import (
    PAPER_ROOT,
    RT3T,
    BRAIN_MASK,
    RELMASK,
    MC_DIR,
    OUT_DIR,
    EVENTS_DIR,
    load_mask,
    load_rtmotion,
    load_events,
)
from rt_glm_variants import make_glover_hrf


@jax.jit
def diag_kalman_update(beta_mean, beta_var, a_post, b_post,
                        x_row, y_obs):
    """Diagonal-covariance streaming Bayesian update.

    beta_mean: (V, P)        running posterior mean
    beta_var:  (V, P)        diagonal posterior variance (no off-diagonals)
    a_post:    (V,)          IG shape
    b_post:    (V,)          IG scale
    x_row:     (P,)          design row at this TR
    y_obs:     (V,)          observed BOLD across voxels at this TR

    The full P×P covariance update is replaced by a per-coefficient diagonal
    update. Equivalent under the assumption that probe regressors are
    near-orthogonal, which holds for LSS impulse designs separated in time.
    """
    # Diagonal Kalman: per-voxel, per-coefficient
    Sx = beta_var * x_row[None, :]                    # (V, P)
    xSx = jnp.einsum("vp,p->v", Sx, x_row)            # (V,)
    S = 1.0 + xSx                                      # (V,)
    K = Sx / S[:, None]                                # (V, P)
    innovation = y_obs - jnp.einsum("vp,p->v", beta_mean, x_row)
    beta_mean_new = beta_mean + K * innovation[:, None]
    beta_var_new = beta_var * (1.0 - x_row[None, :] * K)
    rss_inc = innovation ** 2 / S
    a_new = a_post + 0.5
    b_new = b_post + 0.5 * rss_inc
    return beta_mean_new, beta_var_new, a_new, b_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--runs", nargs="+", type=int, default=list(range(1, 12)))
    ap.add_argument("--post-stim-tr", type=int, default=8,
                    help="TRs after each trial's onset to wait before "
                         "snapshotting that trial's β posterior")
    ap.add_argument("--out-cell-name", default="EKF_session_online_glover_rtm")
    args = ap.parse_args()

    flat_brain, rel = load_mask()
    tr = 1.5
    hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))

    # ---- Load + concatenate all runs of ses-03 ----
    print(f"[1/4] Loading {len(args.runs)} runs of {args.session}", flush=True)
    timeseries_per_run = []
    onsets_per_run = []
    images_per_run = []
    run_lengths = []
    for run in args.runs:
        ts = load_rtmotion(args.session, run, flat_brain, rel)   # (V, T_r)
        events = load_events(args.session, run)
        timeseries_per_run.append(ts)
        onsets_per_run.append(events["onset_rel"].values.astype(np.float32))
        images_per_run.append(events["image_name"].astype(str).tolist())
        run_lengths.append(ts.shape[1])
    Y_session = np.concatenate(timeseries_per_run, axis=1).astype(np.float32)  # (V, T_total)
    V, T_total = Y_session.shape
    print(f"  V={V}  T_total={T_total}", flush=True)

    # ---- Build session-level design matrix ----
    # For each trial in each run, build a probe-impulse regressor that's
    # convolved with HRF, placed at the trial's run-relative TR offset
    # by the run's TR start. n_trials_total ≈ 770. Plus session-level
    # cosine drift (1 col) and intercept (1 col) → P ≈ 772.
    print(f"[2/4] Building session design matrix", flush=True)
    n_hrf_trs = len(hrf)
    run_starts_TR = np.concatenate([[0], np.cumsum(run_lengths)[:-1]])
    all_trial_meta = []   # list of (abs_onset_TR, decode_TR, image_name)
    n_trials = 0
    for run_idx, (onsets, images) in enumerate(zip(onsets_per_run, images_per_run)):
        for trial_i, onset_sec in enumerate(onsets):
            onset_TR = int(round(onset_sec / tr)) + int(run_starts_TR[run_idx])
            decode_TR = min(onset_TR + args.post_stim_tr, T_total - 1)
            all_trial_meta.append((onset_TR, decode_TR, images[trial_i]))
            n_trials += 1
    P = n_trials + 2                                  # +1 drift, +1 intercept
    print(f"  n_trials={n_trials}  P={P}", flush=True)

    X_session = np.zeros((T_total, P), dtype=np.float32)
    for trial_idx, (onset_TR, _, _) in enumerate(all_trial_meta):
        # Probe regressor: impulse at onset_TR convolved with HRF
        probe = np.zeros(T_total, dtype=np.float32)
        if 0 <= onset_TR < T_total:
            probe[onset_TR] = 1.0
        X_session[:, trial_idx] = np.convolve(probe, hrf)[:T_total]
    # Drift + intercept
    X_session[:, -2] = np.cos(2 * np.pi * np.arange(T_total) / max(T_total - 1, 1))
    X_session[:, -1] = 1.0

    # ---- Single streaming pass through every TR ----
    print(f"[3/4] Streaming Kalman over {T_total} TRs (diagonal-cov, "
          f"V×P = {V}×{P})", flush=True)
    # Init prior: weak. Diagonal cov = prior_var * 1
    prior_var = 1e3
    beta_mean = jnp.zeros((V, P), dtype=jnp.float32)
    beta_var = jnp.full((V, P), prior_var, dtype=jnp.float32)
    a_post = jnp.full((V,), 0.01, dtype=jnp.float32)
    b_post = jnp.full((V,), 0.01, dtype=jnp.float32)

    # Snapshot β at each trial's decode TR
    decode_lookup = {}                                # decode_TR → list of trial_idxs
    for trial_idx, (_, decode_TR, _) in enumerate(all_trial_meta):
        decode_lookup.setdefault(decode_TR, []).append(trial_idx)

    betas_at_decode = np.zeros((n_trials, V), dtype=np.float32)
    t0 = time.time()
    for t in range(T_total):
        beta_mean, beta_var, a_post, b_post = diag_kalman_update(
            beta_mean, beta_var, a_post, b_post,
            jnp.asarray(X_session[t]), jnp.asarray(Y_session[:, t]),
        )
        if t in decode_lookup:
            for trial_idx in decode_lookup[t]:
                betas_at_decode[trial_idx] = np.asarray(
                    beta_mean[:, trial_idx], dtype=np.float32
                )
        if (t + 1) % 200 == 0:
            print(f"  TR {t+1}/{T_total}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"  streaming pass complete in {time.time()-t0:.1f}s", flush=True)

    # ---- Save in the standard prereg cell layout ----
    print(f"[4/4] Saving cell {args.out_cell_name}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{args.out_cell_name}_{args.session}_betas.npy",
            betas_at_decode)
    trial_ids = np.asarray([m[2] for m in all_trial_meta])
    np.save(OUT_DIR / f"{args.out_cell_name}_{args.session}_trial_ids.npy",
            trial_ids)
    config = {
        "cell": args.out_cell_name,
        "session": args.session,
        "runs": args.runs,
        "P": P,
        "n_trials": n_trials,
        "post_stim_tr": args.post_stim_tr,
        "method": "online streaming Kalman, diagonal cov, accumulating across trials",
    }
    (OUT_DIR / f"{args.out_cell_name}_{args.session}_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"  saved {args.out_cell_name}: betas {betas_at_decode.shape}",
          flush=True)


if __name__ == "__main__":
    main()
