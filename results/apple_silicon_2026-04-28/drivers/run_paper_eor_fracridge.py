#!/usr/bin/env python3
"""End-of-run + per-voxel SVD fracridge (GLMsingle Stage 3 proxy).

Re-fits per-trial OLS LSS on full-run BOLD, then applies per-voxel SVD-based
fractional ridge to shrink the (n_trials × n_voxels) β matrix toward zero
in coefficient space. This is the proper Rokem & Kay 2020 fracridge applied
to the LSS β matrix per voxel.

We sweep frac ∈ {0.5, 0.7, 0.9} to bracket the regime. Memory note: prior
'real fracridge' tests on partial-window cells dropped retrieval to chance
(0.51-0.56 AUC) when frac<1.0 — but those were tested on short LSS windows
where shrinkage hits the per-trial signal directly. On full-run EoR the
per-trial β has more data behind it, so shrinkage may behave differently.

Compared against the OLS+Glover EoR baseline emitted by the HRF library
driver — apples-to-apples since both use OLS LSS with full-run BOLD.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

from prereg_variant_sweep import load_mask, load_rtmotion, load_events
from rt_glm_variants import build_design_matrix, make_glover_hrf

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT3T = LOCAL / "rt3t" / "data"
OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

import prereg_variant_sweep as P
P.RT3T = RT3T
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = RT3T / "sub-005_final_mask.nii.gz"
P.RELMASK = RT3T / "sub-005_ses-01_task-C_relmask.npy"
P.EVENTS_DIR = RT3T / "events"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
FRACS = [0.5, 0.7, 0.9]


def fit_run_lss_glover(ts: np.ndarray, onsets: np.ndarray, n_trs: int) -> np.ndarray:
    n_hrf_trs = int(np.ceil(32.0 / TR))
    hrf = make_glover_hrf(TR, n_hrf_trs)
    n_trials = len(onsets)
    V = ts.shape[0]
    out = np.zeros((n_trials, V), dtype=np.float32)
    for trial_i in range(n_trials):
        dm, probe_col = build_design_matrix(onsets, TR, n_trs, hrf, trial_i)
        XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
        beta = (XtX_inv @ dm.T @ ts.T).T
        out[trial_i] = beta[:, probe_col].astype(np.float32)
    return out


def fracridge_per_voxel(beta_matrix: np.ndarray, frac: float) -> np.ndarray:
    """Apply SVD-based fractional ridge per voxel.

    Treats the (n_trials, V) β matrix as a design problem where for each
    voxel v we shrink β[:, v] toward 0 via SVD. The 'design' for shrinkage
    is the identity (i.e., we're shrinking the β estimates themselves
    treating them as a linear system). This matches the post-fit fracridge
    application used in GLMsingle Stage 3 over the per-voxel SVD of LSS β.
    """
    if abs(frac - 1.0) < 1e-6:
        return beta_matrix.copy()
    n_trials, V = beta_matrix.shape
    # Per-voxel SVD over the trial dimension is trivial (single column),
    # so the standard formulation: shrinkage λ s.t. ||β(λ)|| / ||β_ols|| = frac.
    # For a 1D vector under L2 ridge: β(λ) = β_ols / (1 + λ); norm scales as 1/(1+λ).
    # So frac = 1/(1+λ) → λ = 1/frac - 1 → β(λ) = β_ols * frac.
    # Equivalent to scalar shrinkage by `frac`.
    return (beta_matrix * frac).astype(np.float32)


def fracridge_global_svd(beta_matrix: np.ndarray, frac: float) -> np.ndarray:
    """Apply SVD-based fractional ridge globally across (n_trials × V).

    Decompose β = U S Vᵀ over the full (n_trials, V) matrix, shrink each
    component s_i → s_i / (1 + λ_i) so that ||β(λ)|| / ||β_ols|| = frac.
    Closed form (uniform λ): equivalent to scaling all singular values by
    f_i where Σ(s_i*f_i)² = frac² Σ s_i². For uniform shrinkage f_i=frac
    this reduces to scalar shrinkage. For *spectral* shrinkage we use a
    per-component λ that minimizes residual subject to the norm constraint —
    standard fracridge uses brentq to find one λ giving the target norm.
    """
    if abs(frac - 1.0) < 1e-6:
        return beta_matrix.copy()
    U, S, Vt = np.linalg.svd(beta_matrix, full_matrices=False)
    target_norm = frac * np.linalg.norm(S)
    from scipy.optimize import brentq

    def norm_at(lam):
        return float(np.linalg.norm(S / (1.0 + lam / (S ** 2 + 1e-12))))

    try:
        lam = brentq(lambda l: norm_at(l) - target_norm, a=1e-12, b=1e12, xtol=1e-3)
    except ValueError:
        lam = 0.0
    S_shrunk = S / (1.0 + lam / (S ** 2 + 1e-12))
    return (U @ np.diag(S_shrunk) @ Vt).astype(np.float32)


def inclusive_cumz(arr: np.ndarray) -> np.ndarray:
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


print("=== fitting OLS+Glover EoR raw βs (no z, no fracridge) ===")
flat_brain, rel = load_mask()
all_betas = []
image_history = []
t0 = time.time()
for run in RUNS:
    ts = load_rtmotion(SESSION, run, flat_brain, rel)
    events = load_events(SESSION, run)
    onsets = events["onset_rel"].values.astype(np.float32)
    n_trs = ts.shape[1]
    run_betas = fit_run_lss_glover(ts, onsets, n_trs)
    all_betas.append(run_betas)
    image_history.extend([str(events.iloc[i].get("image_name", str(i))) for i in range(len(onsets))])
    print(f"  run-{run:02d}: {run_betas.shape} ({time.time()-t0:.1f}s elapsed)")
raw = np.concatenate(all_betas, axis=0)
trial_ids_arr = np.asarray(image_history)
print(f"  raw concat: {raw.shape}")

for frac in FRACS:
    cell_name = f"RT_paper_EoR_OLS_glover_frac{int(frac*100):02d}_inclz"
    print(f"\n=== {cell_name}  (frac={frac}, full-run, OLS+Glover, single-rep, inclusive cum-z) ===")
    shrunk = fracridge_global_svd(raw, frac)
    z = inclusive_cumz(shrunk)
    np.save(OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", z)
    np.save(OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", trial_ids_arr)
    cfg = {
        "cell": cell_name, "session": SESSION, "runs": RUNS, "tr": TR,
        "engine": "OLS LSS + global-SVD fracridge", "frac": frac,
        "bold_source": "rtmotion", "windowing": "full-run (EoR equivalent)",
        "cum_z_formula": "inclusive (arr[:i+1])",
        "fracridge_mode": "global SVD across (n_trials, V), brentq λ for target norm",
    }
    with open(OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {cell_name}: {z.shape}")

print(f"\ntotal time: {time.time()-t0:.1f}s")
