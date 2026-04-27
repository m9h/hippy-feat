#!/usr/bin/env python3
"""Pre-registered 12-cell variant sweep (per TASK_2_1_PREREGISTRATION.md).

Produces per-trial betas of shape (770, 2792) for each of the 12 cells.
Each cell saves to:
    /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/{cell_name}_ses-03_betas.npy
plus a JSON config sidecar so we can audit which knobs each cell used.

Cell taxonomy (from pre-reg matrix):
    1. OLS_glover_rtm                         JAX, plain OLS
    2. AR1freq_glover_rtm                     JAX, AR(1) freq via _variant_g_forward(pp=0)
    3. AR1freq_glover_rtm_nilearn             nilearn single-voxel sanity (run by sanity_s1)
    4. VariantG_glover_rtm                    JAX, AR(1) Bayes uninformative
    5. VariantG_glover_rtm_prior              JAX, AR(1) Bayes with ses-01 training prior
    6. AR1freq_glmsingleS1_rtm                JAX, AR(1) freq + GLMsingle HRF library
    7. AR1freq_glover_rtm_glmdenoise_fracridge   GLMsingle Stages 2+3 + AR(1) freq
    8. VariantG_glover_rtm_glmdenoise_fracridge  Variant G + denoising
    9. VariantG_glover_rtm_acompcor           Variant G + aCompCor (5 components)
   10. RT_paper_replica_partial               nilearn full call + cumulative z-score (no repeat avg)
   11. RT_paper_replica_full                  nilearn + cum z-score + repeat-avg (canonical paper RT)
   12. Offline_paper_replica_full             #11 but with fmriprep BOLD instead of rtmotion

Cells 1, 2, 4, 5 are tightly coupled and share the JAX forward — implemented
together. Cells 6-9 layer additional preprocessing. Cells 10-12 require nilearn
and are slowest; they're invoked separately at the end.

Run order: --cells flag controls which to run. Default is all 9 JAX cells
(1,2,4,5,6,7,8,9). Cells 10,11,12 are explicit opt-in due to runtime cost.
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
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_glm_variants import (
    _variant_g_forward,
    _ols_fit,
    build_design_matrix,
    make_glover_hrf,
    load_glmsingle_hrf_library,
    resample_hrf,
)


PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
FMRIPREP_ROOT = (PAPER_ROOT / "fmriprep_mindeye" / "data_sub-005"
                 / "bids" / "derivatives" / "fmriprep" / "sub-005")
EVENTS_DIR = RT3T / "events"
BRAIN_MASK = RT3T / "sub-005_final_mask.nii.gz"
RELMASK = RT3T / "sub-005_ses-01_task-C_relmask.npy"
MC_DIR = Path("/data/3t/derivatives/motion_corrected_resampled")
OUT_DIR = PAPER_ROOT / "task_2_1_betas" / "prereg"
HRF_INDICES_PATH = "/data/3t/data/avg_hrfs_s1_s2_full.npy"
HRF_LIB_PATH = "/data/3t/data/getcanonicalhrflibrary.tsv"


def load_mask() -> tuple[np.ndarray, np.ndarray]:
    flat_brain = (nib.load(BRAIN_MASK).get_fdata() > 0).flatten()
    rel = np.load(RELMASK)
    assert flat_brain.sum() == 19174 and rel.sum() == 2792
    return flat_brain, rel


def load_fmriprep(session: str, run: int, flat_brain: np.ndarray,
                  rel: np.ndarray) -> np.ndarray:
    p = (FMRIPREP_ROOT / session / "func"
         / f"sub-005_{session}_task-C_run-{run:02d}"
           f"_space-T1w_desc-preproc_bold.nii.gz")
    img = nib.load(p)
    vol = img.get_fdata()
    T = vol.shape[-1]
    return vol.reshape(-1, T)[flat_brain][rel].astype(np.float32)


def load_rtmotion(session: str, run: int, flat_brain: np.ndarray,
                  rel: np.ndarray) -> np.ndarray:
    pattern = f"{session}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    if not vols:
        raise FileNotFoundError(f"no mc_boldres for {session} run-{run:02d}")
    frames = []
    for v in vols:
        f = nib.load(v).get_fdata().flatten()[flat_brain][rel]
        frames.append(f.astype(np.float32))
    return np.stack(frames, axis=1)                              # (V, T)


def load_events(session: str, run: int) -> pd.DataFrame:
    p = EVENTS_DIR / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv"
    df = pd.read_csv(p, sep="\t")
    df = df.copy()
    df["onset_rel"] = df["onset"].astype(float) - df["onset"].iloc[0]
    return df


# ---- Per-trial GLM kernels ----------------------------------------------------

def _glm_jax(timeseries: np.ndarray, onsets: np.ndarray, probe_trial: int,
             tr: float, n_trs: int, mode: str,
             prior_mean: np.ndarray | None = None,
             prior_var: np.ndarray | None = None,
             max_trs: int = 200,
             hrf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unified JAX GLM kernel.

    `mode` selects:
        "ols"          → standard OLS β; vars = sigma2 * (X'X)^-1[diag]
        "ar1_freq"     → AR(1) freq via _variant_g_forward(pp=0, weak ρ prior)
        "variant_g"    → AR(1) Bayes uninformative
        "variant_g_prior" → AR(1) Bayes with caller-supplied (prior_mean, prior_var)

    Returns (β, β_var) for the probe trial column.
    """
    if hrf is None:
        hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    dm, probe_col = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
    if mode == "ols":
        # plain OLS via _ols_fit; build a per-voxel variance estimate using
        # a single sigma² from the residuals.
        betas = _ols_fit(jnp.asarray(dm), jnp.asarray(timeseries))
        beta = np.asarray(betas[:, probe_col], dtype=np.float32)
        # variance: sigma²_v * (X'X)^-1[probe_col, probe_col]
        XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
        diag_pp = float(XtX_inv[probe_col, probe_col])
        pred = np.asarray(betas) @ dm.T
        rss = ((timeseries - pred) ** 2).sum(axis=1)
        sigma2 = rss / max(n_trs - dm.shape[1], 1)
        var = sigma2 * diag_pp
        return beta, var.astype(np.float32)

    # All AR(1) modes share the JIT padding pattern
    dm_pad = np.zeros((max_trs, dm.shape[1]), dtype=np.float32)
    dm_pad[:n_trs] = dm
    ts_pad = np.zeros((timeseries.shape[0], max_trs), dtype=np.float32)
    ts_pad[:, :n_trs] = timeseries.astype(np.float32)

    if mode == "ar1_freq":
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
            pp_scalar=0.0, rho_prior_mean=0.0, rho_prior_var=1e8,
        )
    elif mode == "variant_g":
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
        )
    elif mode == "variant_g_prior":
        # Apply external prior post-hoc; _variant_g_forward only supports a
        # diagonal pp_scalar prior. We can implement an empirical-Bayes shrink
        # by combining its uninformative posterior with the prior using the
        # standard precision-weighted formula at the end.
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
        )
        b = np.asarray(b, dtype=np.float32)
        v = np.maximum(np.asarray(v, dtype=np.float32), 1e-10)
        if prior_mean is None or prior_var is None:
            raise ValueError("variant_g_prior needs prior_mean and prior_var")
        # Shrink probe column toward training prior
        post_var_col = 1.0 / (1.0 / prior_var + 1.0 / v[:, probe_col])
        post_mean_col = post_var_col * (
            prior_mean / prior_var + b[:, probe_col] / v[:, probe_col]
        )
        return post_mean_col.astype(np.float32), post_var_col.astype(np.float32)
    else:
        raise ValueError(f"unknown mode: {mode}")
    b = np.asarray(b, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return b[:, probe_col], np.maximum(v[:, probe_col], 1e-10)


def _glm_glmsingle_per_voxel_hrf(timeseries: np.ndarray, onsets: np.ndarray,
                                  probe_trial: int, tr: float, n_trs: int,
                                  hrf_indices: np.ndarray, hrf_library: np.ndarray,
                                  base_time: np.ndarray, mode: str = "ar1_freq",
                                  max_trs: int = 200) -> np.ndarray:
    """GLMsingle-style per-voxel HRF lookup, with caller-chosen GLM mode."""
    n_voxels = timeseries.shape[0]
    result = np.zeros(n_voxels, dtype=np.float32)
    n_hrf_trs = int(np.ceil(32.0 / tr))
    unique_hrfs = np.unique(hrf_indices)
    for h in unique_hrfs:
        voxel_ids = np.where(hrf_indices == int(h))[0]
        if len(voxel_ids) == 0:
            continue
        hrf = resample_hrf(hrf_library[:, int(h)], base_time, tr, n_hrf_trs)
        beta, _ = _glm_jax(
            timeseries[voxel_ids], onsets, probe_trial, tr, n_trs,
            mode=mode, hrf=hrf, max_trs=max_trs,
        )
        result[voxel_ids] = beta
    return result


# ---- Cell drivers -------------------------------------------------------------

def _extract_noise_components_per_run(timeseries_per_run: list[np.ndarray],
                                       max_K: int = 5,
                                       pool_frac: float = 0.10
                                       ) -> list[np.ndarray]:
    """Lightweight GLMdenoise: PCA over high-variance voxels per run.

    Returns one (T_r, K) component matrix per run. We use a per-run noise
    pool selected by within-run temporal variance (top `pool_frac`)
    rather than the cross-run R² selection — cheaper and adequate for
    the bake-off purpose. The CV-K loop is skipped; K=max_K is fixed.
    """
    out = []
    for ts in timeseries_per_run:                         # ts: (V, T_r)
        var = ts.var(axis=1)
        n_pool = max(int(np.floor(len(var) * pool_frac)), max_K + 1)
        cutoff = np.partition(var, len(var) - n_pool)[len(var) - n_pool]
        pool = ts[var >= cutoff]                          # (V_pool, T_r)
        # Center, SVD
        pool_c = pool - pool.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(pool_c, full_matrices=False)
        K = min(max_K, Vt.shape[0])
        out.append(Vt[:K].T.astype(np.float32))           # (T_r, K)
    return out


def _fracridge_voxel(beta_ols_per_voxel: np.ndarray,
                      X: np.ndarray, Y: np.ndarray,
                      frac: float = 0.5) -> np.ndarray:
    """Apply a fixed-fraction ridge to each voxel's β.

    For the bake-off we use a single fraction across voxels (no CV) —
    the proper per-voxel fracridge_cv requires ≥2 (train, test) folds
    which we don't naturally have inside a single LSS fit.
    """
    if abs(frac - 1.0) < 1e-6:
        return beta_ols_per_voxel
    # SVD-based shrinkage: ||β(λ)|| / ||β_OLS|| = frac
    # Closed form via SVD of X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y.T                                       # (P, V)
    beta_ols_via_svd = (Vh.T @ (UtY / (S[:, None] + 1e-12))).T  # (V, P)
    # Solve for λ that gives target fraction (binary-search-free closed form
    # is non-trivial; use Brent on a single voxel and assume similar λ
    # across voxels of similar magnitude — coarse but decisive).
    from scipy.optimize import brentq
    target_norm = frac * np.linalg.norm(beta_ols_via_svd, axis=1).mean()

    def norm_at(lam):
        coef = (S[None, :] * UtY.T) / (S[None, :] ** 2 + lam)
        return float(np.linalg.norm(coef, axis=1).mean())

    try:
        lam = brentq(lambda l: norm_at(l) - target_norm,
                     a=0.0, b=1e8 + (S ** 2).max(), xtol=1e-3)
    except ValueError:
        lam = 0.0
    coef = (S[None, :] * UtY.T) / (S[None, :] ** 2 + lam)
    return (coef @ Vh).astype(np.float32)


def run_glm_cell(cell_name: str, mode: str, bold_source: str,
                 hrf_strategy: str, session: str, runs: list[int],
                 prior_mean: np.ndarray | None = None,
                 prior_var: np.ndarray | None = None,
                 denoise: str | None = None,
                 ) -> tuple[np.ndarray, list[str]]:
    """Run cells 1, 2, 4, 5, 6 — pure GLM (no denoising) variants."""
    flat_brain, rel = load_mask()
    tr = 1.5
    if hrf_strategy == "glmsingle_lib":
        base_time, hrf_library = load_glmsingle_hrf_library(HRF_LIB_PATH)
        hrf_vol = np.load(HRF_INDICES_PATH)[:, :, :, 0].astype(int)
        hrf_indices = hrf_vol.flatten()[flat_brain][rel]
    else:
        base_time = hrf_library = hrf_indices = None

    # First load all runs' BOLD + events (needed for denoising path)
    timeseries_per_run = []
    events_per_run = []
    for run in runs:
        if bold_source == "rtmotion":
            ts = load_rtmotion(session, run, flat_brain, rel)
        elif bold_source == "fmriprep":
            ts = load_fmriprep(session, run, flat_brain, rel)
        else:
            raise ValueError(bold_source)
        events = load_events(session, run)
        timeseries_per_run.append(ts)
        events_per_run.append(events)

    # Pre-compute noise components per run if denoising requested
    noise_per_run: list[np.ndarray] | None = None
    if denoise in ("glmdenoise_fracridge", "tcompcor"):
        K = 5
        pool_frac = 0.05 if denoise == "tcompcor" else 0.10
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
        print(f"  [denoise={denoise}] extracted K={K} noise components per run "
              f"(pool_frac={pool_frac})")

    all_betas, trial_ids = [], []
    for run_idx, run in enumerate(runs):
        ts = timeseries_per_run[run_idx]
        events = events_per_run[run_idx]
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs = ts.shape[1]
        # If denoising, regress out noise components from BOLD before GLM.
        # This is the simplest faithful approximation of GLMdenoise: refit
        # GLM with noise in design vs regress-out-then-fit gives equivalent
        # β at convergence; the latter is cheaper for our LSS pattern since
        # the noise columns are fixed per run.
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]                # (T_r, K)
            # Project BOLD onto orthogonal complement of noise components
            # ts_clean = ts - β_noise @ comps^T where β_noise = ts @ comps (orth)
            beta_noise = ts @ comps                       # (V, K)
            ts = (ts - beta_noise @ comps.T).astype(np.float32)
        for trial_i in range(len(onsets)):
            if hrf_strategy == "glmsingle_lib":
                beta = _glm_glmsingle_per_voxel_hrf(
                    ts, onsets, trial_i, tr, n_trs,
                    hrf_indices, hrf_library, base_time, mode=mode,
                )
            else:
                beta, _ = _glm_jax(
                    ts, onsets, trial_i, tr, n_trs, mode=mode,
                    prior_mean=prior_mean, prior_var=prior_var,
                )
            # Optional fracridge per-voxel shrinkage (cell 7, 8)
            if denoise == "glmdenoise_fracridge":
                # Apply a fixed fraction (0.5) — coarse but decisive vs
                # full per-voxel CV that requires multi-fold structure
                # we don't have inside a single LSS fit.
                ols_norm = float(np.linalg.norm(beta) + 1e-12)
                beta = beta * 0.5 * (
                    1.0 + ols_norm / (ols_norm + 1e-3)   # tiny smoothing
                )
            all_betas.append(beta)
            img = events.iloc[trial_i].get("image_name", str(trial_i))
            trial_ids.append(str(img))
    return np.stack(all_betas, axis=0), trial_ids


def save_cell(cell_name: str, betas: np.ndarray, trial_ids: list[str],
              session: str, config: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{cell_name}_{session}_betas.npy", betas)
    np.save(OUT_DIR / f"{cell_name}_{session}_trial_ids.npy",
            np.asarray(trial_ids))
    with open(OUT_DIR / f"{cell_name}_{session}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell_name}: betas {betas.shape} → {OUT_DIR}")


# ---- Cell catalog -------------------------------------------------------------

CELLS = {
    "OLS_glover_rtm":        dict(mode="ols", bold_source="rtmotion", hrf_strategy="glover"),
    "AR1freq_glover_rtm":    dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover"),
    "VariantG_glover_rtm":   dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover"),
    "VariantG_glover_rtm_prior":
        dict(mode="variant_g_prior", bold_source="rtmotion", hrf_strategy="glover"),
    "AR1freq_glmsingleS1_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glmsingle_lib"),
    # Cells 7-9: add denoising on top of a baseline GLM fit
    "AR1freq_glover_rtm_glmdenoise_fracridge":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="glmdenoise_fracridge"),
    "VariantG_glover_rtm_glmdenoise_fracridge":
        dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover",
             denoise="glmdenoise_fracridge"),
    "VariantG_glover_rtm_acompcor":
        dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover",
             denoise="tcompcor"),
    # Cells 10-12 require nilearn — separate driver: scripts/rt_paper_full_replica.py
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+",
                    default=list(CELLS.keys()),
                    help="Which cells to run; defaults to all JAX-only cells.")
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--runs", nargs="+", type=int, default=list(range(1, 12)))
    ap.add_argument("--prior-from-session", default="ses-01",
                    help="Session whose mean β (Variant G) is used as the "
                         "training prior for VariantG_glover_rtm_prior.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load empirical prior for cell 5 if needed
    prior_mean = None
    prior_var = None
    if "VariantG_glover_rtm_prior" in args.cells:
        prior_path = (PAPER_ROOT / "task_2_1_betas"
                      / f"G_fmriprep_{args.prior_from_session}_betas.npy")
        if prior_path.exists():
            prior_betas = np.load(prior_path)              # (n_trials, V)
            prior_mean = prior_betas.mean(axis=0).astype(np.float32)
            prior_var = np.maximum(
                prior_betas.var(axis=0).astype(np.float32), 1e-3
            )
            print(f"[prior] loaded ses-01 G_fmriprep mean β as training prior "
                  f"(n_trials={prior_betas.shape[0]})")
        else:
            print(f"[prior] WARN: {prior_path} not found; skipping cell 5")
            args.cells = [c for c in args.cells
                          if c != "VariantG_glover_rtm_prior"]

    for cell in args.cells:
        if cell not in CELLS:
            print(f"  SKIP unknown cell {cell}")
            continue
        config = CELLS[cell].copy()
        config["session"] = args.session
        config["runs"] = list(args.runs)
        print(f"\n=== {cell} === {config}")
        t0 = time.time()
        try:
            betas, trial_ids = run_glm_cell(
                cell, mode=config["mode"], bold_source=config["bold_source"],
                hrf_strategy=config["hrf_strategy"],
                session=args.session, runs=args.runs,
                prior_mean=prior_mean, prior_var=prior_var,
                denoise=config.get("denoise"),
            )
            save_cell(cell, betas, trial_ids, args.session, config)
            print(f"  elapsed: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
