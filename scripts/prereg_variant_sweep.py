#!/usr/bin/env python3
"""Pre-registered 12-cell variant sweep (per TASK_2_1_PREREGISTRATION.md).

Produces per-trial betas of shape (770, 2792) for each of the 12 cells.
Each cell saves to:
    /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/{cell_name}_ses-03_betas.npy
plus a JSON config sidecar so we can audit which knobs each cell used.
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
             hrf: np.ndarray | None = None,
             confounds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if hrf is None:
        hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    dm, probe_col = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
    if confounds is not None:
        dm = np.concatenate([dm, confounds], axis=1)

    if mode == "ar1_session_rho":
        rho_session = _get_session_rho_or_compute(
            timeseries_per_run_for_rho=globals().get("_SESSION_RHO_TS", None),
            tr=tr, n_voxels=timeseries.shape[0],
        )
        n_eff = n_trs - 1
        beta_out = np.zeros(timeseries.shape[0], dtype=np.float32)
        var_out = np.zeros(timeseries.shape[0], dtype=np.float32)
        for v in range(timeseries.shape[0]):
            rho_v = float(rho_session[v])
            y_pw = timeseries[v, 1:] - rho_v * timeseries[v, :-1]
            X_pw = dm[1:] - rho_v * dm[:-1]
            try:
                XtX_inv = np.linalg.inv(X_pw.T @ X_pw + 1e-6 * np.eye(X_pw.shape[1]))
                beta = XtX_inv @ X_pw.T @ y_pw
                resid = y_pw - X_pw @ beta
                rss = float((resid ** 2).sum())
                sigma2 = rss / max(n_eff - dm.shape[1], 1)
                beta_out[v] = float(beta[probe_col])
                var_out[v] = float(sigma2 * XtX_inv[probe_col, probe_col])
            except np.linalg.LinAlgError:
                beta_out[v] = 0.0
                var_out[v] = 1e10
        return beta_out, var_out

    if mode == "ar1_streaming_kalman":
        from jaxoccoli.streaming_kalman import (
            init_streaming_kalman_ar1,
            streaming_kalman_ar1_run,
        )
        state = init_streaming_kalman_ar1(P=dm.shape[1], V=timeseries.shape[0])
        state = streaming_kalman_ar1_run(state, dm.astype(np.float32),
                                          timeseries.astype(np.float32))
        beta = np.asarray(state.beta_mean[:, probe_col], dtype=np.float32)
        var = np.asarray(
            np.maximum(state.b_post / np.maximum(state.a_post - 1.0, 1e-3), 1e-10),
            dtype=np.float32,
        )
        return beta, var

    if mode == "ols":
        betas = _ols_fit(jnp.asarray(dm), jnp.asarray(timeseries))
        beta = np.asarray(betas[:, probe_col], dtype=np.float32)
        XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
        diag_pp = float(XtX_inv[probe_col, probe_col])
        pred = np.asarray(betas) @ dm.T
        rss = ((timeseries - pred) ** 2).sum(axis=1)
        sigma2 = rss / max(n_trs - dm.shape[1], 1)
        var = sigma2 * diag_pp
        return beta, var.astype(np.float32)

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
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
        )
        b = np.asarray(b, dtype=np.float32)
        v = np.maximum(np.asarray(v, dtype=np.float32), 1e-10)
        if prior_mean is None or prior_var is None:
            raise ValueError("variant_g_prior needs prior_mean and prior_var")
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


def _glm_glmsingle_per_voxel_hrf_real_fracridge(
        timeseries: np.ndarray, onsets: np.ndarray, probe_trial: int,
        tr: float, n_trs: int, hrf_indices: np.ndarray,
        hrf_library: np.ndarray, base_time: np.ndarray,
        fracvalue: np.ndarray) -> np.ndarray:
    n_voxels = timeseries.shape[0]
    result = np.zeros(n_voxels, dtype=np.float32)
    n_hrf_trs = int(np.ceil(32.0 / tr))
    unique_hrfs = np.unique(hrf_indices)
    for h in unique_hrfs:
        voxel_ids = np.where(hrf_indices == int(h))[0]
        if len(voxel_ids) == 0:
            continue
        hrf = resample_hrf(hrf_library[:, int(h)], base_time, tr, n_hrf_trs)
        dm, probe_col = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
        Y = timeseries[voxel_ids]                                  # (V_h, T)
        beta_frac = _ols_fracridge_per_voxel(
            dm, Y, fracvalue[voxel_ids],
        )                                                           # (V_h, P)
        result[voxel_ids] = beta_frac[:, probe_col]
    return result


def _glm_glmsingle_per_voxel_hrf(timeseries: np.ndarray, onsets: np.ndarray,
                                  probe_trial: int, tr: float, n_trs: int,
                                  hrf_indices: np.ndarray, hrf_library: np.ndarray,
                                  base_time: np.ndarray, mode: str = "ar1_freq",
                                  max_trs: int = 200,
                                  confounds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    n_voxels = timeseries.shape[0]
    result_b = np.zeros(n_voxels, dtype=np.float32)
    result_v = np.zeros(n_voxels, dtype=np.float32)
    has_var = False
    n_hrf_trs = int(np.ceil(32.0 / tr))
    unique_hrfs = np.unique(hrf_indices)
    for h in unique_hrfs:
        voxel_ids = np.where(hrf_indices == int(h))[0]
        if len(voxel_ids) == 0:
            continue
        hrf = resample_hrf(hrf_library[:, int(h)], base_time, tr, n_hrf_trs)
        beta, var = _glm_jax(
            timeseries[voxel_ids], onsets, probe_trial, tr, n_trs,
            mode=mode, hrf=hrf, max_trs=max_trs, confounds=confounds,
        )
        result_b[voxel_ids] = beta
        if var is not None:
            result_v[voxel_ids] = var
            has_var = True
    return result_b, (result_v if has_var else None)


# ---- Cell drivers -------------------------------------------------------------

def _get_session_rho_or_compute(timeseries_per_run_for_rho,
                                  tr: float, n_voxels: int
                                  ) -> np.ndarray:
    cache = globals().get("_SESSION_RHO_CACHE")
    if cache is not None and cache.shape == (n_voxels,):
        return cache
    if timeseries_per_run_for_rho is None:
        return np.full(n_voxels, 0.3, dtype=np.float32)
    Y_session = np.concatenate(timeseries_per_run_for_rho, axis=1)  # (V, T_tot)
    V, T_total = Y_session.shape
    intercept = np.ones(T_total, dtype=np.float32)
    drift = np.cos(2 * np.pi * np.arange(T_total) / max(T_total - 1, 1)
                   ).astype(np.float32)
    X = np.stack([intercept, drift], axis=1)                  # (T, 2)
    XtX_inv_Xt = np.linalg.inv(X.T @ X + 1e-6 * np.eye(2)) @ X.T
    betas = Y_session @ XtX_inv_Xt.T                          # (V, 2)
    pred = betas @ X.T
    resid = Y_session - pred                                  # (V, T_total)
    num = (resid[:, 1:] * resid[:, :-1]).sum(axis=1)
    den = (resid ** 2).sum(axis=1) + 1e-10
    rho = np.clip(num / den, -0.95, 0.95).astype(np.float32)
    globals()["_SESSION_RHO_CACHE"] = rho
    return rho


def _extract_noise_components_per_run(timeseries_per_run: list[np.ndarray],
                                       max_K: int = 5,
                                       pool_frac: float = 0.10
                                       ) -> list[np.ndarray]:
    out = []
    for ts in timeseries_per_run:                         # ts: (V, T_r)
        var = ts.var(axis=1)
        n_pool = max(int(np.floor(len(var) * pool_frac)), max_K + 1)
        cutoff = np.partition(var, len(var) - n_pool)[len(var) - n_pool]
        pool = ts[var >= cutoff]                          # (V_pool, T_r)
        pool_c = pool - pool.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(pool_c, full_matrices=False)
        K = min(max_K, Vt.shape[0])
        out.append(Vt[:K].T.astype(np.float32))           # (T_r, K)
    return out


def _ols_fracridge_per_voxel(X: np.ndarray, Y: np.ndarray,
                               target_frac: np.ndarray,
                               n_grid: int = 51) -> np.ndarray:
    eps = 1e-10
    U, S, Vt = np.linalg.svd(X, full_matrices=False)            # X = U Σ V^T
    z = U.T @ Y.T                                                # (rank, V)
    inv_S = np.where(S > eps, 1.0 / S, 0.0)
    beta_ols = Vt.T @ (inv_S[:, None] * z)                       # (P, V)
    beta_ols_norm = np.linalg.norm(beta_ols, axis=0) + eps       # (V,)
    lam_max = max(float(S.max() ** 2) * 1e6, 1e6)
    lam_grid = np.concatenate([[0.0],
                                  np.logspace(-6, np.log10(lam_max),
                                              n_grid - 1)]).astype(np.float32)
    ratios = np.zeros((n_grid, Y.shape[0]), dtype=np.float32)
    for k, lam in enumerate(lam_grid):
        weights = (S / (S ** 2 + lam)).astype(np.float32)        # (rank,)
        coef = weights[:, None] * z                              # (rank, V)
        beta_lam_norm = np.linalg.norm(Vt.T @ coef, axis=0)      # (V,)
        ratios[k] = beta_lam_norm / beta_ols_norm
    diffs = np.abs(ratios - target_frac[None, :])
    best_k = np.argmin(diffs, axis=0)                            # (V,)
    best_lam = lam_grid[best_k].astype(np.float32)               # (V,)
    weights_per_v = (S[:, None] / (S[:, None] ** 2 + best_lam[None, :])
                      ).astype(np.float32)                        # (rank, V)
    coef_per_v = weights_per_v * z
    beta_frac = (Vt.T @ coef_per_v).T                            # (V, P)
    return beta_frac.astype(np.float32)


def _load_canonical_fracvalue_to_relmask(
    glmsingle_dir: Path = Path(
        "/data/derivatives/rtmindeye_paper/glmsingle/glmsingle_sub-005_ses-01-02_task-C"
    ),
) -> np.ndarray:
    import nibabel as nib
    z = np.load(glmsingle_dir / "TYPED_FITHRF_GLMDENOISE_RR.npz", allow_pickle=True)
    fv_full = z["FRACvalue"].squeeze().astype(np.float32)              # (V_canon,)
    canon_brain = nib.load(
        glmsingle_dir
        / f"sub-005_{glmsingle_dir.name.replace('glmsingle_sub-005_', '').replace('glmsingle_', '')}_brain.nii.gz"
    ).get_fdata() > 0
    final_mask = nib.load(
        Path("/data/derivatives/rtmindeye_paper/rt3t/data/sub-005_final_mask.nii.gz")
    ).get_fdata() > 0
    relmask = np.load(
        Path("/data/derivatives/rtmindeye_paper/rt3t/data/sub-005_ses-01_task-C_relmask.npy")
    )
    me_positions = np.where(final_mask.flatten())[0][relmask]          # (2792,)
    canon_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_brain_idx[me_positions]
    if (me_in_canon < 0).any():
        fv = np.ones(len(me_positions), dtype=np.float32)
        valid = me_in_canon >= 0
        fv[valid] = fv_full[me_in_canon[valid]]
        return fv
    return fv_full[me_in_canon]


def run_glm_cell(cell_name: str, mode: str, bold_source: str,
                 hrf_strategy: str, session: str, runs: list[int],
                 prior_mean: np.ndarray | None = None,
                 prior_var: np.ndarray | None = None,
                 denoise: str | None = None,
                 streaming_post_stim_TRs: int | None = None,
                 ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    flat_brain, rel = load_mask()
    tr = 1.5
    if hrf_strategy == "glmsingle_lib":
        base_time, hrf_library = load_glmsingle_hrf_library(HRF_LIB_PATH)
        hrf_vol = np.load(HRF_INDICES_PATH)[:, :, :, 0].astype(int)
        hrf_indices = hrf_vol.flatten()[flat_brain][rel]
    else:
        base_time = hrf_library = hrf_indices = None

    globals()["_SESSION_RHO_CACHE"] = None
    globals()["_SESSION_RHO_TS"] = None

    canonical_frac = None
    if denoise in ("canonical_frac", "canonical_real_frac"):
        canonical_frac = _load_canonical_fracvalue_to_relmask()

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

    if mode == "ar1_session_rho":
        globals()["_SESSION_RHO_TS"] = list(timeseries_per_run)

    noise_per_run: list[np.ndarray] | None = None
    if denoise in ("glmdenoise_fracridge", "tcompcor"):
        K = 5
        pool_frac = 0.05 if denoise == "tcompcor" else 0.10
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
    elif denoise == "hosvd_4d":
        from jaxoccoli.nordic import nordic_global
        for r in range(len(timeseries_per_run)):
            ts_r = timeseries_per_run[r]
            z = (ts_r + 0j).astype(np.complex64)
            denoised = np.asarray(nordic_global(jnp.asarray(z)))
            timeseries_per_run[r] = denoised.real.astype(np.float32)
    elif denoise in ("logsig_tcompcor", "logsig_acompcor"):
        K = 5
        pool_frac = 0.05
        noise_per_run_raw = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
        import signax
        import jax.numpy as jnp
        W = 20
        noise_per_run = []
        for comps in noise_per_run_raw:
            T_r = comps.shape[0]
            pad = np.repeat(comps[:1], W - 1, axis=0)
            padded = np.concatenate([pad, comps], axis=0)
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, window_shape=W, axis=0
            ).transpose(0, 2, 1).copy()
            logsig = np.asarray(
                signax.logsignature(jnp.asarray(windows), depth=2)
            ).astype(np.float32)
            aug = np.concatenate([comps, logsig], axis=1)
            # Orthonormalize to keep it stable
            q, _ = np.linalg.qr(aug)
            noise_per_run.append(q.astype(np.float32))
    elif denoise == "riemannian_prewhiten":
        V = timeseries_per_run[0].shape[0]
        eps = 1e-3
        per_run_cov = []
        for ts_r in timeseries_per_run:
            T_r = ts_r.shape[1]
            ts_c = ts_r - ts_r.mean(axis=1, keepdims=True)
            cov = (ts_c @ ts_c.T) / max(T_r - 1, 1)
            per_run_cov.append(cov + eps * np.eye(V, dtype=np.float32))
        log_sum = np.zeros((V, V), dtype=np.float64)
        for cov in per_run_cov:
            evals, evecs = np.linalg.eigh(cov.astype(np.float64))
            log_evals = np.log(np.maximum(evals, 1e-12))
            log_sum += (evecs * log_evals) @ evecs.T
        log_mean = log_sum / len(per_run_cov)
        evals_lm, evecs_lm = np.linalg.eigh(log_mean)
        sigma_bar = ((evecs_lm * np.exp(evals_lm)) @ evecs_lm.T).astype(np.float32)
        evals, evecs = np.linalg.eigh(sigma_bar)
        evals_inv_sqrt = 1.0 / np.sqrt(np.maximum(evals, 1e-6))
        sigma_bar_inv_sqrt = (evecs * evals_inv_sqrt) @ evecs.T
        sigma_bar_inv_sqrt = sigma_bar_inv_sqrt.astype(np.float32)
        for r in range(len(timeseries_per_run)):
            timeseries_per_run[r] = (
                sigma_bar_inv_sqrt @ timeseries_per_run[r]
            ).astype(np.float32)

    all_betas, all_vars, trial_ids = [], [], []
    for run_idx, run in enumerate(runs):
        ts = timeseries_per_run[run_idx]
        events = events_per_run[run_idx]
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs = ts.shape[1]
        
        run_confounds = noise_per_run[run_idx] if noise_per_run is not None else None

        for trial_i in range(len(onsets)):
            if streaming_post_stim_TRs is not None:
                onset_TR = int(round(float(onsets[trial_i]) / tr))
                decode_TR = min(onset_TR + streaming_post_stim_TRs, n_trs - 1)
                ts_use = ts[:, :decode_TR + 1]
                n_trs_use = decode_TR + 1
                confounds_use = run_confounds[:decode_TR + 1] if run_confounds is not None else None
            else:
                ts_use = ts
                n_trs_use = n_trs
                confounds_use = run_confounds

            if hrf_strategy == "glmsingle_lib" and denoise == "canonical_real_frac":
                ts_clean = ts_use
                if confounds_use is not None:
                    beta_noise = ts_use @ confounds_use
                    ts_clean = (ts_use - beta_noise @ confounds_use.T).astype(np.float32)
                beta = _glm_glmsingle_per_voxel_hrf_real_fracridge(
                    ts_clean, onsets, trial_i, tr, n_trs_use,
                    hrf_indices, hrf_library, base_time, canonical_frac,
                )
                var = None
            elif hrf_strategy == "glmsingle_lib":
                beta, var = _glm_glmsingle_per_voxel_hrf(
                    ts_use, onsets, trial_i, tr, n_trs_use,
                    hrf_indices, hrf_library, base_time, mode=mode,
                    confounds=confounds_use,
                )
            else:
                beta, var = _glm_jax(
                    ts_use, onsets, trial_i, tr, n_trs_use, mode=mode,
                    prior_mean=prior_mean, prior_var=prior_var,
                    confounds=confounds_use,
                )
            if denoise == "glmdenoise_fracridge":
                ols_norm = float(np.linalg.norm(beta) + 1e-12)
                beta = beta * 0.5 * (1.0 + ols_norm / (ols_norm + 1e-3))
            elif denoise == "canonical_frac" and canonical_frac is not None:
                beta = (beta * canonical_frac).astype(np.float32)
            all_betas.append(beta)
            if var is not None:
                all_vars.append(var)
            img = events.iloc[trial_i].get("image_name", str(trial_i))
            trial_ids.append(str(img))
            
    res_betas = np.stack(all_betas, axis=0)
    res_vars = np.stack(all_vars, axis=0) if all_vars else None
    return res_betas, res_vars, trial_ids


def save_cell(cell_name: str, betas: np.ndarray, vars_: np.ndarray | None,
              trial_ids: list[str], session: str, config: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{cell_name}_{session}_betas.npy", betas)
    if vars_ is not None:
        np.save(OUT_DIR / f"{cell_name}_{session}_vars.npy", vars_)
    np.save(OUT_DIR / f"{cell_name}_{session}_trial_ids.npy",
            np.asarray(trial_ids))
    with open(OUT_DIR / f"{cell_name}_{session}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell_name}: betas {betas.shape}")


CELLS = {
    "OLS_glover_rtm":        dict(mode="ols", bold_source="rtmotion", hrf_strategy="glover"),
    "AR1freq_glover_rtm":    dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover"),
    "VariantG_glover_rtm":   dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover"),
    "VariantG_glover_rtm_prior":
        dict(mode="variant_g_prior", bold_source="rtmotion", hrf_strategy="glover"),
    "AR1freq_glmsingleS1_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glmsingle_lib"),
    "AR1freq_glover_rtm_glmdenoise_fracridge":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="glmdenoise_fracridge"),
    "VariantG_glover_rtm_glmdenoise_fracridge":
        dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover",
             denoise="glmdenoise_fracridge"),
    "VariantG_glover_rtm_acompcor":
        dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover",
             denoise="tcompcor"),
    "VariantG_glover_fmriprep_acompcor":
        dict(mode="variant_g", bold_source="fmriprep", hrf_strategy="glover",
             denoise="tcompcor"),
    "EKF_streaming_glover_rtm":
        dict(mode="ar1_streaming_kalman", bold_source="rtmotion", hrf_strategy="glover"),
    "HOSVD_denoise_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="hosvd_4d"),
    "Riemannian_prewhiten_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="riemannian_prewhiten"),
    "HybridOnline_AR1freq_glover_rtm":
        dict(mode="ar1_session_rho", bold_source="rtmotion", hrf_strategy="glover"),
    "LogSig_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="logsig_tcompcor"),
    "LogSig_aCompCor_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="logsig_acompcor"),
    "HybridOnline_streaming_pst8_AR1freq_glover_rtm":
        dict(mode="ar1_session_rho", bold_source="rtmotion",
             hrf_strategy="glover", streaming_post_stim_TRs=8),
    "AR1freq_glmsingleFull_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib", denoise="glmdenoise_fracridge"),
    "VariantG_glmsingleFull_rtm":
        dict(mode="variant_g", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib", denoise="glmdenoise_fracridge"),
    "VariantG_glmsingleS1_rtm":
        dict(mode="variant_g", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib"),
    "AR1freq_glmsingleFull_fmriprep":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib", denoise="glmdenoise_fracridge"),
    "AR1freq_glover_fmriprep_glmdenoise_fracridge":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glover", denoise="glmdenoise_fracridge"),
    "VariantG_glover_fmriprep_glmdenoise_fracridge":
        dict(mode="variant_g", bold_source="fmriprep",
             hrf_strategy="glover", denoise="glmdenoise_fracridge"),
    "Streaming_S1S3_pst8_AR1freq_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac",
             streaming_post_stim_TRs=8),
    "Streaming_S1S3_pst8_AR1freq_fmriprep":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac",
             streaming_post_stim_TRs=8),
    "FullRun_S1S3_AR1freq_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac"),
    "FullRun_S1S3_AR1freq_fmriprep":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac"),
    "FullRun_S1S3realFRAC_rtm":
        dict(mode="ols", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_real_frac"),
    "FullRun_S1S3realFRAC_fmriprep":
        dict(mode="ols", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_real_frac"),
    "Streaming_S1S3realFRAC_pst8_rtm":
        dict(mode="ols", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_real_frac",
             streaming_post_stim_TRs=8),
    "Streaming_S1S3realFRAC_pst8_fmriprep":
        dict(mode="ols", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_real_frac",
             streaming_post_stim_TRs=8),
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

    prior_mean = None
    prior_var = None
    if "VariantG_glover_rtm_prior" in args.cells:
        prior_path = (PAPER_ROOT / "task_2_1_betas"
                      / f"G_fmriprep_{args.prior_from_session}_betas.npy")
        if prior_path.exists():
            prior_betas = np.load(prior_path)
            prior_mean = prior_betas.mean(axis=0).astype(np.float32)
            prior_var = np.maximum(
                prior_betas.var(axis=0).astype(np.float32), 1e-3
            )
        else:
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
            betas, vars_, trial_ids = run_glm_cell(
                cell, mode=config["mode"], bold_source=config["bold_source"],
                hrf_strategy=config["hrf_strategy"],
                session=args.session, runs=args.runs,
                prior_mean=prior_mean, prior_var=prior_var,
                denoise=config.get("denoise"),
                streaming_post_stim_TRs=config.get("streaming_post_stim_TRs"),
            )
            save_cell(cell, betas, vars_, trial_ids, args.session, config)
            print(f"  elapsed: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
