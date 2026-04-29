"""Real-time per-TR Bayesian decoding for fMRI experiments.

This module is the deployable interface that real-time fMRI frameworks
(rt-cloud, OpenNFT, custom file-watchers) call once per arriving volume.
It maintains a within-run buffer, builds an LSS design matrix on demand,
and emits posterior `(beta_mean, beta_var)` for each completed probe trial
via an AR(1)-prewhitened conjugate Gaussian GLM (Variant G).

Why AR(1) prewhitening:
    BOLD timeseries are autocorrelated. OLS pretends each TR is an
    independent observation, which inflates t-statistics and produces
    miscalibrated beta variances. The closed-form AR(1) prewhitening
    here is JIT-compiled to ~5 ms / TR on Blackwell-class GPUs and
    matches fMRIPrep+Glover retrieval performance on real RT data
    (see hippy-feat Task 2.1 bake-off).

Why posterior variance:
    Variance per (trial, voxel) is the input neurofeedback experiments
    need to confidence-gate their displays. Conventional GLMs throw
    this away; ours surfaces it.

Use:

    from jaxoccoli.realtime import RTPipeline, RTPipelineConfig

    config = RTPipelineConfig(
        tr=1.5,
        mask=mask_3d_bool,            # boolean (X, Y, Z)
        onsets_sec=event_onsets,      # 1-D array of event onsets in seconds
        max_trs=200,                  # pad-to length for static JIT shapes
    )
    pipeline = RTPipeline(config)
    pipeline.precompute()             # warm the JIT once

    for tr_idx, vol_3d in enumerate(stream):
        result = pipeline.on_volume(vol_3d, tr_idx)
        if result is not None:
            # New probe trial finished — variance-aware feedback signal
            beta = result["beta_mean"]
            var  = result["beta_var"]
            ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# HRF + design matrix
# ---------------------------------------------------------------------------

def make_glover_hrf(tr: float, n_trs: int) -> np.ndarray:
    """Canonical Glover (1999) double-gamma HRF, peak-normalized."""
    from scipy.stats import gamma as gamma_dist

    t = np.arange(n_trs) * tr
    a1, a2, b1, b2, c = 6.0, 16.0, 1.0, 1.0, 1.0 / 6.0
    h = gamma_dist.pdf(t, a1, scale=b1) - c * gamma_dist.pdf(t, a2, scale=b2)
    peak = float(np.abs(h).max())
    if peak > 0:
        h = h / peak
    return h.astype(np.float32)


def build_lss_design_matrix(events_onsets: np.ndarray, tr: float, n_trs: int,
                            hrf: np.ndarray, probe_trial: int,
                            include_drift: bool = True
                            ) -> tuple[np.ndarray, int]:
    """LSS design: probe trial gets its own regressor, all others lumped.

    Returns (n_trs, n_regressors) matrix and the column index of the probe
    regressor.
    """
    probe_boxcar = np.zeros(n_trs, dtype=np.float32)
    probe_tr = int(round(events_onsets[probe_trial] / tr))
    if 0 <= probe_tr < n_trs:
        probe_boxcar[probe_tr] = 1.0
    other_boxcar = np.zeros(n_trs, dtype=np.float32)
    for i, onset in enumerate(events_onsets):
        if i == probe_trial:
            continue
        idx = int(round(float(onset) / tr))
        if 0 <= idx < n_trs:
            other_boxcar[idx] = 1.0

    probe_reg = np.convolve(probe_boxcar, hrf)[:n_trs]
    other_reg = np.convolve(other_boxcar, hrf)[:n_trs]
    intercept = np.ones(n_trs, dtype=np.float32)

    cols = [probe_reg, other_reg, intercept]
    if include_drift:
        cols.append(np.cos(2 * np.pi * np.arange(n_trs) / max(n_trs - 1, 1)
                           ).astype(np.float32))

    dm = np.column_stack(cols).astype(np.float32)
    return dm, 0  # probe is column 0


# ---------------------------------------------------------------------------
# JIT-compiled AR(1) conjugate forward (Variant G core)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def _variant_g_forward_jit(X_pad, Y_pad, n_eff,
                           pp_scalar=0.01,
                           rho_prior_mean=0.5,
                           rho_prior_var=0.09,
                           a0=0.01, b0=0.01):
    """AR(1)-prewhitened conjugate Gaussian GLM, padded inputs, vmapped voxels.

    Args:
        X_pad: (T_max, P) design matrix; rows past `n_eff` are zero.
        Y_pad: (V, T_max) voxel data; cols past `n_eff` are zero.
        n_eff: scalar int32 — effective TR count (traced).

    Returns:
        beta_mean: (V, P)
        beta_var:  (V, P) — strictly positive marginal posterior variance
    """
    T, P = X_pad.shape
    pp = pp_scalar * jnp.eye(P)

    XtX = X_pad.T @ X_pad
    XtX_ols = XtX + 1e-6 * jnp.eye(P)
    Xty = X_pad.T @ Y_pad.T
    beta_ols = jnp.linalg.solve(XtX_ols, Xty).T

    resid = Y_pad - beta_ols @ X_pad.T
    r1 = jnp.sum(resid[:, 1:] * resid[:, :-1], axis=1)
    r0 = jnp.sum(resid ** 2, axis=1)
    rho_ols = r1 / (r0 + 1e-10)

    n_eff_f = jnp.maximum(n_eff.astype(jnp.float32), 2.0)
    var_resid = r0 / jnp.maximum(n_eff_f - 1.0, 1.0)
    rho_precision = 1.0 / rho_prior_var
    data_precision = r0 / (var_resid + 1e-10)
    rho = (rho_precision * rho_prior_mean + data_precision * rho_ols) / (
        rho_precision + data_precision
    )
    rho = jnp.clip(rho, -0.99, 0.99)

    def _per_voxel(rho_v, y_v):
        y_pw = y_v[1:] - rho_v * y_v[:-1]
        X_pw = X_pad[1:] - rho_v * X_pad[:-1]
        XtX_pw = X_pw.T @ X_pw
        post_prec = XtX_pw + pp
        post_prec_inv = jnp.linalg.inv(post_prec)
        Xty_pw = X_pw.T @ y_pw
        beta_mean_v = post_prec_inv @ Xty_pw
        resid_pw = y_pw - X_pw @ beta_mean_v
        rss = jnp.sum(resid_pw ** 2)
        a_post = a0 + (n_eff_f - 1.0) / 2.0
        b_post = b0 + 0.5 * rss
        sigma2 = jnp.maximum(b_post / (a_post - 1.0), 1e-10)
        beta_var_v = sigma2 * jnp.diagonal(post_prec_inv)
        return beta_mean_v, beta_var_v

    return jax.vmap(_per_voxel)(rho, Y_pad)


# ---------------------------------------------------------------------------
# RT pipeline state machine
# ---------------------------------------------------------------------------

@dataclass
class RTPipelineConfig:
    """Configuration for a single-run real-time decoding pipeline.

    Attributes:
        tr: repetition time in seconds (e.g. 1.5).
        mask: boolean array of shape (X, Y, Z) marking voxels of interest.
        onsets_sec: 1-D array of event onsets in seconds, run-relative.
        post_stim_window_sec: how long after a probe onset to wait before
            emitting (β, β_var). 12 s captures the full HRF + tail.
        max_trs: pad-to length for static JIT shapes. Should equal the
            longest run duration in TRs you'll see (e.g. 200 for a 5-min
            run at TR=1.5).
        prior_mean: optional (P,) prior mean over regression coefficients.
            P depends on `include_drift`. Defaults to zero.
        prior_var: optional (P,) prior variance. Larger = weaker prior.
        include_drift: whether to add a cosine drift regressor.
    """
    tr: float
    mask: np.ndarray
    onsets_sec: np.ndarray
    post_stim_window_sec: float = 12.0
    max_trs: int = 200
    prior_mean: Optional[np.ndarray] = None
    prior_var: Optional[np.ndarray] = None
    include_drift: bool = True

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        self.onsets_sec = np.asarray(self.onsets_sec, dtype=np.float32)


@dataclass
class RTPipeline:
    """Per-TR Bayesian decoder for a single fMRI run.

    Call `precompute()` once after construction to JIT-warm the forward.
    Then call `on_volume(volume_3d, tr_index)` once per arriving volume.
    Returns `None` until a probe trial completes; then returns a dict with
    posterior `beta_mean`, `beta_var`, and the SNR-thresholded confidence
    mask the experiment can display.
    """
    config: RTPipelineConfig
    _hrf: np.ndarray = field(init=False, default=None, repr=False)
    _buffer: list[np.ndarray] = field(init=False, default_factory=list, repr=False)
    _emitted_trials: set[int] = field(init=False, default_factory=set, repr=False)
    _is_warm: bool = field(init=False, default=False, repr=False)
    _tr_idx_history: list[int] = field(init=False, default_factory=list, repr=False)

    def precompute(self) -> None:
        """Warm the JIT and validate config dimensions."""
        cfg = self.config
        n_hrf_trs = int(np.ceil(32.0 / cfg.tr))
        self._hrf = make_glover_hrf(cfg.tr, n_hrf_trs)

        # Tracer warm-up: run the forward once on dummies to compile the JIT
        n_voxels = int(cfg.mask.sum())
        if n_voxels == 0:
            raise ValueError("RTPipelineConfig.mask has no True voxels")
        dm, _ = build_lss_design_matrix(
            cfg.onsets_sec, cfg.tr, cfg.max_trs, self._hrf,
            probe_trial=0, include_drift=cfg.include_drift,
        )
        # We pad later; for warm-up the design matrix is already at max_trs
        Y_dummy = jnp.zeros((n_voxels, cfg.max_trs), dtype=jnp.float32)
        n_eff = jnp.asarray(cfg.max_trs, dtype=jnp.int32)
        _variant_g_forward_jit(jnp.asarray(dm), Y_dummy, n_eff)
        self._is_warm = True

    @property
    def n_voxels(self) -> int:
        return int(self.config.mask.sum())

    def reset(self) -> None:
        """Drop the buffer between runs."""
        self._buffer.clear()
        self._emitted_trials.clear()
        self._tr_idx_history.clear()

    def on_volume(self, volume_3d: np.ndarray, tr_index: int
                  ) -> Optional[dict]:
        """Push a volume into the buffer and check whether a probe trial has
        completed.

        Returns dict on emission, else None.
        """
        if not self._is_warm:
            self.precompute()

        cfg = self.config
        if volume_3d.shape != cfg.mask.shape:
            raise ValueError(
                f"volume shape {volume_3d.shape} does not match mask "
                f"shape {cfg.mask.shape}"
            )
        masked = volume_3d[cfg.mask].astype(np.float32)
        self._buffer.append(masked)
        self._tr_idx_history.append(tr_index)

        # Find the most recent probe trial that just finished
        cur_t = (tr_index + 1) * cfg.tr
        ready = None
        for trial_i, onset in enumerate(cfg.onsets_sec):
            if trial_i in self._emitted_trials:
                continue
            if onset + cfg.post_stim_window_sec <= cur_t:
                ready = trial_i  # keep walking; emit the latest ready
        if ready is None:
            return None

        n_trs_seen = len(self._buffer)
        Y = np.stack(self._buffer, axis=1)               # (V, n_trs_seen)
        dm, probe_col = build_lss_design_matrix(
            cfg.onsets_sec, cfg.tr, n_trs_seen, self._hrf,
            probe_trial=ready, include_drift=cfg.include_drift,
        )
        # Pad both to max_trs for static JIT shape
        max_t = cfg.max_trs
        if n_trs_seen > max_t:
            raise RuntimeError(
                f"Run has exceeded max_trs={max_t}; bump RTPipelineConfig.max_trs"
            )
        dm_pad = np.zeros((max_t, dm.shape[1]), dtype=np.float32)
        dm_pad[:n_trs_seen] = dm
        Y_pad = np.zeros((Y.shape[0], max_t), dtype=np.float32)
        Y_pad[:, :n_trs_seen] = Y

        n_eff = jnp.asarray(n_trs_seen, dtype=jnp.int32)
        beta_all, var_all = _variant_g_forward_jit(
            jnp.asarray(dm_pad), jnp.asarray(Y_pad), n_eff,
        )
        beta_mean = np.asarray(beta_all[:, probe_col], dtype=np.float32)
        beta_var = np.maximum(
            np.asarray(var_all[:, probe_col], dtype=np.float32), 1e-10
        )

        # Optional shrinkage toward training-data prior
        if cfg.prior_mean is not None and cfg.prior_var is not None:
            pm = np.asarray(cfg.prior_mean, dtype=np.float32)
            pv = np.maximum(np.asarray(cfg.prior_var, dtype=np.float32), 1e-10)
            post_var = 1.0 / (1.0 / pv + 1.0 / beta_var)
            post_mean = post_var * (pm / pv + beta_mean / beta_var)
            beta_mean, beta_var = post_mean, post_var

        snr = np.abs(beta_mean) / np.sqrt(beta_var + 1e-10)
        self._emitted_trials.add(ready)
        return {
            "probe_trial": int(ready),
            "n_trs_used": int(n_trs_seen),
            "beta_mean": beta_mean,
            "beta_var": beta_var,
            "snr": snr,
        }

    @property
    def emitted_trials(self) -> Sequence[int]:
        return tuple(sorted(self._emitted_trials))


def confidence_mask(snr: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Boolean mask of voxels whose |β|/√var exceeds `threshold`.

    Use this to gate the feedback signal: only voxels with reliably non-zero
    posterior beta contribute to the displayed signal. Mirrors
    `hippy-feat/scripts/rt_glm_variants.confidence_mask`.
    """
    return snr > float(threshold)
