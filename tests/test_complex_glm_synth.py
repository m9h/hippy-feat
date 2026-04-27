"""Red-green TDD for Task #23: complex_variant_g_forward SNR regression fix.

The previous synth in `examples/complex_fmri_demo.py` created
magnitude-only signal: `z(t) = (1 + g·task(t)) · exp(i·φ_voxel)`. That
projects task signal into both real and imaginary parts only via the
voxel's static phase offset φ; voxels with φ ≈ π/2 get nearly-zero
real-part signal, etc. Result: complex Variant G appeared to underperform
magnitude OLS (AUC 0.74 vs 0.91), but the synth violated Rowe's complex-
amplitude assumption.

These tests use a Rowe-compatible complex synth where signal lives in
both components consistently — z(t) = (μ + Δ·task(t)) · exp(i·φ_voxel)
with Δ being a complex amplitude. Under this model, complex Variant G
SHOULD beat magnitude OLS.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from jaxoccoli.complex_glm import complex_snr_map, complex_variant_g_forward
from jaxoccoli.realtime import build_lss_design_matrix, make_glover_hrf


def _build_rowe_complex_synth(V_active=200, V_inactive=200, T=180,
                               n_trials=20, tr=1.5, seed=0):
    """Synth where signal is genuinely complex-amplitude (Rowe 2005 model).

    z_v(t) = baseline_v + Δ_v · task_reg(t) + noise_v(t)

    where Δ_v is a complex number per voxel (active voxels: |Δ_v|>0, inactive:
    Δ_v=0). Both real and imaginary parts of z carry the task signal.
    """
    rng = np.random.default_rng(seed)
    V = V_active + V_inactive
    onsets = np.linspace(8.0, T * tr - 16.0, n_trials).astype(np.float32)
    hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    box = np.zeros(T, dtype=np.float32)
    for o in onsets:
        idx = int(round(float(o) / tr))
        if 0 <= idx < T:
            box[idx] = 1.0
    task_reg = np.convolve(box, hrf)[:T]                     # (T,)

    # Random complex baseline per voxel (sets the static phase)
    baseline = (rng.normal(0, 1.0, V) + 1j * rng.normal(0, 1.0, V)).astype(
        np.complex64)

    # Complex amplitude Δ per voxel (active voxels)
    delta = np.zeros(V, dtype=np.complex64)
    delta_real = rng.normal(0.5, 0.15, V_active)
    delta_imag = rng.normal(0.5, 0.15, V_active)
    delta[:V_active] = (delta_real + 1j * delta_imag).astype(np.complex64)

    # Synthetic signal: z_v(t) = baseline_v + Δ_v · task_reg(t)
    z_signal = (baseline[:, None] + delta[:, None] * task_reg[None, :]
                ).astype(np.complex64)

    # i.i.d. complex Gaussian thermal noise
    noise = (0.4 * (rng.normal(size=z_signal.shape) +
                    1j * rng.normal(size=z_signal.shape))).astype(np.complex64)
    z = z_signal + noise

    active = np.zeros(V, dtype=bool)
    active[:V_active] = True
    return z, active, onsets, hrf, tr, T


def _magnitude_ols_t_stat(magnitude, design, probe_col):
    XtX_inv = np.linalg.inv(design.T @ design + 1e-6 * np.eye(design.shape[1]))
    pinv = XtX_inv @ design.T
    betas = magnitude @ pinv.T
    pred = betas @ design.T
    rss = ((magnitude - pred) ** 2).sum(axis=1)
    dof = max(design.shape[0] - design.shape[1], 1)
    sigma2 = rss / dof
    se = np.sqrt(sigma2 * XtX_inv[probe_col, probe_col] + 1e-12)
    return betas[:, probe_col] / se


def _auc(scores, labels):
    from scipy.stats import rankdata
    pos, neg = labels.astype(bool), ~labels.astype(bool)
    np_, nn = int(pos.sum()), int(neg.sum())
    r = rankdata(scores)
    return float((r[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def test_complex_variant_g_beats_magnitude_ols_on_rowe_synth():
    """Under proper Rowe complex-amplitude model, complex Variant G should
    detect activation at AUC ≥ magnitude OLS (typically a small positive
    margin from using both real+imag evidence)."""
    z, active, onsets, hrf, tr, T = _build_rowe_complex_synth(seed=0)
    dm, probe_col = build_lss_design_matrix(onsets, tr, T, hrf, probe_trial=0)

    # Magnitude OLS baseline
    mag = np.abs(z)
    t_mag = _magnitude_ols_t_stat(mag, dm, probe_col)
    auc_mag = _auc(t_mag, active)

    # Complex Variant G
    n_pad = T + 8
    dm_pad = np.zeros((n_pad, dm.shape[1]), dtype=np.float32)
    dm_pad[:T] = dm
    z_pad = np.zeros((z.shape[0], n_pad), dtype=np.complex64)
    z_pad[:, :T] = z
    beta_c, beta_var, _ = complex_variant_g_forward(
        jnp.asarray(dm_pad), jnp.asarray(z_pad),
        jnp.asarray(T, dtype=jnp.int32),
    )
    snr = np.asarray(complex_snr_map(beta_c, beta_var, probe_col=probe_col))
    auc_complex = _auc(snr, active)

    print(f"\n  AUC magnitude OLS      = {auc_mag:.3f}")
    print(f"  AUC complex Variant G  = {auc_complex:.3f}")
    assert auc_complex >= auc_mag - 0.02, (
        f"complex Variant G failed Rowe synth: mag {auc_mag:.3f} vs "
        f"complex {auc_complex:.3f}"
    )


def test_complex_variant_g_returns_complex_betas():
    z, _, onsets, hrf, tr, T = _build_rowe_complex_synth(
        V_active=10, V_inactive=10, T=60, n_trials=8, seed=1)
    dm, probe_col = build_lss_design_matrix(onsets, tr, T, hrf, probe_trial=0)
    n_pad = T + 8
    dm_pad = np.zeros((n_pad, dm.shape[1]), dtype=np.float32)
    dm_pad[:T] = dm
    z_pad = np.zeros((z.shape[0], n_pad), dtype=np.complex64)
    z_pad[:, :T] = z
    beta_c, beta_var, sigma2 = complex_variant_g_forward(
        jnp.asarray(dm_pad), jnp.asarray(z_pad),
        jnp.asarray(T, dtype=jnp.int32),
    )
    beta_c = np.asarray(beta_c)
    beta_var = np.asarray(beta_var)
    assert np.iscomplexobj(beta_c)
    # Variances should be real and non-negative
    assert (beta_var >= 0).all()
    assert beta_c.shape == (z.shape[0], dm.shape[1])
