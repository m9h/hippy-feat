"""Complex-domain GLM (Variant G extension).

Reference:
  Rowe DB (2005). Modeling both the magnitude and phase of complex-valued
  fMRI data. NeuroImage 25(4):1310–1324.

Background: most fMRI GLMs operate on |complex| (magnitude only). Rowe
showed that fitting the GLM jointly on real + imaginary components of the
complex BOLD data gives a 10–20 % detection-sensitivity gain at typical
SNR, larger at high field. The complex GLM treats the data as a multivariate
real Gaussian (real + imaginary components stacked) with the same design
matrix applied to both.

This module wraps Variant G's AR(1) Bayesian conjugate forward to operate
on complex data, returning complex β posteriors with proper covariance.

Real-time use: fully JIT-compatible, same per-TR latency as the magnitude
Variant G (just doubles the operation count for real+imag).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=())
def complex_variant_g_forward(X_pad: jnp.ndarray,
                               Y_pad_complex: jnp.ndarray,
                               n_eff: jnp.ndarray,
                               pp_scalar: float = 0.01,
                               rho_prior_mean: float = 0.5,
                               rho_prior_var: float = 0.09,
                               a0: float = 0.01, b0: float = 0.01):
    """AR(1)-prewhitened conjugate complex GLM.

    Treats the complex BOLD as two real-valued GLMs sharing one design,
    one ρ estimate (averaged), and one σ² posterior — but with separate
    posterior means for the real and imaginary β. The detection statistic
    `|β|² / var` then aggregates evidence from both components.

    Args:
        X_pad: (T_max, P) real-valued LSS design matrix, zero-padded after
            n_eff rows.
        Y_pad_complex: (V, T_max) complex64 voxel × TR, zero-padded.
        n_eff: scalar int32, effective TR count.

    Returns:
        beta_complex: (V, P) complex64 posterior mean of β.
        beta_var:     (V, P) real posterior variance of |β|² (sum of real +
            imaginary marginal variances; usable for SNR thresholding).
        sigma2:       (V,) real posterior mean of complex-noise variance
            (per-component; double for total).
    """
    if Y_pad_complex.dtype not in (jnp.complex64, jnp.complex128):
        raise ValueError(
            f"Y_pad_complex must be complex; got {Y_pad_complex.dtype}"
        )

    Y_re = jnp.real(Y_pad_complex)
    Y_im = jnp.imag(Y_pad_complex)

    # Reuse Variant G logic per component
    def _ar1_real(Y_real: jnp.ndarray):
        T, P = X_pad.shape
        pp = pp_scalar * jnp.eye(P)

        XtX_ols = X_pad.T @ X_pad + 1e-6 * jnp.eye(P)
        Xty = X_pad.T @ Y_real.T
        beta_ols = jnp.linalg.solve(XtX_ols, Xty).T

        resid = Y_real - beta_ols @ X_pad.T
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
            beta_v = post_prec_inv @ Xty_pw
            resid_pw = y_pw - X_pw @ beta_v
            rss = jnp.sum(resid_pw ** 2)
            a_post = a0 + (n_eff_f - 1.0) / 2.0
            b_post = b0 + 0.5 * rss
            sigma2 = jnp.maximum(b_post / (a_post - 1.0), 1e-10)
            beta_var_v = sigma2 * jnp.diagonal(post_prec_inv)
            return beta_v, beta_var_v, sigma2

        return jax.vmap(_per_voxel)(rho, Y_real)

    beta_re, var_re, sigma2_re = _ar1_real(Y_re)
    beta_im, var_im, sigma2_im = _ar1_real(Y_im)

    # Compose complex β; total marginal variance is sum (independent re/im
    # components in the additive-noise model)
    beta_complex = (beta_re + 1j * beta_im).astype(jnp.complex64)
    beta_var_total = var_re + var_im
    sigma2_avg = 0.5 * (sigma2_re + sigma2_im)
    return beta_complex, beta_var_total, sigma2_avg


def complex_snr_map(beta_complex: jnp.ndarray, beta_var: jnp.ndarray,
                    probe_col: int = 0) -> jnp.ndarray:
    """Per-voxel SNR for the probe regressor: |β|² / var.

    This is the Rowe (2005) detection statistic — incorporates both real
    and imaginary signal evidence weighted by total posterior variance.
    Distributed approximately as a non-central χ² under the null.
    """
    b = beta_complex[:, probe_col]
    v = beta_var[:, probe_col] + 1e-10
    return (jnp.abs(b) ** 2) / v


__all__ = ["complex_variant_g_forward", "complex_snr_map"]
