"""Prais-Winsten + iterated AR(1) variant of _variant_g_forward.

Differences vs scripts/rt_glm_variants.py::_variant_g_forward:
  1. Prais-Winsten first-row scaling `(1-ρ²)^½` instead of dropping the first sample
     (Cochrane-Orcutt). Keeps T rows.
  2. Iterates ρ estimation (default 3 passes) so ρ converges, matching the way
     nilearn / statsmodels AR(1) noise model fits.
  3. ρ estimator denominator uses lagged-residual variance (`Σ u_{t-1}²`)
     rather than full residual variance (`Σ u_t²`), matching the OLS-on-lagged-
     residuals form nilearn uses.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _variant_g_forward_pw(X_pad, Y_pad, n_eff,
                          pp_scalar: float = 0.0,
                          rho_prior_mean: float = 0.0,
                          rho_prior_var: float = 1e8,
                          a0: float = 0.01, b0: float = 0.01,
                          n_iter: int = 3):
    """AR(1)-prewhitened conjugate GLM with Prais-Winsten whitening + iterated ρ."""
    T, P = X_pad.shape
    pp = pp_scalar * jnp.eye(P)
    n_eff_f = jnp.maximum(n_eff.astype(jnp.float32), 2.0)

    # Initial OLS
    XtX_ridge = X_pad.T @ X_pad + 1e-6 * jnp.eye(P)
    Xty = X_pad.T @ Y_pad.T
    beta = jnp.linalg.solve(XtX_ridge, Xty).T              # (V, P)

    rho_precision = 1.0 / rho_prior_var

    def _estimate_rho(beta_v):
        # Per-voxel residuals in original (un-whitened) scale
        resid = Y_pad - beta_v @ X_pad.T                    # (V, T_max)
        # Use only the first n_eff rows (zero-padded tail otherwise contaminates AC1)
        # ... but we already padded with zeros so r1, r0 only need n_eff-bound sums
        # Build a mask vector
        idx = jnp.arange(T)
        mask_full = (idx < n_eff_f.astype(jnp.int32)).astype(jnp.float32)  # (T,)
        # lagged sums over rows 1..n_eff-1
        u_t = resid[:, 1:] * mask_full[1:]
        u_tm1 = resid[:, :-1] * mask_full[1:]
        r1 = jnp.sum(u_t * u_tm1, axis=1)
        r0 = jnp.sum(u_tm1 ** 2, axis=1)                    # nilearn-style denom
        var_resid = r0 / jnp.maximum(n_eff_f - 1.0, 1.0)
        data_precision = r0 / (var_resid + 1e-10)
        rho_data = r1 / (r0 + 1e-10)
        rho = (rho_precision * rho_prior_mean + data_precision * rho_data) / (
            rho_precision + data_precision
        )
        return jnp.clip(rho, -0.99, 0.99)

    # Iterate ρ + whitened β
    def _refit(carry, _):
        beta_v = carry
        rho = _estimate_rho(beta_v)

        def _per_voxel(rho_v, y_v):
            # Prais-Winsten: scale row 0 by (1-ρ²)^½, then Cochrane-Orcutt for rows 1..
            scale0 = jnp.sqrt(jnp.maximum(1.0 - rho_v ** 2, 1e-10))
            y_pw_head = scale0 * y_v[:1]
            X_pw_head = scale0 * X_pad[:1]
            y_pw_tail = y_v[1:] - rho_v * y_v[:-1]
            X_pw_tail = X_pad[1:] - rho_v * X_pad[:-1]
            y_pw = jnp.concatenate([y_pw_head, y_pw_tail], axis=0)
            X_pw = jnp.concatenate([X_pw_head, X_pw_tail], axis=0)
            XtX_pw = X_pw.T @ X_pw
            post_prec = XtX_pw + pp
            post_prec_inv = jnp.linalg.inv(post_prec)
            Xty_pw = X_pw.T @ y_pw
            return post_prec_inv @ Xty_pw, post_prec_inv

        beta_new, post_prec_inv = jax.vmap(_per_voxel)(rho, Y_pad)
        return beta_new, (rho, post_prec_inv)

    beta, _ = jax.lax.scan(_refit, beta, jnp.arange(n_iter))
    rho_final = _estimate_rho(beta)
    # One more pass to capture final post_prec_inv for variance estimate
    def _per_voxel_final(rho_v, y_v, beta_v):
        scale0 = jnp.sqrt(jnp.maximum(1.0 - rho_v ** 2, 1e-10))
        y_pw_head = scale0 * y_v[:1]
        X_pw_head = scale0 * X_pad[:1]
        y_pw_tail = y_v[1:] - rho_v * y_v[:-1]
        X_pw_tail = X_pad[1:] - rho_v * X_pad[:-1]
        y_pw = jnp.concatenate([y_pw_head, y_pw_tail], axis=0)
        X_pw = jnp.concatenate([X_pw_head, X_pw_tail], axis=0)
        XtX_pw = X_pw.T @ X_pw
        post_prec = XtX_pw + pp
        post_prec_inv = jnp.linalg.inv(post_prec)
        resid_pw = y_pw - X_pw @ beta_v
        rss = jnp.sum(resid_pw ** 2)
        a_post = a0 + (n_eff_f - 1.0) / 2.0
        b_post = b0 + 0.5 * rss
        sigma2 = jnp.maximum(b_post / (a_post - 1.0), 1e-10)
        beta_var_v = sigma2 * jnp.diagonal(post_prec_inv)
        return beta_var_v

    var = jax.vmap(_per_voxel_final)(rho_final, Y_pad, beta)
    return beta, var
