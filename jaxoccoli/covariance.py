"""Differentiable covariance estimation and variance-aware connectivity.

Ports essential algorithms from hypercoil functional/cov.py in vbjax style
(pure functions, JIT/vmap/grad compatible, no Equinox).

Includes variance-aware extensions for beta series correlation analysis
that address the Rissman/Mumford variance propagation gap.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core covariance estimation
# ---------------------------------------------------------------------------

def cov(X, rowvar=True, bias=False, weight=None, l2=0.0):
    """Empirical covariance with optional observation weights and L2 regularisation.

    Args:
        X: (..., C, T) if rowvar else (..., T, C).  C = variables, T = observations.
        rowvar: If True, each row is a variable.
        bias: If True, normalise by N; otherwise N-1.
        weight: Optional (T,) observation weights.
        l2: L2 (Tikhonov) regularisation added to diagonal.

    Returns:
        (..., C, C) covariance matrix.
    """
    if not rowvar:
        X = jnp.swapaxes(X, -2, -1)

    if weight is not None:
        w = weight / jnp.sum(weight)
        mean = jnp.einsum('...ct,t->...c', X, w)[..., None]
    else:
        mean = jnp.mean(X, axis=-1, keepdims=True)

    centered = X - mean

    if weight is not None:
        w_sqrt = jnp.sqrt(w)
        centered_w = centered * w_sqrt[None, :]
        S = centered_w @ centered_w.swapaxes(-2, -1)
        if not bias:
            w2 = jnp.sum(w ** 2)
            S = S / (1.0 - w2)
    else:
        n = X.shape[-1]
        ddof = 0 if bias else 1
        S = (centered @ centered.swapaxes(-2, -1)) / (n - ddof)

    if l2 > 0:
        eye = jnp.eye(S.shape[-1])
        S = S + l2 * eye

    return S


def corr(X, rowvar=True, weight=None, l2=0.0):
    """Pearson correlation matrix.

    Args:
        X: (..., C, T) if rowvar else (..., T, C).
        rowvar: If True, each row is a variable.
        weight: Optional (T,) observation weights.
        l2: L2 regularisation on covariance before normalisation.

    Returns:
        (..., C, C) correlation matrix.
    """
    S = cov(X, rowvar=rowvar, weight=weight, l2=l2)
    d = jnp.sqrt(jnp.diagonal(S, axis1=-2, axis2=-1))
    d = jnp.where(d == 0, 1.0, d)
    return S / (d[..., :, None] * d[..., None, :])


def precision(X, rowvar=True, weight=None, l2=0.0):
    """Precision (inverse covariance) matrix.

    Args:
        X: (..., C, T) if rowvar else (..., T, C).
        rowvar: If True, each row is a variable.
        weight: Optional (T,) observation weights.
        l2: L2 regularisation (ensures invertibility).

    Returns:
        (..., C, C) precision matrix.
    """
    S = cov(X, rowvar=rowvar, weight=weight, l2=l2)
    return jnp.linalg.inv(S)


def partial_corr(X, rowvar=True, weight=None, l2=1e-6):
    """Partial correlation via the precision matrix.

    partial_corr(i,j) = -P(i,j) / sqrt(P(i,i) * P(j,j))

    Args:
        X: (..., C, T) if rowvar else (..., T, C).
        rowvar: If True, each row is a variable.
        weight: Optional (T,) observation weights.
        l2: L2 regularisation (default 1e-6 for numerical stability).

    Returns:
        (..., C, C) partial correlation matrix.
    """
    P = precision(X, rowvar=rowvar, weight=weight, l2=l2)
    d = jnp.sqrt(jnp.diagonal(P, axis1=-2, axis2=-1))
    d = jnp.where(d == 0, 1.0, d)
    pcorr = -P / (d[..., :, None] * d[..., None, :])
    # Diagonal should be 1 (partial correlation of a variable with itself)
    eye = jnp.eye(pcorr.shape[-1])
    return pcorr * (1.0 - eye) + eye


def partial_cov(X, rowvar=True, weight=None, l2=1e-6):
    """Partial covariance: diagonal entries of the inverse precision.

    Equivalent to the conditional variance of each variable given all others.

    Args:
        X: (..., C, T) if rowvar else (..., T, C).
        rowvar: If True, each row is a variable.
        weight: Optional (T,) observation weights.
        l2: L2 regularisation.

    Returns:
        (..., C, C) partial covariance matrix.
    """
    P = precision(X, rowvar=rowvar, weight=weight, l2=l2)
    d = jnp.diagonal(P, axis1=-2, axis2=-1)
    d = jnp.where(d == 0, 1.0, d)
    return -P / (d[..., :, None] * d[..., None, :]) / d[..., None, :]


def paired_cov(X, Y, rowvar=True, bias=False, weight=None):
    """Cross-covariance between two variable sets.

    Args:
        X: (..., Cx, T) if rowvar.
        Y: (..., Cy, T) if rowvar.
        rowvar: If True, each row is a variable.
        bias: If True, normalise by N; otherwise N-1.
        weight: Optional (T,) observation weights.

    Returns:
        (..., Cx, Cy) cross-covariance matrix.
    """
    if not rowvar:
        X = jnp.swapaxes(X, -2, -1)
        Y = jnp.swapaxes(Y, -2, -1)

    if weight is not None:
        w = weight / jnp.sum(weight)
        mean_x = jnp.einsum('...ct,t->...c', X, w)[..., None]
        mean_y = jnp.einsum('...ct,t->...c', Y, w)[..., None]
    else:
        mean_x = jnp.mean(X, axis=-1, keepdims=True)
        mean_y = jnp.mean(Y, axis=-1, keepdims=True)

    cx = X - mean_x
    cy = Y - mean_y

    if weight is not None:
        w_sqrt = jnp.sqrt(w)
        S = (cx * w_sqrt) @ (cy * w_sqrt).swapaxes(-2, -1)
        if not bias:
            w2 = jnp.sum(w ** 2)
            S = S / (1.0 - w2)
    else:
        n = X.shape[-1]
        ddof = 0 if bias else 1
        S = (cx @ cy.swapaxes(-2, -1)) / (n - ddof)

    return S


def conditional_cov(X, Y, rowvar=True, weight=None, l2=1e-6):
    """Conditional covariance of X given Y via Schur complement.

    Cov(X|Y) = Cov(X) - Cov(X,Y) Cov(Y)^{-1} Cov(Y,X)

    Args:
        X: (..., Cx, T) if rowvar.
        Y: (..., Cy, T) if rowvar.
        rowvar: If True, each row is a variable.
        weight: Optional (T,) observation weights.
        l2: L2 regularisation on Cov(Y) for invertibility.

    Returns:
        (..., Cx, Cx) conditional covariance matrix.
    """
    Sxx = cov(X, rowvar=rowvar, weight=weight)
    Syy = cov(Y, rowvar=rowvar, weight=weight, l2=l2)
    Sxy = paired_cov(X, Y, rowvar=rowvar, weight=weight)
    Syy_inv = jnp.linalg.inv(Syy)
    return Sxx - Sxy @ Syy_inv @ Sxy.swapaxes(-2, -1)


# ---------------------------------------------------------------------------
# Variance-aware extensions (addresses Rissman/Mumford beta series gap)
# ---------------------------------------------------------------------------

def weighted_corr(X, weights, rowvar=True):
    """Reliability-weighted Pearson correlation.

    Downweights noisy observations (e.g. trials with high beta SE).
    Use weights = 1 / beta_std from a Bayesian GLM.

    Args:
        X: (..., C, T) if rowvar.  T = trials.
        weights: (T,) positive reliability weights per observation.
        rowvar: If True, each row is a variable.

    Returns:
        (..., C, C) weighted correlation matrix.
    """
    return corr(X, rowvar=rowvar, weight=weights)


def attenuated_corr(X, reliabilities, rowvar=True):
    """Disattenuated correlation correcting for measurement error.

    Applies Spearman (1904) correction for attenuation:
        r_corrected(i,j) = r_observed(i,j) / sqrt(rel_i * rel_j)

    where reliability = var(true) / var(observed) in [0, 1].

    Args:
        X: (..., C, T) if rowvar.
        reliabilities: (C,) reliability coefficient per variable in (0, 1].
        rowvar: If True, each row is a variable.

    Returns:
        (..., C, C) disattenuated correlation matrix.
    """
    r = corr(X, rowvar=rowvar)
    rel = jnp.clip(reliabilities, 1e-8, 1.0)
    correction = jnp.sqrt(rel[..., :, None] * rel[..., None, :])
    r_corrected = r / correction
    # Clip to valid correlation range
    r_corrected = jnp.clip(r_corrected, -1.0, 1.0)
    # Restore diagonal
    eye = jnp.eye(r.shape[-1])
    return r_corrected * (1.0 - eye) + eye


def posterior_corr(beta_mean, beta_var):
    """Correlation from joint posterior, marginalising over beta uncertainty.

    Given independent posterior distributions N(mu_i, sigma_i^2) per trial
    per region, computes the expected correlation under the posterior.

    For independent posteriors (diagonal covariance per trial):
        E[corr(X_i, X_j)] ≈ corr(mu_i, mu_j) * attenuation_factor

    where the attenuation factor accounts for posterior variance relative
    to the variance of the means across trials.

    Args:
        beta_mean: (..., C, T) posterior means.  C = regions, T = trials.
        beta_var: (..., C, T) posterior variances (diagonal).

    Returns:
        (..., C, C) posterior-marginalised correlation matrix.
    """
    # Observed correlation of posterior means
    r_obs = corr(beta_mean, rowvar=True)

    # Per-variable: variance of means vs mean of variances
    var_of_means = jnp.var(beta_mean, axis=-1)  # (..., C)
    mean_of_vars = jnp.mean(beta_var, axis=-1)  # (..., C)

    # Reliability: proportion of observed variance that is signal
    # rel_i = var(mu_i) / (var(mu_i) + E[sigma_i^2])
    total_var = var_of_means + mean_of_vars
    rel = jnp.where(total_var > 0, var_of_means / total_var, 1.0)

    # Disattenuation
    correction = jnp.sqrt(rel[..., :, None] * rel[..., None, :])
    correction = jnp.where(correction > 0, correction, 1.0)
    r_corrected = r_obs / correction
    r_corrected = jnp.clip(r_corrected, -1.0, 1.0)

    eye = jnp.eye(r_corrected.shape[-1])
    return r_corrected * (1.0 - eye) + eye
