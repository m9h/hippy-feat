"""Bayesian single-trial beta estimation with variance propagation.

Implements vbjax-style factory functions for:
- Conjugate Gaussian GLM (closed-form, real-time compatible)
- Full Bayesian GLM via NUTS (offline, requires blackjax)

Inspired by BROCCOLI's Gibbs sampler (Eklund et al. 2014) but using JAX
autodiff.  Addresses the Rissman/Mumford beta series variance gap by
outputting (beta_mean, beta_var) rather than point estimates.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Conjugate GLM (closed-form, real-time path)
# ---------------------------------------------------------------------------

class ConjugateGLMParams(NamedTuple):
    """Precomputed matrices for conjugate Gaussian GLM."""
    XtX_inv: jnp.ndarray        # (P, P) precomputed (X^T X + prior_prec)^{-1}
    XtX_inv_Xt: jnp.ndarray     # (P, T) precomputed (X^T X + prior_prec)^{-1} X^T
    prior_precision: jnp.ndarray # (P, P) prior precision matrix
    prior_mean: jnp.ndarray      # (P,) prior mean
    a0: float                     # InverseGamma shape prior
    b0: float                     # InverseGamma scale prior
    n_obs: int                    # Number of observations (T)
    n_params: int                 # Number of parameters (P)


def make_conjugate_glm(design_matrix, prior_precision=None, prior_mean=None,
                       a0=0.01, b0=0.01):
    """Factory for conjugate normal-inverse-gamma GLM.

    Computes posterior (beta_mean, beta_var, sigma2) in closed form.
    This is BROCCOLI's Gibbs Block 1 as a single analytic step.

    Model:
        y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)
        beta | sigma^2 ~ N(prior_mean, sigma^2 * prior_precision^{-1})
        sigma^2 ~ InverseGamma(a0, b0)

    Posterior:
        beta | y, sigma^2 ~ N(beta_post, sigma^2 * Omega_post)
        sigma^2 | y ~ InverseGamma(a_post, b_post)

    Args:
        design_matrix: (T, P) design matrix.
        prior_precision: (P, P) prior precision on beta.
            If None, uses weak prior 0.01 * I.
        prior_mean: (P,) prior mean on beta.
            If None, uses zero.
        a0: InverseGamma shape parameter.
        b0: InverseGamma scale parameter.

    Returns:
        params: ConjugateGLMParams (precomputed matrices).
        forward_fn: (params, voxel_data) -> (beta_mean, beta_var, sigma2_mean).
            voxel_data: (T,) single voxel time series.
            beta_mean: (P,) posterior mean.
            beta_var: (P,) posterior marginal variances (diagonal of cov).
            sigma2_mean: scalar posterior mean of noise variance.
    """
    X = design_matrix
    T, P = X.shape

    if prior_precision is None:
        prior_precision = 0.01 * jnp.eye(P)
    if prior_mean is None:
        prior_mean = jnp.zeros(P)

    # Posterior precision: Omega_post^{-1} = X^T X + prior_precision
    XtX = X.T @ X
    post_prec = XtX + prior_precision
    post_prec_inv = jnp.linalg.inv(post_prec)

    # Precompute for speed: (X^T X + Lambda)^{-1} X^T
    XtX_inv_Xt = post_prec_inv @ X.T

    params = ConjugateGLMParams(
        XtX_inv=post_prec_inv,
        XtX_inv_Xt=XtX_inv_Xt,
        prior_precision=prior_precision,
        prior_mean=prior_mean,
        a0=a0,
        b0=b0,
        n_obs=T,
        n_params=P,
    )

    def forward(params, voxel_data):
        # Posterior mean: beta_post = Omega_post (X^T y + Lambda mu_0)
        Xty = X.T @ voxel_data
        prior_contrib = params.prior_precision @ params.prior_mean
        beta_mean = params.XtX_inv @ (Xty + prior_contrib)

        # Posterior sigma^2 (InverseGamma posterior)
        residuals = voxel_data - X @ beta_mean
        rss = jnp.sum(residuals ** 2)
        prior_deviation = beta_mean - params.prior_mean
        prior_quad = prior_deviation @ params.prior_precision @ prior_deviation
        a_post = params.a0 + params.n_obs / 2.0
        b_post = params.b0 + 0.5 * (rss + prior_quad)
        sigma2_mean = b_post / (a_post - 1.0)
        sigma2_mean = jnp.maximum(sigma2_mean, 1e-10)

        # Posterior variance of beta: diag(sigma^2 * Omega_post)
        beta_var = sigma2_mean * jnp.diagonal(params.XtX_inv)

        return beta_mean, beta_var, sigma2_mean

    return params, forward


def make_conjugate_glm_vmap(design_matrix, prior_precision=None,
                            prior_mean=None, a0=0.01, b0=0.01):
    """Vmapped conjugate GLM for whole-brain parallel estimation.

    Same as make_conjugate_glm but returns a forward function that
    operates on all voxels simultaneously.

    Args:
        design_matrix: (T, P) design matrix.
        prior_precision: (P, P) or None.
        prior_mean: (P,) or None.
        a0, b0: InverseGamma priors.

    Returns:
        params: ConjugateGLMParams.
        forward_fn: (params, all_voxels) -> (beta_means, beta_vars, sigma2s).
            all_voxels: (V, T) all voxel time series.
            beta_means: (V, P) posterior means.
            beta_vars: (V, P) posterior variances.
            sigma2s: (V,) noise variance estimates.
    """
    params, single_forward = make_conjugate_glm(
        design_matrix, prior_precision, prior_mean, a0, b0
    )

    def forward_all(params, all_voxels):
        return jax.vmap(lambda y: single_forward(params, y))(all_voxels)

    return params, forward_all


# ---------------------------------------------------------------------------
# AR(1) prewhitening extension
# ---------------------------------------------------------------------------

class ARConjugateParams(NamedTuple):
    """Parameters for AR-prewhitened conjugate GLM."""
    base_params: ConjugateGLMParams
    S_matrices: jnp.ndarray  # (2, 2, P, P) precomputed cross-product matrices


def make_ar1_conjugate_glm(design_matrix, prior_precision=None,
                           prior_mean=None, a0=0.01, b0=0.01,
                           rho_prior_mean=0.5, rho_prior_var=0.09):
    """Factory for AR(1)-prewhitened conjugate GLM.

    Implements a two-step procedure:
    1. Estimate rho from OLS residuals
    2. Prewhiten and apply conjugate GLM

    This is the fast approximation of BROCCOLI's Gibbs sampler.

    Args:
        design_matrix: (T, P) design matrix.
        prior_precision: (P, P) or None.
        prior_mean: (P,) or None.
        a0, b0: InverseGamma priors.
        rho_prior_mean: Prior mean for AR(1) coefficient.
        rho_prior_var: Prior variance for AR(1) coefficient.

    Returns:
        params: ConjugateGLMParams (for the base GLM).
        forward_fn: (params, voxel_data) -> (beta_mean, beta_var, sigma2, rho).
    """
    X = design_matrix
    T, P = X.shape

    base_params, base_forward = make_conjugate_glm(
        X, prior_precision, prior_mean, a0, b0
    )

    # Precompute S matrices for fast prewhitening update
    # S_ij = sum_t x_{t-i}^T x_{t-j}  for i,j in {0, 1}
    S00 = X.T @ X
    S01 = X[1:].T @ X[:-1]
    S11 = X[:-1].T @ X[:-1]

    def forward(params, voxel_data):
        # Step 1: OLS estimate of rho
        beta_ols = jnp.linalg.lstsq(X, voxel_data)[0]
        residuals = voxel_data - X @ beta_ols
        # Yule-Walker estimate
        r1 = jnp.sum(residuals[1:] * residuals[:-1])
        r0 = jnp.sum(residuals ** 2)
        rho_ols = r1 / (r0 + 1e-10)

        # Shrink toward prior
        rho_precision = 1.0 / rho_prior_var
        data_precision = r0 / (jnp.var(residuals) + 1e-10)
        rho = (rho_precision * rho_prior_mean + data_precision * rho_ols) / (
            rho_precision + data_precision
        )
        rho = jnp.clip(rho, -0.99, 0.99)

        # Step 2: Prewhiten
        y_pw = voxel_data[1:] - rho * voxel_data[:-1]
        X_pw = X[1:] - rho * X[:-1]

        # Recompute posterior with prewhitened data
        XtX_pw = S00 - 2 * rho * S01 + rho ** 2 * S11
        if prior_precision is not None:
            pp = prior_precision
        else:
            pp = 0.01 * jnp.eye(P)
        post_prec_pw = XtX_pw + pp
        post_prec_inv_pw = jnp.linalg.inv(post_prec_pw)

        Xty_pw = X_pw.T @ y_pw
        pm = prior_mean if prior_mean is not None else jnp.zeros(P)
        prior_contrib = pp @ pm
        beta_mean = post_prec_inv_pw @ (Xty_pw + prior_contrib)

        # Noise variance
        resid_pw = y_pw - X_pw @ beta_mean
        rss = jnp.sum(resid_pw ** 2)
        a_post = a0 + (T - 1) / 2.0
        b_post = b0 + 0.5 * rss
        sigma2 = b_post / (a_post - 1.0)
        sigma2 = jnp.maximum(sigma2, 1e-10)

        beta_var = sigma2 * jnp.diagonal(post_prec_inv_pw)

        return beta_mean, beta_var, sigma2, rho

    return base_params, forward


# ---------------------------------------------------------------------------
# Full Bayesian GLM (NUTS, offline path)
# ---------------------------------------------------------------------------

class BayesianGLMConfig(NamedTuple):
    """Configuration for full Bayesian GLM."""
    design_matrix: jnp.ndarray   # (T, P)
    ar_order: int                 # AR noise order
    n_warmup: int                 # NUTS warmup steps
    n_samples: int                # NUTS sampling steps


def make_bayesian_glm(design_matrix, ar_order=2, n_warmup=500, n_samples=1000):
    """Factory for full Bayesian GLM via NUTS (requires blackjax).

    This is the offline path for complete posterior sampling. Outputs
    full posterior summaries including beta, AR coefficients, and sigma^2.

    Args:
        design_matrix: (T, P) design matrix.
        ar_order: AR noise model order (default 2).
        n_warmup: NUTS warmup iterations.
        n_samples: NUTS sampling iterations.

    Returns:
        config: BayesianGLMConfig.
        sample_fn: (config, voxel_data, key) -> dict with posterior summaries.
    """
    config = BayesianGLMConfig(
        design_matrix=design_matrix,
        ar_order=ar_order,
        n_warmup=n_warmup,
        n_samples=n_samples,
    )

    def _log_posterior(params, voxel_data, X):
        P = X.shape[1]
        beta = params[:P]
        log_sigma = params[P]
        ar_coeffs = params[P + 1:P + 1 + ar_order]
        sigma = jnp.exp(log_sigma)

        # Predicted signal
        predicted = X @ beta
        residuals = voxel_data - predicted

        # AR(p) likelihood
        T = len(voxel_data)
        ll = 0.0
        for t_idx in range(ar_order, T):
            ar_pred = jnp.sum(
                ar_coeffs * jnp.array([residuals[t_idx - j - 1] for j in range(ar_order)])
            )
            innovation = residuals[t_idx] - ar_pred
            ll = ll - 0.5 * (innovation / sigma) ** 2 - log_sigma

        # Priors
        lp_beta = -0.5 * jnp.sum((beta / 10.0) ** 2)
        lp_sigma = -2.0 * log_sigma  # Jeffrey's prior approx
        lp_ar = -0.5 * jnp.sum((ar_coeffs / 0.5) ** 2)

        return ll + lp_beta + lp_sigma + lp_ar

    def sample_fn(config, voxel_data, key):
        try:
            import blackjax
        except ImportError:
            raise ImportError(
                "blackjax is required for full Bayesian GLM. "
                "Install with: pip install blackjax"
            )

        X = config.design_matrix
        P = X.shape[1]
        n_params = P + 1 + config.ar_order

        def logdensity(params):
            return _log_posterior(params, voxel_data, X)

        # Initialise from OLS
        beta_init = jnp.linalg.lstsq(X, voxel_data)[0]
        resid = voxel_data - X @ beta_init
        log_sigma_init = jnp.log(jnp.std(resid) + 1e-8)
        ar_init = jnp.zeros(config.ar_order)
        init_params = jnp.concatenate([beta_init, jnp.array([log_sigma_init]), ar_init])

        # NUTS setup
        warmup_key, sample_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity, num_steps=config.n_warmup
        )
        (state, kernel_params), _ = warmup.run(warmup_key, init_params)

        kernel = blackjax.nuts(logdensity, **kernel_params)

        def _step(state, key):
            state, info = kernel.step(key, state)
            return state, state.position

        keys = jax.random.split(sample_key, config.n_samples)
        _, samples = jax.lax.scan(_step, state, keys)

        # Summarise
        beta_samples = samples[:, :P]
        log_sigma_samples = samples[:, P]
        ar_samples = samples[:, P + 1:P + 1 + config.ar_order]

        return {
            'beta_mean': jnp.mean(beta_samples, axis=0),
            'beta_var': jnp.var(beta_samples, axis=0),
            'beta_samples': beta_samples,
            'sigma2_mean': jnp.mean(jnp.exp(2 * log_sigma_samples)),
            'ar_coeffs_mean': jnp.mean(ar_samples, axis=0),
        }

    return config, sample_fn
