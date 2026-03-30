"""Differentiable loss functions for connectivity analysis.

Ports essential loss functions from hypercoil loss/functional.py in vbjax
style (pure functions, JIT/vmap/grad compatible, no Equinox).

Includes variance-aware losses for Bayesian beta estimation.
"""

import jax
import jax.numpy as jnp

from .graph import modularity_matrix, graph_laplacian


# ---------------------------------------------------------------------------
# Information-theoretic losses
# ---------------------------------------------------------------------------

def entropy(X, axis=-1, eps=1e-8):
    """Shannon entropy of a probability distribution.

    Args:
        X: (..., K) probability distribution (should sum to 1 along axis).
        axis: Axis over which to compute.
        eps: Small constant for numerical stability.

    Returns:
        (...,) entropy values.
    """
    X = jnp.clip(X, eps, 1.0)
    return -jnp.sum(X * jnp.log(X), axis=axis)


def kl_divergence(P, Q, axis=-1, eps=1e-8):
    """KL divergence KL(P || Q).

    Args:
        P: (..., K) true distribution.
        Q: (..., K) approximate distribution.
        axis: Axis over which to compute.
        eps: Small constant.

    Returns:
        (...,) KL divergence values.
    """
    P = jnp.clip(P, eps, 1.0)
    Q = jnp.clip(Q, eps, 1.0)
    return jnp.sum(P * (jnp.log(P) - jnp.log(Q)), axis=axis)


def js_divergence(P, Q, axis=-1, eps=1e-8):
    """Jensen-Shannon divergence (symmetric KL).

    Args:
        P: (..., K) distribution.
        Q: (..., K) distribution.
        axis: Axis over which to compute.
        eps: Small constant.

    Returns:
        (...,) JS divergence values.
    """
    M = (P + Q) / 2.0
    return (kl_divergence(P, M, axis=axis, eps=eps) +
            kl_divergence(Q, M, axis=axis, eps=eps)) / 2.0


# ---------------------------------------------------------------------------
# Network / community losses
# ---------------------------------------------------------------------------

def modularity_loss(A, C, gamma=1.0):
    """Negative relaxed modularity (for minimisation).

    Minimising this loss maximises modularity of the soft partition C.

    Args:
        A: (N, N) adjacency matrix.
        C: (N, K) soft community assignment (e.g. softmax output).
        gamma: Resolution parameter.

    Returns:
        Scalar loss (negative modularity).
    """
    B = modularity_matrix(A, gamma=gamma)
    n = B.shape[-1]
    B = B * (1.0 - jnp.eye(n))
    m2 = jnp.sum(A)
    m2 = jnp.where(m2 == 0, 1.0, m2)
    Q = jnp.trace(C.T @ B @ C) / m2
    return -Q


def connectopy_loss(Q, A, dissimilarity=None):
    """Connectopic functional loss.

    Encourages embedding coordinates Q to be smooth over the connectivity
    graph: connected nodes should have similar coordinates.

    Args:
        Q: (N, K) embedding coordinates.
        A: (N, N) weighted adjacency / affinity matrix.
        dissimilarity: Optional (N, N) dissimilarity matrix.
            If None, uses squared Euclidean distance between Q rows.

    Returns:
        Scalar loss.
    """
    if dissimilarity is None:
        diff = Q[:, None, :] - Q[None, :, :]  # (N, N, K)
        dissimilarity = jnp.sum(diff ** 2, axis=-1)  # (N, N)
    return jnp.sum(A * dissimilarity) / jnp.sum(A)


def eigenmaps_loss(Q, A, normalise=True):
    """Laplacian eigenmaps loss (Rayleigh quotient).

    L_loss = tr(Q^T L Q) / tr(Q^T D Q)

    where L is the graph Laplacian and D is the degree matrix.

    Args:
        Q: (N, K) embedding coordinates.
        A: (N, N) adjacency matrix.
        normalise: If True, normalise by degree.

    Returns:
        Scalar loss.
    """
    L = graph_laplacian(A, normalise=False)
    numerator = jnp.trace(Q.T @ L @ Q)
    if normalise:
        d = jnp.sum(A, axis=-1)
        D = jnp.diag(d)
        denominator = jnp.trace(Q.T @ D @ Q)
        denominator = jnp.where(denominator == 0, 1.0, denominator)
        return numerator / denominator
    return numerator


# ---------------------------------------------------------------------------
# Spatial constraint losses (for learnable parcellation)
# ---------------------------------------------------------------------------

def compactness_loss(assignment, coor, radius=100.0):
    """Parcel compactness: weighted distance from center of mass.

    Encourages parcels to be spatially compact.

    Args:
        assignment: (N, K) soft assignment weights.
        coor: (N, D) spatial coordinates of nodes/voxels.
        radius: Normalisation radius.

    Returns:
        Scalar compactness loss.
    """
    # Center of mass per parcel: (K, D)
    w = assignment / (jnp.sum(assignment, axis=0, keepdims=True) + 1e-8)
    com = w.T @ coor  # (K, D)

    # Weighted distance from COM
    diff = coor[:, None, :] - com[None, :, :]  # (N, K, D)
    dist2 = jnp.sum(diff ** 2, axis=-1)  # (N, K)
    weighted_dist = jnp.sum(assignment * dist2) / (radius ** 2)
    return weighted_dist / assignment.shape[1]


def dispersion_loss(assignment, coor):
    """Mutual separation: negative mean pairwise distance between parcel centers.

    Encourages parcels to spread out.

    Args:
        assignment: (N, K) soft assignment weights.
        coor: (N, D) spatial coordinates.

    Returns:
        Scalar dispersion loss (negative = minimise to maximise spread).
    """
    w = assignment / (jnp.sum(assignment, axis=0, keepdims=True) + 1e-8)
    com = w.T @ coor  # (K, D)
    diff = com[:, None, :] - com[None, :, :]  # (K, K, D)
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # (K, K)
    # Mean off-diagonal distance
    K = com.shape[0]
    mask = 1.0 - jnp.eye(K)
    mean_dist = jnp.sum(dist * mask) / (K * (K - 1) + 1e-8)
    return -mean_dist


def reference_tether_loss(assignment, coor, ref_coor, radius=100.0):
    """Distance of parcel centers from reference coordinates.

    Args:
        assignment: (N, K) soft assignment weights.
        coor: (N, D) spatial coordinates.
        ref_coor: (K, D) reference coordinates for each parcel.
        radius: Normalisation radius.

    Returns:
        Scalar tether loss.
    """
    w = assignment / (jnp.sum(assignment, axis=0, keepdims=True) + 1e-8)
    com = w.T @ coor  # (K, D)
    dist2 = jnp.sum((com - ref_coor) ** 2, axis=-1)  # (K,)
    return jnp.mean(dist2) / (radius ** 2)


# ---------------------------------------------------------------------------
# Regularisation losses
# ---------------------------------------------------------------------------

def smoothness_loss(X, n=1, axis=-1):
    """Penalises large temporal or spatial gradients (n-th order differences).

    Args:
        X: Input array.
        n: Order of differences.
        axis: Axis along which to compute.

    Returns:
        Scalar smoothness loss.
    """
    diff = X
    for _ in range(n):
        diff = jnp.diff(diff, axis=axis)
    return jnp.mean(diff ** 2)


def equilibrium_loss(X, axis=-1):
    """Penalises unequal parcel sizes / weight imbalance.

    Entropy-based: maximum when all weights are equal.

    Args:
        X: (..., K) assignment weights (should be positive).
        axis: Axis over which to measure balance.

    Returns:
        Scalar loss (negative entropy = minimise for balance).
    """
    sums = jnp.sum(X, axis=0)
    p = sums / (jnp.sum(sums) + 1e-8)
    return -entropy(p)


# ---------------------------------------------------------------------------
# Quality control losses
# ---------------------------------------------------------------------------

def qcfc_loss(fc, qc):
    """QC-FC correlation: motion-connectivity confound.

    Measures the correlation between functional connectivity edges
    and a quality control metric (e.g. mean framewise displacement).

    Args:
        fc: (S, E) functional connectivity edges for S subjects.
        qc: (S,) quality control metric per subject.

    Returns:
        Scalar mean absolute QC-FC correlation.
    """
    qc_centered = qc - jnp.mean(qc)
    fc_centered = fc - jnp.mean(fc, axis=0, keepdims=True)
    qc_std = jnp.std(qc) + 1e-8
    fc_std = jnp.std(fc, axis=0) + 1e-8
    r = jnp.mean(fc_centered * qc_centered[:, None], axis=0) / (qc_std * fc_std)
    return jnp.mean(jnp.abs(r))


def multivariate_kurtosis(ts, l2=0.0):
    """Mardia's multivariate kurtosis for non-Gaussianity detection.

    Args:
        ts: (C, T) multivariate time series.
        l2: L2 regularisation on covariance.

    Returns:
        Scalar kurtosis statistic.
    """
    C, T = ts.shape
    mean = jnp.mean(ts, axis=1, keepdims=True)
    centered = ts - mean
    S = (centered @ centered.T) / (T - 1) + l2 * jnp.eye(C)
    S_inv = jnp.linalg.inv(S)
    # Mahalanobis distances: d_i^2 = (x_i - mu)^T S^{-1} (x_i - mu)
    d2 = jnp.einsum('ct,cd,dt->t', centered, S_inv, centered)
    return jnp.mean(d2 ** 2)


# ---------------------------------------------------------------------------
# Variance-aware losses (Bayesian beta estimation)
# ---------------------------------------------------------------------------

def expected_modularity_loss(beta_mean, beta_var, atlas_forward, atlas_params,
                             C, gamma=1.0):
    """Modularity computed under posterior uncertainty over betas.

    Uses the variance-aware atlas to propagate uncertainty, then computes
    modularity on the expected connectivity.

    Args:
        beta_mean: (V, T) posterior mean betas.
        beta_var: (V, T) posterior variance betas.
        atlas_forward: Forward function from make_atlas_linear_uncertain.
        atlas_params: AtlasParams.
        C: (P, K) community assignment for parcels.
        gamma: Resolution parameter.

    Returns:
        Scalar expected modularity loss.
    """
    from .covariance import posterior_corr
    parc_mean, parc_var = atlas_forward(atlas_params, beta_mean, beta_var)
    fc = posterior_corr(parc_mean, parc_var)
    return modularity_loss(fc, C, gamma=gamma)


def reliability_weighted_loss(loss_values, weights):
    """Weighted mean of per-trial loss values by reliability.

    Generic wrapper to downweight contributions from unreliable trials.

    Args:
        loss_values: (T,) per-trial loss values.
        weights: (T,) reliability weights (e.g. 1/beta_std).

    Returns:
        Scalar weighted mean loss.
    """
    w = jnp.abs(weights)
    w = w / (jnp.sum(w) + 1e-8)
    return jnp.sum(loss_values * w)
