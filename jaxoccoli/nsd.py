"""NSD validation utilities for comparing predicted and actual fMRI betas.

Provides the analysis primitives for validating TRIBEv2 predictions against
real Natural Scenes Dataset data:

    - ``rdm_from_betas`` — representational dissimilarity matrix (correlation
      or Euclidean distance)
    - ``compare_rdms`` — Spearman rank correlation of upper-triangle RDM
      vectors (the standard RSA comparison metric)
    - ``noise_ceiling_r`` — split-half noise ceiling estimation
    - ``split_half_rdms`` — generate two RDMs from random feature splits
    - ``category_selectivity`` — per-feature d-prime for each category
    - ``upper_triangle`` — extract upper triangle as a flat vector
    - ``load_nsd_betas`` — NIfTI loader for NSD nsdgeneral-masked betas

References
----------
Kriegeskorte N et al. (2008) Representational similarity analysis.
Allen EJ et al. (2022) A massive 7T fMRI dataset. Nature Neuroscience. (NSD)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# RDM computation
# ---------------------------------------------------------------------------

def rdm_from_betas(
    betas: jnp.ndarray,
    metric: str = "correlation",
) -> jnp.ndarray:
    """Compute a representational dissimilarity matrix from trial betas.

    Parameters
    ----------
    betas : (n_trials, n_features) activation patterns
    metric : "correlation" (1 - Pearson r) or "euclidean"

    Returns
    -------
    (n_trials, n_trials) symmetric RDM with zero diagonal.
    """
    if metric == "correlation":
        # Center each trial
        centered = betas - jnp.mean(betas, axis=1, keepdims=True)
        # Normalize
        norms = jnp.sqrt(jnp.sum(centered ** 2, axis=1, keepdims=True))
        norms = jnp.where(norms > 0, norms, 1.0)
        normed = centered / norms
        # Correlation matrix
        r = normed @ normed.T
        rdm = 1.0 - r
    elif metric == "euclidean":
        # Pairwise squared Euclidean
        sq = jnp.sum(betas ** 2, axis=1)
        rdm = sq[:, None] + sq[None, :] - 2.0 * (betas @ betas.T)
        rdm = jnp.sqrt(jnp.maximum(rdm, 0.0))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Zero diagonal and symmetrise
    rdm = rdm - jnp.diag(jnp.diag(rdm))
    rdm = (rdm + rdm.T) / 2.0
    return rdm


# ---------------------------------------------------------------------------
# Upper triangle extraction
# ---------------------------------------------------------------------------

def upper_triangle(mat: jnp.ndarray) -> jnp.ndarray:
    """Extract the strict upper triangle of a square matrix as a flat vector.

    Parameters
    ----------
    mat : (N, N) square matrix

    Returns
    -------
    (N*(N-1)/2,) flat vector of upper-triangle entries.
    """
    n = mat.shape[0]
    idx = jnp.triu_indices(n, k=1)
    return mat[idx]


# ---------------------------------------------------------------------------
# RDM comparison (Spearman)
# ---------------------------------------------------------------------------

def _rank(x: jnp.ndarray) -> jnp.ndarray:
    """Compute ranks of a 1D array (ties get average rank)."""
    order = jnp.argsort(x)
    ranks = jnp.empty_like(x)
    ranks = ranks.at[order].set(jnp.arange(len(x), dtype=x.dtype))
    return ranks


def compare_rdms(rdm1: jnp.ndarray, rdm2: jnp.ndarray) -> jnp.ndarray:
    """Compare two RDMs via Spearman rank correlation of upper triangles.

    This is the standard RSA comparison metric (Kriegeskorte et al. 2008).

    Parameters
    ----------
    rdm1, rdm2 : (N, N) representational dissimilarity matrices

    Returns
    -------
    Scalar Spearman rho in [-1, 1].
    """
    v1 = upper_triangle(rdm1)
    v2 = upper_triangle(rdm2)

    # Rank-transform
    r1 = _rank(v1)
    r2 = _rank(v2)

    # Pearson on ranks = Spearman
    r1c = r1 - jnp.mean(r1)
    r2c = r2 - jnp.mean(r2)
    num = jnp.sum(r1c * r2c)
    den = jnp.sqrt(jnp.sum(r1c ** 2) * jnp.sum(r2c ** 2))
    return num / jnp.where(den > 0, den, 1.0)


# ---------------------------------------------------------------------------
# Split-half RDMs
# ---------------------------------------------------------------------------

def split_half_rdms(
    betas: jnp.ndarray,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split features into two halves and compute an RDM from each.

    Parameters
    ----------
    betas : (n_trials, n_features)
    seed : random seed for the split

    Returns
    -------
    (rdm_half1, rdm_half2), each (n_trials, n_trials)
    """
    n_features = betas.shape[1]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_features)
    mid = n_features // 2

    half1 = betas[:, perm[:mid]]
    half2 = betas[:, perm[mid:]]

    return rdm_from_betas(half1), rdm_from_betas(half2)


# ---------------------------------------------------------------------------
# Noise ceiling
# ---------------------------------------------------------------------------

def noise_ceiling_r(
    betas: jnp.ndarray,
    n_splits: int = 20,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate the noise ceiling of the RDM via split-half reliability.

    The upper bound is the mean correlation between each split-half RDM
    and the full RDM. The lower bound is the mean correlation between
    the two split halves.

    Parameters
    ----------
    betas : (n_trials, n_features)
    n_splits : number of random splits to average
    seed : base random seed

    Returns
    -------
    (lower_bound, upper_bound) as scalar arrays
    """
    full_rdm = rdm_from_betas(betas)
    lowers = []
    uppers = []

    for i in range(n_splits):
        rdm1, rdm2 = split_half_rdms(betas, seed=seed + i)
        # Lower bound: correlation between split halves
        lowers.append(compare_rdms(rdm1, rdm2))
        # Upper bound: mean correlation of each half with full
        r1_full = compare_rdms(rdm1, full_rdm)
        r2_full = compare_rdms(rdm2, full_rdm)
        uppers.append((r1_full + r2_full) / 2.0)

    lower = jnp.mean(jnp.stack(lowers))
    upper = jnp.mean(jnp.stack(uppers))
    return lower, upper


# ---------------------------------------------------------------------------
# Category selectivity
# ---------------------------------------------------------------------------

def category_selectivity(
    betas: jnp.ndarray,
    categories: list[str],
) -> dict[str, jnp.ndarray]:
    """Compute per-feature selectivity (d-prime) for each category.

    d' = (mean_in - mean_out) / pooled_std

    Parameters
    ----------
    betas : (n_trials, n_features)
    categories : list of category labels (length n_trials)

    Returns
    -------
    Dict mapping category name → (n_features,) d-prime array.
    """
    cats = np.array(categories)
    unique_cats = sorted(set(categories))
    result = {}

    for cat in unique_cats:
        in_mask = cats == cat
        out_mask = ~in_mask

        in_betas = betas[in_mask]
        out_betas = betas[out_mask]

        mean_in = jnp.mean(in_betas, axis=0)
        mean_out = jnp.mean(out_betas, axis=0)

        var_in = jnp.var(in_betas, axis=0)
        var_out = jnp.var(out_betas, axis=0)
        n_in = in_betas.shape[0]
        n_out = out_betas.shape[0]

        pooled_std = jnp.sqrt(
            ((n_in - 1) * var_in + (n_out - 1) * var_out)
            / max(n_in + n_out - 2, 1)
        )
        pooled_std = jnp.where(pooled_std > 0, pooled_std, 1.0)

        result[cat] = (mean_in - mean_out) / pooled_std

    return result


# ---------------------------------------------------------------------------
# NSD data loader
# ---------------------------------------------------------------------------

def load_nsd_betas(
    nsd_dir: str | Path,
    subject: str = "subj01",
    sessions: list[str] = ("01", "02", "03"),
    mask_suffix: str = "_nsdgeneral.nii.gz",
) -> Optional[np.ndarray]:
    """Load NSD betas masked by nsdgeneral ROI.

    Parameters
    ----------
    nsd_dir : path to NSD data directory
    subject : subject ID (e.g. "subj01")
    sessions : session numbers to load
    mask_suffix : mask filename suffix

    Returns
    -------
    (n_trials, n_voxels) float32 array, or None if data missing.
    """
    import nibabel as nib

    nsd_dir = Path(nsd_dir)
    mask_path = nsd_dir / f"{subject}{mask_suffix}"
    if not mask_path.exists():
        return None

    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata().flatten() > 0

    all_betas = []
    for ses in sessions:
        beta_path = nsd_dir / subject / f"betas_session{ses}.nii.gz"
        if not beta_path.exists():
            continue
        img = nib.load(str(beta_path))
        data = img.get_fdata()
        flat = data.reshape(-1, data.shape[3])
        masked = flat[mask, :].T  # (n_trials, n_voxels)
        all_betas.append(masked)

    if not all_betas:
        return None
    return np.concatenate(all_betas, axis=0).astype(np.float32)
