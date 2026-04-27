"""aCompCor / tCompCor — anatomical / temporal component-based noise correction.

Reference:
  Behzadi Y, Restom K, Liau J, Liu TT (2007). A component based noise
  correction method (CompCor) for BOLD and perfusion based fMRI.
  NeuroImage 37(1):90–101.

Two flavors:
  - aCompCor: PCA on voxels in a CSF + white-matter MASK (anatomical), then
    add the top-K components as nuisance regressors. Voxels in CSF/WM
    shouldn't carry task signal, so any structured variance there is noise.
  - tCompCor: PCA on the top-N% highest-temporal-variance voxels (no
    anatomical mask required). Useful when CSF/WM masks aren't available.

Both are simple: PCA on a specified noise pool, take top-K components,
add to the GLM design matrix. Standard in modern fMRI pipelines (fMRIPrep
exposes both).

Real-time use: components computed offline from a calibration block
(recognition run) become FIXED nuisance regressors during feedback runs.
For pure online aCompCor, components could be updated incrementally via
streaming PCA — out of scope for v1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def acompcor_components(bold: np.ndarray, mask_csf_wm: np.ndarray,
                         n_components: int = 5,
                         high_pass_TR: int | None = None,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """aCompCor: top-K PCA components from CSF + WM masked timeseries.

    Args:
        bold: (V, T) full-brain (or whole-volume-flattened) timeseries.
        mask_csf_wm: (V,) boolean — True where voxel is CSF or WM.
        n_components: K, number of components to return.
        high_pass_TR: optional cosine-drift high-pass cutoff (in TRs).
            Voxels in CSF/WM have strong low-frequency drift; remove it
            before PCA so the top components capture finer noise structure.

    Returns:
        components: (T, K) top-K temporal noise components, unit-norm.
        explained_var: (K,) singular-value² / total variance fraction.
    """
    if mask_csf_wm.dtype != bool:
        mask_csf_wm = mask_csf_wm.astype(bool)
    Y = bold[mask_csf_wm].astype(np.float32)                      # (V_pool, T)
    if Y.shape[0] < n_components:
        raise ValueError(
            f"CSF+WM mask has only {Y.shape[0]} voxels; need ≥ {n_components}"
        )
    # Mean-center per voxel
    Y = Y - Y.mean(axis=1, keepdims=True)
    # Optional high-pass filter via cosine-drift regression
    if high_pass_TR is not None:
        T = Y.shape[1]
        n_basis = max(1, int(np.floor(T / max(high_pass_TR, 2))))
        cosine = np.stack([
            np.cos(np.pi * (k + 1) * np.arange(T) / T)
            for k in range(n_basis)
        ], axis=1).astype(np.float32)                              # (T, n_basis)
        # Regress out cosine basis from each pool voxel
        XtX_inv = np.linalg.inv(cosine.T @ cosine + 1e-8 * np.eye(n_basis))
        coefs = Y @ cosine @ XtX_inv.T                             # (V_pool, n_basis)
        Y = Y - coefs @ cosine.T
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    K = min(n_components, len(S))
    components = Vt[:K].T.astype(np.float32)                       # (T, K)
    explained_var = ((S[:K] ** 2) / (S ** 2).sum()).astype(np.float32)
    return components, explained_var


def tcompcor_components(bold: np.ndarray, n_components: int = 5,
                         top_variance_frac: float = 0.02,
                         high_pass_TR: int | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tCompCor: PCA over the top-N% highest-temporal-variance voxels.

    Useful when CSF/WM anatomical masks aren't available — the assumption
    is that the highest-variance voxels are dominated by physiological /
    motion / noise rather than task signal.

    Args:
        bold: (V, T) timeseries.
        n_components: K, number of components.
        top_variance_frac: fraction of voxels to use as the noise pool
            (default 2 %; standard in fMRIPrep is 5 %).
        high_pass_TR: see `acompcor_components`.

    Returns:
        components: (T, K).
        explained_var: (K,).
        pool_mask: (V,) bool — which voxels were used.
    """
    V, T = bold.shape
    voxel_var = bold.var(axis=1)
    n_pool = max(int(np.floor(V * top_variance_frac)), n_components)
    cutoff = np.partition(voxel_var, V - n_pool)[V - n_pool]
    pool_mask = voxel_var >= cutoff
    components, explained_var = acompcor_components(
        bold, pool_mask, n_components=n_components, high_pass_TR=high_pass_TR
    )
    return components, explained_var, pool_mask


def append_compcor_to_design(design: np.ndarray, components: np.ndarray
                              ) -> np.ndarray:
    """Stack CompCor components onto an existing design matrix as columns."""
    if design.shape[0] != components.shape[0]:
        raise ValueError(
            f"design.T={design.shape[0]} != components.T={components.shape[0]}"
        )
    return np.concatenate([design, components], axis=1)


__all__ = [
    "acompcor_components",
    "tcompcor_components",
    "append_compcor_to_design",
]
