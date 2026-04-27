"""Phase regression for physiological-noise removal.

Reference:
  Hagberg GE, Bianciardi M, Maraviglia B (2008). Challenges for detection
  of neuronal currents by MRI. Magn Reson Imaging 26:1192–1201.
  Petridou N, Schäfer A, Gowland P, Bowtell R (2009). Phase vs. magnitude
  information in functional magnetic resonance imaging time series. NeuroImage.

Core idea: at typical field strengths (3 T, 7 T), the phase of a voxel's
fMRI signal carries B0-modulation effects from breathing and cardiac
pulsation. These are largely orthogonal to BOLD task signal in magnitude.
Subtracting per-voxel phase as a confound regressor before fitting the
task GLM removes physiological noise WITHOUT needing external pulse-ox
or respiration-belt recordings.

This complements (does not replace) RETROICOR / aCompCor: it captures
B0-mediated physiology that those methods can miss.

Real-time use: per-voxel linear regression is O(P²) per voxel per TR
when augmented into the GLM design matrix. For RT we recommend the
"phase as additional regressor" path rather than the "phase-corrected
BOLD" path, since the latter requires re-fitting whenever new TRs arrive.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def phase_regress_residuals(magnitude: jnp.ndarray,
                            phase: jnp.ndarray,
                            ) -> jnp.ndarray:
    """Per-voxel: subtract the linear-fit-of-phase from magnitude.

    For each voxel:
        magnitude[t] = α + β · phase[t] + ε[t]
        residual[t]  = magnitude[t] − (α̂ + β̂ · phase[t])

    Args:
        magnitude: (V, T) real BOLD magnitude.
        phase: (V, T) phase timeseries (already detrended/unwrapped).

    Returns:
        (V, T) residuals — magnitude with per-voxel phase contribution removed.
    """
    if magnitude.shape != phase.shape:
        raise ValueError(
            f"magnitude {magnitude.shape} != phase {phase.shape}"
        )
    V, T = magnitude.shape

    # Build per-voxel design matrix [1, phase[v]] and OLS fit per voxel.
    # vmap over voxels.
    def fit_one(mag_v: jnp.ndarray, phase_v: jnp.ndarray) -> jnp.ndarray:
        X = jnp.stack([jnp.ones_like(phase_v), phase_v], axis=1)  # (T, 2)
        XtX = X.T @ X
        XtX_inv = jnp.linalg.inv(XtX + 1e-8 * jnp.eye(2))
        coefs = XtX_inv @ X.T @ mag_v
        return mag_v - X @ coefs

    return jax.vmap(fit_one)(magnitude, phase)


def phase_as_design_column(phase: jnp.ndarray) -> jnp.ndarray:
    """Return phase as a (T,) regressor column, suitable for stacking into
    the GLM design matrix.

    Use when you want the GLM to jointly estimate task β AND the phase
    nuisance coefficient — preferred over pre-residualizing because the
    statistics are correct (DOF accounting) and it composes with Variant G's
    posterior covariance.

    Args:
        phase: (T,) global phase regressor (e.g., mean phase across CSF
            voxels) OR (V, T) per-voxel phase. The (V, T) form must be
            handled by the caller as a per-voxel design column.

    Returns:
        Same shape as input — pass-through reshape for clarity.
    """
    return phase


__all__ = ["phase_regress_residuals", "phase_as_design_column"]
