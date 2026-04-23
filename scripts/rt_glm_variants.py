"""
MindEye Real-Time Preprocessing Comparison Framework

Each variant implements a different GLM/preprocessing strategy for single-trial
beta estimation from fMRI data. All variants produce the same output: a z-scored
(8627,) beta vector per stimulus trial.

The MindEye model checkpoint is frozen — only the preprocessing stage varies.
"""

import abc
import time
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jaxoccoli.glm import GeneralLinearModel
from jaxoccoli.stats import compute_f_stat
from jaxoccoli.bayesian_beta import make_ar1_conjugate_glm


# ---------------------------------------------------------------------------
# AR(1) conjugate GLM on zero-padded inputs (Variant G hot path)
# ---------------------------------------------------------------------------
# Wraps the math from jaxoccoli.bayesian_beta.make_ar1_conjugate_glm into a
# jax.jit function that takes X, Y, and an effective-length scalar as runtime
# inputs. This lets process_tr pad design matrix and volume to max_trs so the
# shape stays static across TRs (single JIT compile), while still using the
# *effective* number of TRs for the InverseGamma shape parameter so posterior
# variance shrinks correctly as data accumulates.

@partial(jax.jit, static_argnames=("pp_scalar", "rho_prior_mean",
                                    "rho_prior_var", "a0", "b0"))
def _variant_g_forward(X_pad, Y_pad, n_eff,
                        pp_scalar: float = 0.01,
                        rho_prior_mean: float = 0.5,
                        rho_prior_var: float = 0.09,
                        a0: float = 0.01, b0: float = 0.01):
    """AR(1)-prewhitened conjugate GLM, padded inputs, vmapped over voxels.

    Args:
        X_pad:  (T_max, P) design matrix, zero rows after index n_eff.
        Y_pad:  (V, T_max) voxel data, zero columns after index n_eff.
        n_eff:  scalar int32 — effective TR count (traced, not static).
    Returns:
        beta_mean: (V, P), beta_var: (V, P).
    """
    T, P = X_pad.shape
    pp = pp_scalar * jnp.eye(P)

    XtX = X_pad.T @ X_pad
    XtX_ols = XtX + 1e-6 * jnp.eye(P)
    Xty = X_pad.T @ Y_pad.T                              # (P, V)
    beta_ols = jnp.linalg.solve(XtX_ols, Xty).T          # (V, P)

    resid = Y_pad - beta_ols @ X_pad.T                   # (V, T_max)
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
# Functional JIT-compiled OLS (avoids per-instance recompilation)
# ---------------------------------------------------------------------------

@jax.jit
def _ols_fit(design_matrix: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """
    Functional OLS: betas = (X'X)^-1 X' Y.
    No class instantiation → single JIT compilation regardless of design matrix values.

    Args:
        design_matrix: (max_trs, n_regressors)
        data: (n_voxels, max_trs)

    Returns:
        betas: (n_voxels, n_regressors)
    """
    XtX = design_matrix.T @ design_matrix
    XtX_inv = jnp.linalg.inv(XtX)
    pinv = XtX_inv @ design_matrix.T
    betas = jnp.tensordot(data, pinv, axes=(-1, 1))
    return betas


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    """Configuration shared across all variants."""
    tr: float = 1.5
    n_voxels: int = 8627
    vol_shape: Tuple[int, int, int] = (76, 90, 74)
    max_trs: int = 192  # pad to this for static JIT shapes

    # Data paths
    union_mask_path: str = "/data/3t/data/union_mask_from_ses-01-02.npy"
    brain_mask_path: str = "/data/3t/data/sub-005_final_mask.nii.gz"
    hrf_indices_path: str = "/data/3t/data/avg_hrfs_s1_s2_full.npy"
    hrf_library_path: str = "/data/3t/data/getcanonicalhrflibrary.tsv"
    flobs_path: str = "/home/mhough/fsl/src/fsl-feat5/data/default_flobs.flobs/hrfbasisfns.txt"

    # Output
    output_base: str = "/data/derivatives/mindeye_variants"

    # Training data for priors (Variant D)
    training_betas_path: str = "/data/3t/data/real_time_betas/all_betas_ses-01_all_runs_delay0.npy"

    # Events
    events_dir: str = "/data/3t/data/events"

    # Volumes
    mc_volumes_dir: str = "/data/3t/derivatives/motion_corrected_resampled"


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def load_brain_mask(path: str) -> np.ndarray:
    """Load 3D brain mask NIfTI, return flat boolean."""
    import nibabel as nib
    img = nib.load(path)
    return (img.get_fdata().flatten() > 0)


def load_union_mask(path: str) -> np.ndarray:
    """Load 1D union mask boolean array."""
    return np.load(path)


def apply_masks(vol_3d: np.ndarray, brain_mask_flat: np.ndarray,
                union_mask: np.ndarray) -> np.ndarray:
    """
    Apply two-stage masking: 3D volume → brain voxels → union mask voxels.

    Args:
        vol_3d: (76, 90, 74) volume
        brain_mask_flat: (506160,) bool — 19174 True
        union_mask: (19174,) bool — 8627 True

    Returns:
        (8627,) float32 array
    """
    brain_voxels = vol_3d.flatten()[brain_mask_flat]
    return brain_voxels[union_mask].astype(np.float32)


# ---------------------------------------------------------------------------
# HRF utilities
# ---------------------------------------------------------------------------

def load_glmsingle_hrf_library(tsv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GLMsingle 20-HRF library.

    Returns:
        base_time: (n_timepoints,) time axis 0..32s
        hrfs: (n_timepoints, 20) peak-normalized HRFs
    """
    hrfs = np.loadtxt(tsv_path, delimiter="\t")  # (501, 20)
    n_tp = hrfs.shape[0]
    base_time = np.linspace(0.0, 32.0, n_tp, endpoint=True)
    # Peak-normalize each column
    peak = np.maximum(np.abs(hrfs).max(axis=0), 1e-12)
    hrfs = hrfs / peak
    return base_time, hrfs


def load_hrf_indices(hrf_path: str, brain_mask_flat: np.ndarray,
                     union_mask: np.ndarray) -> np.ndarray:
    """
    Load per-voxel HRF index array, apply masks.

    Returns:
        (8627,) int array with values 0..19
    """
    hrf_vol = np.load(hrf_path)[:, :, :, 0]  # (76, 90, 74)
    hrf_brain = hrf_vol.flatten()[brain_mask_flat]
    hrf_union = hrf_brain[union_mask].astype(int)
    return hrf_union


def load_flobs_basis(path: str) -> np.ndarray:
    """Load FLOBS 3-basis-function HRFs. Returns (559, 3)."""
    return np.loadtxt(path)


def resample_hrf(hrf_values: np.ndarray, base_time: np.ndarray,
                 tr: float, n_trs: int) -> np.ndarray:
    """
    Resample an HRF from its native time axis to TR-spaced samples.

    Args:
        hrf_values: (n_native,) HRF values
        base_time: (n_native,) time axis
        tr: repetition time
        n_trs: number of TR samples to generate

    Returns:
        (n_trs,) resampled HRF
    """
    t_out = np.arange(n_trs) * tr
    return np.interp(t_out, base_time, hrf_values, left=0.0, right=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def make_glover_hrf(tr: float, n_trs: int) -> np.ndarray:
    """Compute canonical Glover HRF sampled at TR resolution."""
    t = np.arange(n_trs) * tr
    # Glover (1999) double-gamma
    a1, a2, b1, b2, c = 6.0, 16.0, 1.0, 1.0, 1.0 / 6.0
    from scipy.stats import gamma as gamma_dist
    h = gamma_dist.pdf(t, a1, scale=b1) - c * gamma_dist.pdf(t, a2, scale=b2)
    # Normalize to peak = 1
    peak = np.abs(h).max()
    if peak > 0:
        h = h / peak
    return h.astype(np.float32)


def build_design_matrix(events_onsets: np.ndarray, tr: float, n_trs: int,
                        hrf: np.ndarray, probe_trial: int,
                        include_drift: bool = True) -> np.ndarray:
    """
    Build a design matrix for LSS (Least Squares Separate).

    Args:
        events_onsets: (n_events,) onset times in seconds
        tr: repetition time
        n_trs: number of TRs in the run so far
        hrf: (n_hrf_samples,) HRF kernel
        probe_trial: index of the probe trial (the one we want beta for)
        include_drift: whether to include cosine drift regressors

    Returns:
        (n_trs, n_regressors) design matrix
        probe_regressor_index: int, column index of the probe regressor
    """
    n_hrf = len(hrf)

    # Create stimulus boxcar for each trial
    # LSS: probe gets its own regressor, all others lumped together
    probe_onset = events_onsets[probe_trial]
    probe_boxcar = np.zeros(n_trs, dtype=np.float32)
    probe_tr = int(round(probe_onset / tr))
    if 0 <= probe_tr < n_trs:
        probe_boxcar[probe_tr] = 1.0

    ref_boxcar = np.zeros(n_trs, dtype=np.float32)
    for i, onset in enumerate(events_onsets):
        if i == probe_trial:
            continue
        ref_tr = int(round(onset / tr))
        if 0 <= ref_tr < n_trs:
            ref_boxcar[ref_tr] = 1.0

    # Convolve with HRF
    probe_reg = np.convolve(probe_boxcar, hrf)[:n_trs]
    ref_reg = np.convolve(ref_boxcar, hrf)[:n_trs]

    regressors = [probe_reg, ref_reg]
    probe_idx = 0

    # Cosine drift regressors
    if include_drift:
        t = np.arange(n_trs, dtype=np.float32) / max(n_trs - 1, 1)
        regressors.append(np.ones(n_trs, dtype=np.float32))  # intercept
        regressors.append(np.cos(2 * np.pi * t))  # 1st order drift

    dm = np.column_stack(regressors).astype(np.float32)
    return dm, probe_idx


def build_nuisance_regressors(timeseries: np.ndarray, wm_mask: np.ndarray,
                              csf_mask: np.ndarray,
                              include_derivatives: bool = False) -> np.ndarray:
    """
    Build CSF/WM nuisance regressors from tissue masks.

    Analogous to fMRIPrep's CompCor / GLMdenoise noise regression.
    Extracts mean timeseries from WM and CSF voxels to use as
    confound regressors in the GLM design matrix.

    Args:
        timeseries: (n_voxels, n_trs) masked voxel timeseries
        wm_mask: (n_voxels,) bool — white matter voxels
        csf_mask: (n_voxels,) bool — CSF voxels
        include_derivatives: if True, also include temporal derivatives

    Returns:
        (n_trs, n_regressors) nuisance regressor matrix
    """
    n_voxels, n_trs = timeseries.shape

    # Extract mean timeseries from each tissue compartment
    if wm_mask.any():
        wm_mean = timeseries[wm_mask].mean(axis=0)
    else:
        wm_mean = np.zeros(n_trs, dtype=np.float32)

    if csf_mask.any():
        csf_mean = timeseries[csf_mask].mean(axis=0)
    else:
        csf_mean = np.zeros(n_trs, dtype=np.float32)

    regressors = [wm_mean, csf_mean]

    if include_derivatives:
        # Temporal derivative (backward difference, zero-padded)
        wm_deriv = np.zeros_like(wm_mean)
        wm_deriv[1:] = np.diff(wm_mean)
        csf_deriv = np.zeros_like(csf_mean)
        csf_deriv[1:] = np.diff(csf_mean)
        regressors.extend([wm_deriv, csf_deriv])

    return np.column_stack(regressors).astype(np.float32)


def build_design_matrix_flobs(events_onsets: np.ndarray, tr: float, n_trs: int,
                              flobs_basis: np.ndarray, probe_trial: int,
                              include_drift: bool = True) -> Tuple[np.ndarray, int]:
    """
    Build design matrix using FLOBS 3-basis-function HRF.
    Each event gets 3 regressors (one per basis function).

    Returns:
        (n_trs, n_regressors) design matrix
        probe_regressor_start: int, first column index of probe's 3 regressors
    """
    n_basis = flobs_basis.shape[1]  # 3
    # Resample FLOBS to TR resolution
    n_native = flobs_basis.shape[0]
    flobs_time = np.linspace(0, 32.0, n_native, endpoint=True)
    n_hrf_trs = int(np.ceil(32.0 / tr))
    basis_resampled = np.zeros((n_hrf_trs, n_basis), dtype=np.float32)
    for b in range(n_basis):
        basis_resampled[:, b] = resample_hrf(flobs_basis[:, b], flobs_time, tr, n_hrf_trs)

    probe_onset = events_onsets[probe_trial]
    probe_boxcar = np.zeros(n_trs, dtype=np.float32)
    probe_tr = int(round(probe_onset / tr))
    if 0 <= probe_tr < n_trs:
        probe_boxcar[probe_tr] = 1.0

    ref_boxcar = np.zeros(n_trs, dtype=np.float32)
    for i, onset in enumerate(events_onsets):
        if i == probe_trial:
            continue
        ref_tr = int(round(onset / tr))
        if 0 <= ref_tr < n_trs:
            ref_boxcar[ref_tr] = 1.0

    regressors = []
    # Probe: 3 basis regressors
    probe_start = 0
    for b in range(n_basis):
        reg = np.convolve(probe_boxcar, basis_resampled[:, b])[:n_trs]
        regressors.append(reg)

    # Reference: 3 basis regressors
    for b in range(n_basis):
        reg = np.convolve(ref_boxcar, basis_resampled[:, b])[:n_trs]
        regressors.append(reg)

    if include_drift:
        t = np.arange(n_trs, dtype=np.float32) / max(n_trs - 1, 1)
        regressors.append(np.ones(n_trs, dtype=np.float32))
        regressors.append(np.cos(2 * np.pi * t))

    dm = np.column_stack(regressors).astype(np.float32)
    return dm, probe_start


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class RTPreprocessingVariant(abc.ABC):
    """
    Abstract base class for real-time preprocessing variants.

    All variants must produce a z-scored (8627,) beta vector per stimulus trial.
    """

    def __init__(self, config: VariantConfig):
        self.config = config
        self.name: str = "base"
        self._precomputed = False
        self._timing: List[float] = []
        self._betas: List[np.ndarray] = []
        self._all_raw_betas: List[np.ndarray] = []  # for z-scoring

    @abc.abstractmethod
    def precompute(self) -> None:
        """Pre-compute variant-specific data (HRF weights, priors, etc.)."""
        ...

    @abc.abstractmethod
    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        """
        Process a single TR and return the beta estimate for the probe trial.

        Args:
            volume: (n_voxels_masked, n_trs_so_far) masked voxel timeseries
                    accumulated up to this TR
            tr_index: current TR index (0-based)
            events_onsets: (n_events,) onset times in seconds
            probe_trial: index of the current probe trial

        Returns:
            (8627,) float32 raw beta vector (z-scoring applied separately)
        """
        ...

    def warmup(self) -> None:
        """JIT warmup with dummy data."""
        dummy = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        dummy_onsets = np.array([0.0, 3.0, 6.0])
        try:
            self.process_tr(dummy, 10, dummy_onsets, 0)
        except Exception:
            pass  # warmup may fail for some variants with degenerate data

    def z_score_beta(self, raw_beta: np.ndarray) -> np.ndarray:
        """Z-score a beta vector using running statistics from all collected betas."""
        self._all_raw_betas.append(raw_beta)
        all_betas = np.array(self._all_raw_betas)
        z_mean = all_betas.mean(axis=0)
        z_std = all_betas.std(axis=0) + 1e-6
        return ((raw_beta - z_mean) / z_std).astype(np.float32)

    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save betas and timing to disk."""
        if output_dir is None:
            output_dir = str(Path(self.config.output_base) / f"variant_{self.name}")
        out = Path(output_dir)
        (out / "betas").mkdir(parents=True, exist_ok=True)

        for i, beta in enumerate(self._betas):
            np.save(out / "betas" / f"run-01_tr-{i:03d}.npy", beta)

        # Save timing
        with open(out / "timing.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tr_index", "wall_time_s"])
            for i, t in enumerate(self._timing):
                writer.writerow([i, f"{t:.6f}"])


# ---------------------------------------------------------------------------
# Variant A: Baseline nilearn-equivalent GLM (JAX OLS with Glover HRF)
# ---------------------------------------------------------------------------

class VariantA_Baseline(RTPreprocessingVariant):
    """
    Baseline: Glover canonical HRF, OLS via jaxoccoli, growing window.
    Equivalent to the nilearn FirstLevelModel with hrf_model='glover'.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "a_baseline"
        self.hrf: Optional[np.ndarray] = None

    def precompute(self) -> None:
        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.hrf = make_glover_hrf(self.config.tr, n_hrf_trs)
        self._precomputed = True

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        """
        Args:
            volume: (n_voxels, n_trs_so_far) masked timeseries
            tr_index: current TR
            events_onsets: trial onsets in seconds
            probe_trial: which trial to extract beta for

        Returns:
            (8627,) float32 raw beta
        """
        n_trs = volume.shape[1]
        dm, probe_idx = build_design_matrix(
            events_onsets, self.config.tr, n_trs, self.hrf, probe_trial
        )

        # Pad to max_trs for static JIT shape
        dm_padded = np.zeros((self.config.max_trs, dm.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm

        vol_padded = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        dm_jax = jnp.array(dm_padded)
        vol_jax = jnp.array(vol_padded)

        glm = GeneralLinearModel(dm_jax)
        betas, _ = glm.fit(vol_jax)

        # Extract probe beta
        probe_beta = np.asarray(betas[:, probe_idx], dtype=np.float32)
        return probe_beta


# ---------------------------------------------------------------------------
# Variant A+N: Baseline + CSF/WM Nuisance Regression
# ---------------------------------------------------------------------------

class VariantA_NuisanceRegression(RTPreprocessingVariant):
    """
    Baseline Glover HRF GLM augmented with CSF/WM nuisance regressors.

    Bridges the gap between the current RT pipeline (no confound regression)
    and GLMsingle/fMRIPrep (CompCor-style noise removal). Extracts mean
    WM and CSF timeseries as confound regressors in the design matrix.

    Requires tissue masks from FastSurfer segmentation or T1prep.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "a_nuisance"
        self.hrf: Optional[np.ndarray] = None
        self.wm_mask: Optional[np.ndarray] = None   # (n_voxels,) bool
        self.csf_mask: Optional[np.ndarray] = None   # (n_voxels,) bool

    def precompute(self) -> None:
        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.hrf = make_glover_hrf(self.config.tr, n_hrf_trs)

        # Tissue masks can be set externally after precompute()
        # or loaded from FastSurfer segmentation files
        if self.wm_mask is None:
            self.wm_mask = np.zeros(self.config.n_voxels, dtype=bool)
        if self.csf_mask is None:
            self.csf_mask = np.zeros(self.config.n_voxels, dtype=bool)

        self._precomputed = True

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        n_trs = volume.shape[1]

        # Build stimulus design matrix
        dm, probe_idx = build_design_matrix(
            events_onsets, self.config.tr, n_trs, self.hrf, probe_trial
        )

        # Build nuisance regressors from tissue masks
        nuisance = build_nuisance_regressors(
            volume, self.wm_mask, self.csf_mask, include_derivatives=True
        )

        # Augment design matrix with nuisance columns
        dm_full = np.column_stack([dm, nuisance])

        # Pad to max_trs for static JIT shape
        dm_padded = np.zeros((self.config.max_trs, dm_full.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm_full

        vol_padded = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        dm_jax = jnp.array(dm_padded)
        vol_jax = jnp.array(vol_padded)

        betas = _ols_fit(dm_jax, vol_jax)

        # Probe beta is still column 0 (nuisance appended after stimulus regressors)
        probe_beta = np.asarray(betas[:, probe_idx], dtype=np.float32)
        return probe_beta


# ---------------------------------------------------------------------------
# Variant B: FLOBS 3-Basis-Function HRF
# ---------------------------------------------------------------------------

class VariantB_FLOBS(RTPreprocessingVariant):
    """
    FLOBS 3-basis-function HRF. Pre-computes per-voxel optimal FLOBS weights
    from training data, then combines the 3 basis betas via learned weights.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "b_flobs"
        self.flobs_basis: Optional[np.ndarray] = None
        self.voxel_weights: Optional[np.ndarray] = None  # (n_voxels, 3)

    def precompute(self, training_betas: Optional[np.ndarray] = None) -> None:
        self.flobs_basis = load_flobs_basis(self.config.flobs_path)

        # Pre-compute per-voxel FLOBS weights from training data
        # For now, use equal weights (1/3 each) as default
        # In production, these would be fit from ses-01-03 training runs
        if training_betas is not None and training_betas.shape[1] == self.config.n_voxels:
            # Fit weights from training data (simplified: use canonical weights)
            self.voxel_weights = np.ones((self.config.n_voxels, 3), dtype=np.float32) / 3.0
        else:
            # Default equal weights
            self.voxel_weights = np.ones((self.config.n_voxels, 3), dtype=np.float32) / 3.0

        self._precomputed = True

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        n_trs = volume.shape[1]
        dm, probe_start = build_design_matrix_flobs(
            events_onsets, self.config.tr, n_trs, self.flobs_basis, probe_trial
        )

        dm_padded = np.zeros((self.config.max_trs, dm.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm

        vol_padded = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        dm_jax = jnp.array(dm_padded)
        vol_jax = jnp.array(vol_padded)

        glm = GeneralLinearModel(dm_jax)
        betas, residuals = glm.fit(vol_jax)

        # Extract 3 probe betas and combine via per-voxel weights
        probe_betas = np.asarray(betas[:, probe_start:probe_start + 3], dtype=np.float32)
        # probe_betas: (n_voxels, 3), voxel_weights: (n_voxels, 3)
        combined = np.sum(probe_betas * self.voxel_weights, axis=1)

        return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Variant C: GLMsingle Per-Voxel HRF
# ---------------------------------------------------------------------------

class VariantC_PerVoxelHRF(RTPreprocessingVariant):
    """
    Per-voxel HRF selection from GLMsingle 20-HRF library.
    Groups voxels by HRF index and runs parallel JIT-compiled GLMs.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "c_pervoxel_hrf"
        self.hrf_indices: Optional[np.ndarray] = None  # (8627,) int 0..19
        self.hrf_library: Optional[np.ndarray] = None  # (n_tp, 20)
        self.base_time: Optional[np.ndarray] = None
        self.voxel_groups: Optional[Dict[int, np.ndarray]] = None  # hrf_idx → voxel indices
        self.resampled_hrfs: Optional[Dict[int, np.ndarray]] = None  # hrf_idx → (n_hrf_trs,)

    def precompute(self, brain_mask_flat: Optional[np.ndarray] = None,
                   union_mask: Optional[np.ndarray] = None) -> None:
        if brain_mask_flat is None:
            brain_mask_flat = load_brain_mask(self.config.brain_mask_path)
        if union_mask is None:
            union_mask = load_union_mask(self.config.union_mask_path)

        self.hrf_indices = load_hrf_indices(
            self.config.hrf_indices_path, brain_mask_flat, union_mask
        )
        self.base_time, self.hrf_library = load_glmsingle_hrf_library(
            self.config.hrf_library_path
        )

        # Group voxels by HRF index
        unique_hrfs = np.unique(self.hrf_indices)
        self.voxel_groups = {}
        for h in unique_hrfs:
            self.voxel_groups[int(h)] = np.where(self.hrf_indices == h)[0]

        # Pre-resample each unique HRF to TR resolution
        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.resampled_hrfs = {}
        for h in unique_hrfs:
            self.resampled_hrfs[int(h)] = resample_hrf(
                self.hrf_library[:, int(h)], self.base_time, self.config.tr, n_hrf_trs
            )

        self._precomputed = True

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        n_trs = volume.shape[1]
        result = np.zeros(self.config.n_voxels, dtype=np.float32)

        # Pre-pad volume once to max_trs
        vol_padded_full = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded_full[:, :n_trs] = volume

        for hrf_idx, voxel_ids in self.voxel_groups.items():
            hrf = self.resampled_hrfs[hrf_idx]
            dm, probe_col = build_design_matrix(
                events_onsets, self.config.tr, n_trs, hrf, probe_trial
            )

            dm_padded = np.zeros((self.config.max_trs, dm.shape[1]), dtype=np.float32)
            dm_padded[:n_trs] = dm

            # Use functional OLS — single JIT, no per-group recompilation
            dm_jax = jnp.array(dm_padded)
            vol_jax = jnp.array(vol_padded_full[voxel_ids, :])
            betas = _ols_fit(dm_jax, vol_jax)

            result[voxel_ids] = np.asarray(betas[:, probe_col], dtype=np.float32)

        return result


# ---------------------------------------------------------------------------
# Variant D: Bayesian Shrinkage (conjugate Gaussian)
# ---------------------------------------------------------------------------

class VariantD_Bayesian(RTPreprocessingVariant):
    """
    Bayesian shrinkage using closed-form conjugate Gaussian update.
    Pre-computes prior mean/variance per voxel from training session betas.
    Biggest impact on early TRs where OLS has high variance.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "d_bayesian"
        self.hrf: Optional[np.ndarray] = None
        self.prior_mean: Optional[np.ndarray] = None  # (8627,)
        self.prior_var: Optional[np.ndarray] = None    # (8627,)

    def precompute(self, training_betas: Optional[np.ndarray] = None) -> None:
        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.hrf = make_glover_hrf(self.config.tr, n_hrf_trs)

        if training_betas is not None and training_betas.shape[1] == self.config.n_voxels:
            self.prior_mean = training_betas.mean(axis=0).astype(np.float32)
            self.prior_var = np.maximum(
                training_betas.var(axis=0).astype(np.float32), 1e-6
            )
        else:
            # Default uninformative prior
            self.prior_mean = np.zeros(self.config.n_voxels, dtype=np.float32)
            self.prior_var = np.ones(self.config.n_voxels, dtype=np.float32) * 1e6

        self._precomputed = True

    @staticmethod
    @jax.jit
    def _bayesian_update(ols_beta: jnp.ndarray, ols_var: jnp.ndarray,
                         prior_mean: jnp.ndarray, prior_var: jnp.ndarray) -> jnp.ndarray:
        """Conjugate Gaussian posterior update."""
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / ols_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + ols_beta / ols_var)
        return posterior_mean

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        n_trs = volume.shape[1]
        dm, probe_idx = build_design_matrix(
            events_onsets, self.config.tr, n_trs, self.hrf, probe_trial
        )

        dm_padded = np.zeros((self.config.max_trs, dm.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm

        vol_padded = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        dm_jax = jnp.array(dm_padded)
        vol_jax = jnp.array(vol_padded)

        glm = GeneralLinearModel(dm_jax)
        betas, residuals = glm.fit(vol_jax)

        ols_beta = betas[:, probe_idx]

        # Estimate OLS variance per voxel
        rss = jnp.sum(residuals ** 2, axis=-1)
        df = max(n_trs - dm.shape[1], 1)
        sigma2 = rss / df
        # Variance of beta estimate: sigma^2 * (X'X)^-1[probe_idx, probe_idx]
        c_var = glm.XtX_inv[probe_idx, probe_idx]
        ols_var = jnp.maximum(sigma2 * c_var, 1e-10)

        posterior = self._bayesian_update(
            ols_beta, ols_var,
            jnp.array(self.prior_mean), jnp.array(self.prior_var)
        )

        return np.asarray(posterior, dtype=np.float32)


# ---------------------------------------------------------------------------
# Variant E: Spatial Regularization
# ---------------------------------------------------------------------------

class VariantE_Spatial(RTPreprocessingVariant):
    """
    Spatial regularization via Laplacian smoothness penalty.
    min ||y - Xb||^2 + lambda * ||Lb||^2
    Adds lambda * L'L to X'X — negligible extra cost.
    """

    def __init__(self, config: VariantConfig, lam: float = 0.1):
        super().__init__(config)
        self.name = "e_spatial"
        self.lam = lam
        self.hrf: Optional[np.ndarray] = None
        self.LtL: Optional[np.ndarray] = None  # (n_voxels, n_voxels) scipy sparse CSC
        self._solve = None  # Pre-factored sparse solver: callable(rhs) -> solution

    def precompute(self, brain_mask_flat: Optional[np.ndarray] = None,
                   union_mask: Optional[np.ndarray] = None) -> None:
        from scipy import sparse
        from scipy.sparse.linalg import factorized

        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.hrf = make_glover_hrf(self.config.tr, n_hrf_trs)

        if brain_mask_flat is None:
            brain_mask_flat = load_brain_mask(self.config.brain_mask_path)
        if union_mask is None:
            union_mask = load_union_mask(self.config.union_mask_path)

        # Build spatial adjacency Laplacian from 3D mask (sparse)
        self.LtL = self._build_laplacian_LtL(brain_mask_flat, union_mask)

        # Pre-factorize (I + λ L'L) as sparse for O(nnz) per-TR solves
        n = self.LtL.shape[0]
        reg_matrix = sparse.eye(n, dtype=np.float64, format="csc") + self.lam * self.LtL.astype(np.float64)
        self._solve = factorized(reg_matrix.tocsc())

        self._precomputed = True

    @staticmethod
    def _build_laplacian_LtL(brain_mask_flat: np.ndarray,
                             union_mask: np.ndarray):
        """
        Build L'L where L is the graph Laplacian of the spatial adjacency
        among union mask voxels in the 3D volume.

        Vectorized: no Python loops over voxels.
        Returns scipy sparse CSC matrix (n_voxels, n_voxels).
        """
        from scipy import sparse

        vol_shape = (76, 90, 74)
        n_voxels = int(union_mask.sum())

        # Map: flat-volume-index → union-voxel-index (-1 if not in union)
        brain_indices = np.where(brain_mask_flat)[0]
        union_indices = np.where(union_mask)[0]

        vol_to_union = -np.ones(np.prod(vol_shape), dtype=np.int32)
        union_vol_indices = brain_indices[union_indices]
        vol_to_union[union_vol_indices] = np.arange(n_voxels, dtype=np.int32)

        # Vectorized 6-connectivity: for each union voxel, check all 6 neighbors
        xyz = np.array(np.unravel_index(union_vol_indices, vol_shape)).T  # (n_voxels, 3)
        offsets = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])

        rows_list, cols_list = [], []
        for off in offsets:
            neighbors = xyz + off  # (n_voxels, 3)
            valid = (
                (neighbors[:, 0] >= 0) & (neighbors[:, 0] < vol_shape[0]) &
                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < vol_shape[1]) &
                (neighbors[:, 2] >= 0) & (neighbors[:, 2] < vol_shape[2])
            )
            neighbor_flat = np.zeros(n_voxels, dtype=np.int64)
            neighbor_flat[valid] = np.ravel_multi_index(
                neighbors[valid].T, vol_shape
            )
            neighbor_union = np.full(n_voxels, -1, dtype=np.int32)
            neighbor_union[valid] = vol_to_union[neighbor_flat[valid]]

            connected = valid & (neighbor_union >= 0)
            src = np.arange(n_voxels)[connected]
            dst = neighbor_union[connected]
            rows_list.append(src)
            cols_list.append(dst)

        if rows_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            adj = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_voxels, n_voxels)
            )
            degree = np.array(adj.sum(axis=1)).flatten()
            L = sparse.diags(degree) - adj
            LtL = (L.T @ L).tocsc().astype(np.float32)
        else:
            LtL = sparse.csc_matrix((n_voxels, n_voxels), dtype=np.float32)

        return LtL

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        """
        Spatially regularized GLM: solve (X'X + λL'L) β = X'y
        But since L'L is (n_voxels x n_voxels) and X'X is (n_regressors x n_regressors),
        we actually solve per-regressor spatial regularization.

        The correct formulation for spatial regularization in the GLM is:
        For each regressor j: add λ * L'L penalty on the spatial map of β_j.
        This modifies the OLS solution per voxel.

        Simplified approach: run standard OLS, then apply spatial smoothing
        to the resulting beta map via the Laplacian.
        """
        n_trs = volume.shape[1]
        dm, probe_idx = build_design_matrix(
            events_onsets, self.config.tr, n_trs, self.hrf, probe_trial
        )

        dm_padded = np.zeros((self.config.max_trs, dm.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm

        vol_padded = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        dm_jax = jnp.array(dm_padded)
        vol_jax = jnp.array(vol_padded)

        glm = GeneralLinearModel(dm_jax)
        betas, _ = glm.fit(vol_jax)

        ols_beta = np.asarray(betas[:, probe_idx], dtype=np.float32)

        # Apply spatial regularization: solve (I + λL'L) β_smooth = β_ols
        # Uses pre-factored sparse Cholesky from precompute() — O(nnz) per solve
        smooth_beta = self._solve(ols_beta.astype(np.float64))

        return smooth_beta.astype(np.float32)


# ---------------------------------------------------------------------------
# Variant F: Log Signature Streaming Features
# ---------------------------------------------------------------------------

class VariantF_LogSignature(RTPreprocessingVariant):
    """
    Augments OLS betas with log signature features for streaming monitoring.
    Uses signax for incremental log signatures via Chen's identity.
    Operates in PCA-reduced beta space (top n_components).
    """

    def __init__(self, config: VariantConfig, n_components: int = 32,
                 sig_depth: int = 2, sig_window: int = 20):
        super().__init__(config)
        self.name = "f_logsig"
        self.n_components = n_components
        self.sig_depth = sig_depth
        self.sig_window = sig_window  # only use last N betas for signature
        self.hrf: Optional[np.ndarray] = None
        self.pca_components: Optional[np.ndarray] = None  # (n_components, n_voxels)
        self.pca_mean: Optional[np.ndarray] = None  # (n_voxels,)
        self._beta_trajectory: List[np.ndarray] = []  # PCA-projected betas over time
        self.logsig_norm: float = 0.0

    def precompute(self, training_betas: Optional[np.ndarray] = None) -> None:
        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.hrf = make_glover_hrf(self.config.tr, n_hrf_trs)

        if training_betas is not None and training_betas.shape[1] == self.config.n_voxels:
            # Fit PCA from training betas
            self.pca_mean = training_betas.mean(axis=0).astype(np.float32)
            centered = training_betas - self.pca_mean
            # SVD for PCA
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self.pca_components = Vt[:self.n_components].astype(np.float32)
        else:
            # Random projection as fallback
            rng = np.random.RandomState(42)
            self.pca_mean = np.zeros(self.config.n_voxels, dtype=np.float32)
            self.pca_components = rng.randn(
                self.n_components, self.config.n_voxels
            ).astype(np.float32)
            # Normalize rows
            norms = np.linalg.norm(self.pca_components, axis=1, keepdims=True)
            self.pca_components /= np.maximum(norms, 1e-8)

        self._precomputed = True

    def _project_to_pca(self, beta: np.ndarray) -> np.ndarray:
        """Project (n_voxels,) beta to (n_components,) PCA space."""
        return (self.pca_components @ (beta - self.pca_mean)).astype(np.float32)

    def _compute_logsig(self) -> Tuple[float, np.ndarray]:
        """Compute log signature of recent beta trajectory (windowed)."""
        import signax

        if len(self._beta_trajectory) < 2:
            return 0.0, np.zeros(self.n_components, dtype=np.float32)

        # Window: only use last sig_window points to bound compute time
        recent = self._beta_trajectory[-self.sig_window:]
        path = jnp.array(recent)[None, :, :]  # (1, W, n_components)
        logsig = signax.logsignature(path, depth=self.sig_depth)
        logsig_flat = jnp.ravel(logsig)
        norm = float(jnp.linalg.norm(logsig_flat))
        return norm, np.asarray(logsig_flat)

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        n_trs = volume.shape[1]
        dm, probe_idx = build_design_matrix(
            events_onsets, self.config.tr, n_trs, self.hrf, probe_trial
        )

        dm_padded = np.zeros((self.config.max_trs, dm.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm

        vol_padded = np.zeros((self.config.n_voxels, self.config.max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        dm_jax = jnp.array(dm_padded)
        vol_jax = jnp.array(vol_padded)

        glm = GeneralLinearModel(dm_jax)
        betas, _ = glm.fit(vol_jax)

        raw_beta = np.asarray(betas[:, probe_idx], dtype=np.float32)

        # Update trajectory and compute log signature
        projected = self._project_to_pca(raw_beta)
        self._beta_trajectory.append(projected)
        self.logsig_norm, _ = self._compute_logsig()

        return raw_beta


# ---------------------------------------------------------------------------
# Variant G: Bayesian first-level GLM (AR(1) conjugate, with posterior variance)
# ---------------------------------------------------------------------------

class VariantG_Bayesian(RTPreprocessingVariant):
    """
    Bayesian first-level GLM on the real-time path (Phase 1: conjugate).

    Wraps jaxoccoli.bayesian_beta.make_ar1_conjugate_glm so every TR yields
    (posterior_mean, posterior_var) for the probe beta. The variance is
    exposed via `_last_beta_var` for confidence-gated downstream use
    (see `confidence_mask`).

    Uninformative prior  → mirrors Variant A OLS, AR(1)-corrected.
    Training-data prior  → mirrors Variant D shrinkage, AR(1)-corrected.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "g_bayesian"
        self.hrf: Optional[np.ndarray] = None
        self.prior_mean: Optional[np.ndarray] = None   # (n_voxels,)
        self.prior_var: Optional[np.ndarray] = None    # (n_voxels,)
        self._last_beta_var: Optional[np.ndarray] = None  # (n_voxels,)

    def precompute(self, training_betas: Optional[np.ndarray] = None) -> None:
        n_hrf_trs = int(np.ceil(32.0 / self.config.tr))
        self.hrf = make_glover_hrf(self.config.tr, n_hrf_trs)

        if training_betas is not None and training_betas.shape[1] == self.config.n_voxels:
            self.prior_mean = training_betas.mean(axis=0).astype(np.float32)
            self.prior_var = np.maximum(
                training_betas.var(axis=0).astype(np.float32), 1e-6
            )
        else:
            self.prior_mean = np.zeros(self.config.n_voxels, dtype=np.float32)
            self.prior_var = np.ones(self.config.n_voxels, dtype=np.float32) * 1e6

        self._precomputed = True

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        n_trs = volume.shape[1]
        dm, probe_idx = build_design_matrix(
            events_onsets, self.config.tr, n_trs, self.hrf, probe_trial
        )

        # Pad to max_trs so the JIT forward sees a static shape and compiles
        # only once across the whole run. Effective n_trs is threaded through
        # as a traced scalar so InverseGamma shape (a_post) and noise-variance
        # scaling use the real count, not the padded length.
        max_trs = self.config.max_trs
        dm_padded = np.zeros((max_trs, dm.shape[1]), dtype=np.float32)
        dm_padded[:n_trs] = dm
        vol_padded = np.zeros((self.config.n_voxels, max_trs), dtype=np.float32)
        vol_padded[:, :n_trs] = volume

        X = jnp.asarray(dm_padded)
        Y = jnp.asarray(vol_padded)
        n_eff = jnp.asarray(n_trs, dtype=jnp.int32)

        betas_all, vars_all = _variant_g_forward(X, Y, n_eff)

        ar1_beta = np.asarray(betas_all[:, probe_idx], dtype=np.float32)
        ar1_var = np.maximum(
            np.asarray(vars_all[:, probe_idx], dtype=np.float32), 1e-10
        )

        prior_mean = self.prior_mean.astype(np.float32)
        prior_var = self.prior_var.astype(np.float32)

        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / ar1_var)
        posterior_mean = posterior_var * (
            prior_mean / prior_var + ar1_beta / ar1_var
        )

        self._last_beta_var = posterior_var.astype(np.float32)
        return posterior_mean.astype(np.float32)


# ---------------------------------------------------------------------------
# Confidence-gating helper
# ---------------------------------------------------------------------------

def confidence_mask(beta_mean: np.ndarray, beta_var: np.ndarray,
                    threshold: float) -> np.ndarray:
    """
    Build a boolean mask of voxels whose posterior beta is "confident enough".

    Semantics: mask is True where |beta_mean| / sqrt(beta_var) > threshold
    (a z-score-like SNR cutoff). threshold=0 excludes all voxels with any
    posterior uncertainty; threshold=inf includes all.
    """
    beta_mean = np.asarray(beta_mean)
    beta_var = np.asarray(beta_var)
    if np.isinf(threshold):
        return np.ones(beta_mean.shape, dtype=bool)
    if threshold == 0.0:
        return np.zeros(beta_mean.shape, dtype=bool)
    snr = np.abs(beta_mean) / np.sqrt(np.maximum(beta_var, 1e-30))
    return np.asarray(snr > threshold, dtype=bool)


# ---------------------------------------------------------------------------
# Variant CD: Combined Per-Voxel HRF + Bayesian Shrinkage
# ---------------------------------------------------------------------------

class VariantCD_Combined(RTPreprocessingVariant):
    """
    Combines Variant C (per-voxel HRF) with Variant D (Bayesian shrinkage).
    Theoretically the strongest combination.
    """

    def __init__(self, config: VariantConfig):
        super().__init__(config)
        self.name = "cd_combined"
        self.variant_c = VariantC_PerVoxelHRF(config)
        self.prior_mean: Optional[np.ndarray] = None
        self.prior_var: Optional[np.ndarray] = None

    def precompute(self, brain_mask_flat: Optional[np.ndarray] = None,
                   union_mask: Optional[np.ndarray] = None,
                   training_betas: Optional[np.ndarray] = None) -> None:
        self.variant_c.precompute(brain_mask_flat, union_mask)

        if training_betas is not None and training_betas.shape[1] == self.config.n_voxels:
            self.prior_mean = training_betas.mean(axis=0).astype(np.float32)
            self.prior_var = np.maximum(
                training_betas.var(axis=0).astype(np.float32), 1e-6
            )
        else:
            self.prior_mean = np.zeros(self.config.n_voxels, dtype=np.float32)
            self.prior_var = np.ones(self.config.n_voxels, dtype=np.float32) * 1e6

        self._precomputed = True

    def process_tr(self, volume: np.ndarray, tr_index: int,
                   events_onsets: np.ndarray, probe_trial: int) -> np.ndarray:
        # Get per-voxel HRF OLS betas from Variant C
        ols_beta = self.variant_c.process_tr(volume, tr_index, events_onsets, probe_trial)

        # Estimate OLS variance (simplified: use global residual variance)
        # For a proper implementation, each HRF group would provide its own variance
        n_trs = volume.shape[1]
        beta_var = np.var(volume, axis=1) / max(n_trs, 1)
        ols_var = np.maximum(beta_var.astype(np.float32), 1e-10)

        # Bayesian update
        posterior = VariantD_Bayesian._bayesian_update(
            jnp.array(ols_beta), jnp.array(ols_var),
            jnp.array(self.prior_mean), jnp.array(self.prior_var)
        )

        return np.asarray(posterior, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VARIANT_REGISTRY = {
    "a_baseline": VariantA_Baseline,
    "a_nuisance": VariantA_NuisanceRegression,
    "b_flobs": VariantB_FLOBS,
    "c_pervoxel_hrf": VariantC_PerVoxelHRF,
    "d_bayesian": VariantD_Bayesian,
    "e_spatial": VariantE_Spatial,
    "f_logsig": VariantF_LogSignature,
    "g_bayesian": VariantG_Bayesian,
    "cd_combined": VariantCD_Combined,
}


def create_variant(name: str, config: Optional[VariantConfig] = None,
                   **kwargs) -> RTPreprocessingVariant:
    """Factory function to create a variant by name."""
    if config is None:
        config = VariantConfig()
    cls = VARIANT_REGISTRY[name]
    return cls(config, **kwargs)
