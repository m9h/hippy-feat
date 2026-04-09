"""TOF-MRA preprocessing pipeline: vessel segmentation → centerlines → VesselTree.

Implements steps 1-4 of the hippy-feat angiography pipeline:

    TOF-MRA NIfTI → Frangi enhancement → threshold → skeletonize
                   → radii (distance transform) → branch labeling
                   → VesselTree dict {points, radii, branch_ids}

The output is compatible with ``vpjax.vascular.angiography.VesselTree``.

Additionally provides ``vessel_density_map`` for use as a spatial confound
regressor in the fMRI GLM (Vigneau-Roy et al. 2014).

Dependencies: scikit-image, scipy (already in project deps).

References
----------
Frangi AF et al. (1998) MICCAI. Multiscale vessel enhancement filtering.
Vigneau-Roy N et al. (2014) NeuroImage. Regional variations in vascular
    density correlate with resting-state and task-evoked fMRI signal.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label as ndimage_label,
)


# ---------------------------------------------------------------------------
# Frangi vesselness enhancement
# ---------------------------------------------------------------------------

def _hessian_eigenvalues(volume: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sorted absolute eigenvalues of the 3D Hessian at scale *sigma*.

    Returns (|λ1|, |λ2|, |λ3|) with |λ1| <= |λ2| <= |λ3|.
    """
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)

    # Second derivatives via finite differences on the smoothed volume
    gzz = np.diff(smoothed, n=2, axis=0)
    gyy = np.diff(smoothed, n=2, axis=1)
    gxx = np.diff(smoothed, n=2, axis=2)

    # Crop to common shape
    s = tuple(min(a, b, c) for a, b, c in zip(gzz.shape, gyy.shape, gxx.shape))
    gzz = gzz[:s[0], :s[1], :s[2]]
    gyy = gyy[:s[0], :s[1], :s[2]]
    gxx = gxx[:s[0], :s[1], :s[2]]

    # Cross derivatives
    dzy = np.diff(np.diff(smoothed, axis=0), axis=1)[:s[0], :s[1], :s[2]]
    dzx = np.diff(np.diff(smoothed, axis=0), axis=2)[:s[0], :s[1], :s[2]]
    dyx = np.diff(np.diff(smoothed, axis=1), axis=2)[:s[0], :s[1], :s[2]]

    # Scale normalization (Lindeberg)
    scale = sigma ** 2
    gzz *= scale
    gyy *= scale
    gxx *= scale
    dzy *= scale
    dzx *= scale
    dyx *= scale

    # Eigenvalues via analytic cubic formula for 3×3 symmetric matrices
    # For each voxel: H = [[gzz, dzy, dzx], [dzy, gyy, dyx], [dzx, dyx, gxx]]
    # Use numpy to solve per-voxel (vectorized via reshape)
    flat_shape = (-1,)
    n = gzz.size
    H = np.zeros((n, 3, 3), dtype=np.float64)
    H[:, 0, 0] = gzz.ravel()
    H[:, 1, 1] = gyy.ravel()
    H[:, 2, 2] = gxx.ravel()
    H[:, 0, 1] = H[:, 1, 0] = dzy.ravel()
    H[:, 0, 2] = H[:, 2, 0] = dzx.ravel()
    H[:, 1, 2] = H[:, 2, 1] = dyx.ravel()

    eigvals = np.linalg.eigvalsh(H)  # (n, 3), sorted ascending
    abs_eigvals = np.abs(eigvals)
    # Sort by absolute value ascending
    idx = np.argsort(abs_eigvals, axis=1)
    abs_sorted = np.take_along_axis(abs_eigvals, idx, axis=1)
    # Also need the signed values sorted by absolute value for dark-on-bright check
    signed_sorted = np.take_along_axis(eigvals, idx, axis=1)

    l1 = abs_sorted[:, 0].reshape(s)
    l2 = abs_sorted[:, 1].reshape(s)
    l3 = abs_sorted[:, 2].reshape(s)
    s2 = signed_sorted[:, 1].reshape(s)
    s3 = signed_sorted[:, 2].reshape(s)

    return l1, l2, l3, s2, s3


def frangi_enhance(
    volume: np.ndarray,
    sigmas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float | None = None,
    bright_on_dark: bool = True,
) -> np.ndarray:
    """Frangi multiscale vesselness filter for 3D volumes.

    Parameters
    ----------
    volume : 3D array
    sigmas : scales for Gaussian smoothing
    alpha : plate-vs-line sensitivity (R_A)
    beta : blob-vs-background sensitivity (R_B)
    gamma : structureness threshold (auto if None)
    bright_on_dark : True for TOF-MRA (bright vessels on dark background)

    Returns
    -------
    Vesselness image (same shape as input), values in [0, 1].
    """
    volume = np.asarray(volume, dtype=np.float64)
    if np.all(volume == 0):
        return np.zeros_like(volume, dtype=np.float32)

    output_shape = volume.shape
    vesselness = np.zeros(output_shape, dtype=np.float64)

    for sigma in sigmas:
        l1, l2, l3, s2, s3 = _hessian_eigenvalues(volume, sigma)
        s = l1.shape  # may be smaller due to diff cropping

        # Frangi conditions: for bright vessels, λ2 < 0 and λ3 < 0
        if bright_on_dark:
            condition = (s2 < 0) & (s3 < 0)
        else:
            condition = (s2 > 0) & (s3 > 0)

        # Avoid division by zero
        l2_safe = np.maximum(l2, 1e-10)
        l3_safe = np.maximum(l3, 1e-10)

        # Frangi ratios
        R_A = l2_safe / l3_safe  # plate vs line
        R_B = l1 / np.sqrt(l2_safe * l3_safe)  # blob vs background
        S = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)  # structureness

        if gamma is None:
            gamma_val = S.max() / 2.0 if S.max() > 0 else 1.0
        else:
            gamma_val = gamma

        v = (1 - np.exp(-R_A ** 2 / (2 * alpha ** 2))) * \
            np.exp(-R_B ** 2 / (2 * beta ** 2)) * \
            (1 - np.exp(-S ** 2 / (2 * gamma_val ** 2)))

        v[~condition] = 0.0

        # Pad back to original shape and take max across scales
        padded = np.zeros(output_shape, dtype=np.float64)
        # Center the cropped result
        offsets = tuple((o - c) // 2 for o, c in zip(output_shape, s))
        padded[
            offsets[0]:offsets[0] + s[0],
            offsets[1]:offsets[1] + s[1],
            offsets[2]:offsets[2] + s[2],
        ] = v
        vesselness = np.maximum(vesselness, padded)

    # Normalize to [0, 1]
    vmax = vesselness.max()
    if vmax > 0:
        vesselness /= vmax

    return vesselness.astype(np.float32)


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def threshold_vessels(
    vesselness: np.ndarray,
    method: str = "otsu",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Binarize a vesselness image.

    Parameters
    ----------
    vesselness : output of frangi_enhance
    method : "otsu" or a float threshold in [0, 1]
    mask : optional brain mask; voxels outside are zeroed

    Returns
    -------
    Binary segmentation (uint8, 0 or 1).
    """
    from skimage.filters import threshold_otsu

    v = vesselness.copy()
    if mask is not None:
        v[~mask] = 0.0

    vals = v[v > 0]
    if len(vals) == 0:
        return np.zeros_like(v, dtype=np.uint8)

    if method == "otsu":
        thresh = threshold_otsu(vals)
    else:
        thresh = float(method)

    seg = (v >= thresh).astype(np.uint8)
    if mask is not None:
        seg[~mask] = 0
    return seg


# ---------------------------------------------------------------------------
# Skeletonization
# ---------------------------------------------------------------------------

def skeletonize_vessels(segmentation: np.ndarray) -> np.ndarray:
    """Extract skeleton coordinates from a binary vessel segmentation.

    Parameters
    ----------
    segmentation : binary 3D volume (0=background, nonzero=vessel)

    Returns
    -------
    Centerline coordinates (N, 3) as float32, or empty (0, 3) array.
    """
    from skimage.morphology import skeletonize

    binary = (segmentation > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros((0, 3), dtype=np.float32)

    skeleton = skeletonize(binary)
    coords = np.argwhere(skeleton > 0).astype(np.float32)
    return coords


# ---------------------------------------------------------------------------
# Radius estimation
# ---------------------------------------------------------------------------

def estimate_radii(
    segmentation: np.ndarray,
    centerline_points: np.ndarray,
) -> np.ndarray:
    """Estimate vessel radius at each centerline point via distance transform.

    Parameters
    ----------
    segmentation : binary 3D volume
    centerline_points : (N, 3) coordinates in voxel space

    Returns
    -------
    Radius at each point (N,) in voxel units.
    """
    dt = distance_transform_edt((segmentation > 0).astype(np.uint8))
    coords = np.clip(
        np.round(centerline_points).astype(int),
        0,
        np.array(segmentation.shape) - 1,
    )
    radii = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
    return radii.astype(np.float32)


# ---------------------------------------------------------------------------
# Branch labeling
# ---------------------------------------------------------------------------

def label_branches(centerline_points: np.ndarray, connectivity: float = 2.0) -> np.ndarray:
    """Assign branch IDs to centerline points via connected-component analysis.

    Reconstructs a sparse skeleton volume from the points, labels connected
    components, then reads back the label per point.

    Parameters
    ----------
    centerline_points : (N, 3) voxel coordinates
    connectivity : max distance between connected skeleton voxels

    Returns
    -------
    Branch IDs (N,) as int32.
    """
    if len(centerline_points) == 0:
        return np.zeros(0, dtype=np.int32)

    coords = np.round(centerline_points).astype(int)
    lo = coords.min(axis=0)
    coords_shifted = coords - lo
    shape = tuple(coords_shifted.max(axis=0) + 1)

    # Reconstruct skeleton volume
    skel_vol = np.zeros(shape, dtype=np.uint8)
    skel_vol[coords_shifted[:, 0], coords_shifted[:, 1], coords_shifted[:, 2]] = 1

    # Label connected components (26-connectivity for 3D)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n_features = ndimage_label(skel_vol, structure=structure)

    # Read back labels
    branch_ids = labeled[coords_shifted[:, 0], coords_shifted[:, 1], coords_shifted[:, 2]]
    return branch_ids.astype(np.int32)


# ---------------------------------------------------------------------------
# VesselTree builder
# ---------------------------------------------------------------------------

def build_vessel_tree(
    segmentation: np.ndarray,
    voxel_size: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build a VesselTree-compatible dict from a binary vessel segmentation.

    Parameters
    ----------
    segmentation : binary 3D volume
    voxel_size : isotropic voxel size (mm or µm) for coordinate scaling

    Returns
    -------
    Dict with keys 'points' (N,3), 'radii' (N,), 'branch_ids' (N,).
    Compatible with ``vpjax.vascular.angiography.VesselTree``.
    """
    coords = skeletonize_vessels(segmentation)
    if len(coords) == 0:
        return {
            "points": np.zeros((0, 3), dtype=np.float32),
            "radii": np.zeros(0, dtype=np.float32),
            "branch_ids": np.zeros(0, dtype=np.int32),
        }

    radii = estimate_radii(segmentation, coords)
    branch_ids = label_branches(coords)

    # Scale to world coordinates
    points = coords * voxel_size
    radii_scaled = radii * voxel_size

    return {
        "points": points.astype(np.float32),
        "radii": radii_scaled.astype(np.float32),
        "branch_ids": branch_ids,
    }


# ---------------------------------------------------------------------------
# Vessel density map (GLM confound regressor)
# ---------------------------------------------------------------------------

def vessel_density_map(
    segmentation: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Compute a smooth vessel density map from a binary segmentation.

    The density map quantifies local vascular density and is intended
    for use as a spatial confound regressor in the fMRI GLM, following
    Vigneau-Roy et al. (2014).

    Parameters
    ----------
    segmentation : binary 3D volume (0=background, nonzero=vessel)
    sigma : Gaussian smoothing kernel width (voxels)

    Returns
    -------
    Density map in [0, 1], same shape as input.
    """
    binary = (segmentation > 0).astype(np.float64)
    density = gaussian_filter(binary, sigma=sigma)
    dmax = density.max()
    if dmax > 0:
        density /= dmax
    return density.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def tof_pipeline(
    tof_volume: np.ndarray,
    mask: np.ndarray | None = None,
    sigmas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    voxel_size: float = 1.0,
) -> dict[str, np.ndarray]:
    """End-to-end TOF-MRA → VesselTree pipeline.

    Steps:
        1. Frangi vessel enhancement
        2. Otsu thresholding (with optional mask)
        3. Skeletonization → centerlines
        4. Radius estimation + branch labeling

    Parameters
    ----------
    tof_volume : 3D TOF-MRA volume
    mask : optional brain mask
    sigmas : Frangi filter scales
    voxel_size : isotropic voxel dimension for coordinate scaling

    Returns
    -------
    VesselTree-compatible dict {points, radii, branch_ids}.
    """
    enhanced = frangi_enhance(tof_volume, sigmas=sigmas)
    seg = threshold_vessels(enhanced, mask=mask)
    return build_vessel_tree(seg, voxel_size=voxel_size)
