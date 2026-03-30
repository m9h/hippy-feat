"""
JAX-accelerated 3D bilateral filter for neuroimaging volumes.

The bilateral filter smooths based on both spatial proximity and intensity
similarity, preserving edges while reducing noise:

    w(i,j) = exp(-||pos_i - pos_j||^2 / (2*sigma_s^2))
           * exp(-||I_i - I_j||^2 / (2*sigma_r^2))

Uses jax.lax.conv_general_dilated_patches for efficient 3D neighborhood
extraction (same approach as patch2self).
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def _estimate_sigma_range(volume, mask=None):
    """
    Estimate a reasonable sigma_range from the data.

    Uses the median absolute deviation (MAD) of voxel intensities within
    the mask as a robust scale estimate.

    Args:
        volume: 3D array (X, Y, Z)
        mask: optional boolean mask (X, Y, Z)

    Returns:
        float: estimated sigma_range
    """
    if mask is not None:
        vals = volume[mask]
    else:
        vals = volume.ravel()
    # MAD-based robust scale estimate
    mad = jnp.median(jnp.abs(vals - jnp.median(vals)))
    # Scale MAD to approximate std (factor 1.4826 for Gaussian)
    sigma_r = 1.4826 * mad
    # Clamp to avoid degenerate zero
    return jnp.maximum(sigma_r, 1e-6)


def _build_spatial_weights(kernel_radius, sigma_spatial):
    """
    Precompute the spatial Gaussian kernel weights for a cubic neighborhood.

    Args:
        kernel_radius: int, half-width of kernel
        sigma_spatial: float, spatial std in voxels

    Returns:
        spatial_weights: 1D array of length kernel_size^3, flattened spatial
                         Gaussian weights
    """
    r = kernel_radius
    ax = jnp.arange(-r, r + 1, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(ax, ax, ax, indexing='ij')
    sq_dist = gx ** 2 + gy ** 2 + gz ** 2
    spatial_w = jnp.exp(-sq_dist / (2.0 * sigma_spatial ** 2))
    return spatial_w.ravel()


@partial(jax.jit, static_argnames=['kernel_radius'])
def bilateral_filter_3d(volume, sigma_spatial=1.5, sigma_range=None,
                        kernel_radius=2, mask=None):
    """
    JIT-compiled 3D bilateral filter.

    Args:
        volume: jnp.array of shape (X, Y, Z), float32.
        sigma_spatial: float, spatial kernel width in voxels (default 1.5).
        sigma_range: float, intensity range parameter. If None, estimated
                     from data via MAD.
        kernel_radius: int, half-width of the cubic spatial kernel (default 2).
                       Total kernel side = 2*kernel_radius + 1.
        mask: optional jnp.array of shape (X, Y, Z), bool. If provided,
              only voxels inside the mask are filtered; outside voxels are
              returned unchanged.

    Returns:
        filtered: jnp.array of shape (X, Y, Z), float32.
    """
    X, Y, Z = volume.shape

    # Estimate sigma_range if not provided
    if sigma_range is None:
        sigma_range = _estimate_sigma_range(volume, mask)

    # --- 1. Extract 3D neighborhoods using conv_general_dilated_patches ---
    k = 2 * kernel_radius + 1
    k_shape = (k, k, k)

    # Reshape for conv: (N=1, C=1, D, H, W)
    vol_5d = volume[None, None, :, :, :]

    # Extract patches with SAME padding
    patches = jax.lax.conv_general_dilated_patches(
        lhs=vol_5d,
        filter_shape=k_shape,
        window_strides=(1, 1, 1),
        padding='SAME',
        dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'),
    )
    # patches shape: (1, k^3, X, Y, Z)
    patches = patches[0]  # (k^3, X, Y, Z)

    # --- 2. Compute bilateral weights ---
    # Spatial weights (precomputed, same for every voxel)
    spatial_w = _build_spatial_weights(kernel_radius, sigma_spatial)
    # Broadcast to (k^3, 1, 1, 1) for element-wise ops
    spatial_w = spatial_w[:, None, None, None]

    # Range (intensity) weights
    center_intensity = volume[None, :, :, :]  # (1, X, Y, Z)
    intensity_diff = patches - center_intensity  # (k^3, X, Y, Z)
    range_w = jnp.exp(-(intensity_diff ** 2) / (2.0 * sigma_range ** 2))

    # Combined bilateral weight
    bilateral_w = spatial_w * range_w  # (k^3, X, Y, Z)

    # --- 3. Weighted average ---
    numerator = jnp.sum(bilateral_w * patches, axis=0)   # (X, Y, Z)
    denominator = jnp.sum(bilateral_w, axis=0)            # (X, Y, Z)
    filtered = numerator / jnp.maximum(denominator, 1e-10)

    # --- 4. Apply mask: keep original values outside mask ---
    if mask is not None:
        filtered = jnp.where(mask, filtered, volume)

    return filtered


# ---------------------------------------------------------------------------
# Spherical geometry operations (from hypercoil functional/sphere.py)
# ---------------------------------------------------------------------------

def spherical_geodesic(X, Y=None, r=1.0):
    """Great-circle distance between points on a sphere.

    Args:
        X: (..., 3) Cartesian coordinates on sphere of radius r.
        Y: (..., 3) Cartesian coordinates.  If None, compute pairwise
           distances within X.
        r: Sphere radius.

    Returns:
        (..., ...) or (..., N, N) geodesic distances.
    """
    X_norm = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + 1e-12) * r
    if Y is None:
        # Pairwise: X is (N, 3)
        dot = X_norm @ X_norm.T / (r * r)
    else:
        Y_norm = Y / (jnp.linalg.norm(Y, axis=-1, keepdims=True) + 1e-12) * r
        dot = jnp.sum(X_norm * Y_norm, axis=-1) / (r * r)

    dot = jnp.clip(dot, -1.0, 1.0)
    return r * jnp.arccos(dot)


def kernel_gaussian(x, scale=1.0):
    """Isotropic Gaussian kernel.

    Args:
        x: Distance values.
        scale: Kernel bandwidth.

    Returns:
        Kernel weights.
    """
    return jnp.exp(-x ** 2 / (2.0 * scale ** 2))


def spatial_conv(data, coor, kernel_fn=None, metric_fn=None, truncate=None):
    """Manifold convolution with isotropic kernel.

    Args:
        data: (N, ...) data at N vertices.
        coor: (N, D) coordinates of vertices.
        kernel_fn: Distance -> weight function.  Default: Gaussian(scale=1).
        metric_fn: (coor, coor) -> distances.  Default: Euclidean.
        truncate: If not None, zero out kernel weights for distances > truncate.

    Returns:
        (N, ...) convolved data.
    """
    if kernel_fn is None:
        kernel_fn = kernel_gaussian
    if metric_fn is None:
        metric_fn = lambda X, Y=None: (
            jnp.sqrt(jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
            if Y is None else
            jnp.sqrt(jnp.sum((X - Y) ** 2, axis=-1))
        )

    dist = metric_fn(coor)  # (N, N)
    W = kernel_fn(dist)  # (N, N)

    if truncate is not None:
        W = jnp.where(dist <= truncate, W, 0.0)

    # Normalise rows
    W = W / (jnp.sum(W, axis=-1, keepdims=True) + 1e-12)

    return W @ data


def spherical_conv(data, coor, scale=1.0, r=1.0, truncate=None):
    """Gaussian convolution on the 2-sphere.

    Convenience wrapper combining spherical_geodesic + kernel_gaussian.

    Args:
        data: (N, ...) data at N vertices.
        coor: (N, 3) Cartesian coordinates on sphere.
        scale: Gaussian kernel bandwidth.
        r: Sphere radius.
        truncate: Maximum geodesic distance for kernel support.

    Returns:
        (N, ...) convolved data.
    """
    def metric(X, Y=None):
        return spherical_geodesic(X, Y, r=r)

    def kernel(x):
        return kernel_gaussian(x, scale=scale)

    return spatial_conv(data, coor, kernel_fn=kernel, metric_fn=metric,
                        truncate=truncate)
