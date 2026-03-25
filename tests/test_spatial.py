"""
Tests for jaxoccoli spatial processing: bilateral filter and Gauss-Newton
motion correction.
"""

import time
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxoccoli.spatial import bilateral_filter_3d
from jaxoccoli.motion import RigidBodyRegistration, GaussNewtonRegistration


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def rng():
    return np.random.RandomState(123)


@pytest.fixture(scope="module")
def step_edge_volume():
    """
    Synthetic volume with a sharp step edge along the X axis.
    Left half = 100, right half = 200.
    Shape: (40, 40, 40).
    """
    vol = np.zeros((40, 40, 40), dtype=np.float32)
    vol[:20, :, :] = 100.0
    vol[20:, :, :] = 200.0
    return jnp.array(vol)


@pytest.fixture(scope="module")
def noisy_uniform_volume(rng):
    """
    Uniform volume (value=500) with additive Gaussian noise (std=30).
    Shape: (40, 40, 40).
    """
    vol = 500.0 + rng.randn(40, 40, 40).astype(np.float32) * 30.0
    return jnp.array(vol)


@pytest.fixture(scope="module")
def brain_mask_small():
    """Boolean mask that covers the interior of a (40,40,40) volume."""
    mask = np.ones((40, 40, 40), dtype=bool)
    # Exclude 2-voxel border
    mask[:2, :, :] = False
    mask[-2:, :, :] = False
    mask[:, :2, :] = False
    mask[:, -2:, :] = False
    mask[:, :, :2] = False
    mask[:, :, -2:] = False
    return jnp.array(mask)


@pytest.fixture(scope="module")
def template_volume(rng):
    """
    Smooth synthetic template with spatial structure for motion correction tests.
    Uses a Gaussian blob pattern so registration has features to track.
    Shape: (76, 90, 74).
    """
    from scipy.ndimage import gaussian_filter
    # Create structured volume with blobs that registration can track
    vol = rng.randn(76, 90, 74).astype(np.float32)
    vol = gaussian_filter(vol, sigma=5.0)  # smooth to create trackable features
    vol = vol * 500 + 1000  # scale to fMRI-like range
    return jnp.array(vol)


# ===========================================================================
# Bilateral Filter Tests
# ===========================================================================

class TestBilateralFilterEdgePreservation:
    def test_edge_preserved(self, step_edge_volume):
        """
        Bilateral filter should preserve the step edge: voxels far from
        the edge should remain close to their original values.
        """
        filtered = bilateral_filter_3d(
            step_edge_volume,
            sigma_spatial=1.5,
            sigma_range=10.0,  # small range -> strong edge preservation
            kernel_radius=2,
        )
        # Check voxels well inside each region (5 voxels from edge)
        left_orig = step_edge_volume[5, 20, 20]
        left_filt = filtered[5, 20, 20]
        right_orig = step_edge_volume[35, 20, 20]
        right_filt = filtered[35, 20, 20]

        # Should be very close to original values
        assert jnp.abs(left_filt - left_orig) < 5.0, \
            f"Left region changed too much: {left_orig} -> {left_filt}"
        assert jnp.abs(right_filt - right_orig) < 5.0, \
            f"Right region changed too much: {right_orig} -> {right_filt}"

        # The edge itself: left side near edge stays closer to 100 than 200
        near_edge_left = filtered[18, 20, 20]
        assert near_edge_left < 160.0, \
            f"Near-edge left voxel too high: {near_edge_left} (edge not preserved)"

    def test_edge_contrast_maintained(self, step_edge_volume):
        """
        The contrast across the edge should remain substantial.
        """
        filtered = bilateral_filter_3d(
            step_edge_volume,
            sigma_spatial=1.5,
            sigma_range=10.0,
            kernel_radius=2,
        )
        left_mean = jnp.mean(filtered[5:15, 10:30, 10:30])
        right_mean = jnp.mean(filtered[25:35, 10:30, 10:30])
        contrast = right_mean - left_mean
        assert contrast > 80.0, \
            f"Edge contrast too low: {contrast:.1f} (expected >80)"


class TestBilateralFilterNoiseReduction:
    def test_noise_reduced(self, noisy_uniform_volume):
        """
        On a uniform region with noise, the bilateral filter should reduce
        the standard deviation.
        """
        filtered = bilateral_filter_3d(
            noisy_uniform_volume,
            sigma_spatial=1.5,
            sigma_range=50.0,  # large range -> smooth uniformly
            kernel_radius=2,
        )
        # Compare std in interior (away from boundary effects)
        interior = slice(5, 35)
        orig_std = jnp.std(noisy_uniform_volume[interior, interior, interior])
        filt_std = jnp.std(filtered[interior, interior, interior])

        assert filt_std < orig_std, \
            f"Filtered std ({filt_std:.2f}) not less than original ({orig_std:.2f})"
        # Should be substantially reduced
        assert filt_std < 0.7 * orig_std, \
            f"Noise reduction insufficient: {filt_std:.2f} vs {orig_std:.2f}"

    def test_mean_preserved(self, noisy_uniform_volume):
        """
        The mean intensity should be approximately preserved.
        """
        filtered = bilateral_filter_3d(
            noisy_uniform_volume,
            sigma_spatial=1.5,
            sigma_range=50.0,
            kernel_radius=2,
        )
        interior = slice(5, 35)
        orig_mean = jnp.mean(noisy_uniform_volume[interior, interior, interior])
        filt_mean = jnp.mean(filtered[interior, interior, interior])
        assert jnp.abs(filt_mean - orig_mean) < 2.0, \
            f"Mean shifted: {orig_mean:.2f} -> {filt_mean:.2f}"


class TestBilateralFilterShape:
    def test_output_shape_matches_input(self, noisy_uniform_volume):
        filtered = bilateral_filter_3d(
            noisy_uniform_volume,
            sigma_spatial=1.5,
            sigma_range=50.0,
            kernel_radius=2,
        )
        assert filtered.shape == noisy_uniform_volume.shape

    def test_output_dtype_float32(self, noisy_uniform_volume):
        filtered = bilateral_filter_3d(
            noisy_uniform_volume,
            sigma_spatial=1.5,
            sigma_range=50.0,
            kernel_radius=2,
        )
        assert filtered.dtype == jnp.float32

    def test_mask_preserves_outside(self, noisy_uniform_volume, brain_mask_small):
        """Voxels outside the mask should be unchanged."""
        filtered = bilateral_filter_3d(
            noisy_uniform_volume,
            sigma_spatial=1.5,
            sigma_range=50.0,
            kernel_radius=2,
            mask=brain_mask_small,
        )
        outside = ~brain_mask_small
        np.testing.assert_array_equal(
            np.asarray(filtered[outside]),
            np.asarray(noisy_uniform_volume[outside]),
        )


class TestBilateralFilterJIT:
    def test_jit_compilable(self, noisy_uniform_volume):
        """The bilateral filter should be fully JIT-compilable."""
        jitted = jax.jit(
            lambda v: bilateral_filter_3d(v, sigma_spatial=1.5,
                                           sigma_range=50.0, kernel_radius=2)
        )
        result = jitted(noisy_uniform_volume)
        assert result.shape == noisy_uniform_volume.shape
        assert not jnp.any(jnp.isnan(result))

    def test_no_nan_output(self, step_edge_volume):
        filtered = bilateral_filter_3d(
            step_edge_volume,
            sigma_spatial=1.5,
            sigma_range=10.0,
            kernel_radius=2,
        )
        assert not jnp.any(jnp.isnan(filtered))


# ===========================================================================
# Gauss-Newton Motion Correction Tests
# ===========================================================================

class TestGaussNewtonConvergence:
    def test_converges_on_known_translation(self, template_volume):
        """
        Apply a known 2mm translation, then recover it with GN.
        """
        vol_shape = template_volume.shape

        # Create a moved version: shift by [2, 0, 0] voxels
        gn_reg = GaussNewtonRegistration(
            template=template_volume,
            vol_shape=vol_shape,
            n_iter=10,
            damping=1e-3,
        )
        true_params = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        moved = gn_reg.apply_transform(template_volume, true_params)

        # Register
        recovered_params, registered = gn_reg.register_volume(moved)

        # The recovered params are the INVERSE transform (moving→template),
        # so they should be close to [-2, 0, 0] (negated translation)
        np.testing.assert_allclose(
            np.asarray(recovered_params[:3]),
            -np.asarray(true_params[:3]),
            atol=0.5,
        )

    def test_identity_stays_identity(self, template_volume):
        """
        Registering the template to itself should return near-zero params.
        """
        vol_shape = template_volume.shape
        gn_reg = GaussNewtonRegistration(
            template=template_volume,
            vol_shape=vol_shape,
            n_iter=10,
        )
        params, registered = gn_reg.register_volume(template_volume)
        np.testing.assert_allclose(
            np.asarray(params),
            np.zeros(6),
            atol=0.1,
        )


class TestGaussNewtonFewerIterations:
    """
    Verify that Gauss-Newton converges with fewer iterations than Adam
    for the same accuracy on a simple translation.
    """

    def test_fewer_iterations_than_adam(self, template_volume):
        vol_shape = template_volume.shape
        true_params = jnp.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Create moved image
        gn_reg_helper = GaussNewtonRegistration(
            template=template_volume, vol_shape=vol_shape, n_iter=1)
        moved = gn_reg_helper.apply_transform(template_volume, true_params)

        # Gauss-Newton with 10 iterations
        gn_reg = GaussNewtonRegistration(
            template=template_volume,
            vol_shape=vol_shape,
            n_iter=10,
            damping=1e-3,
        )
        gn_params, _ = gn_reg.register_volume(moved)
        # Recovered params are the inverse transform
        inv_true = -true_params
        gn_error = jnp.sum((gn_params - inv_true) ** 2)

        # Adam with 10 iterations (same budget)
        adam_reg = RigidBodyRegistration(
            template=template_volume,
            vol_shape=vol_shape,
            step_size=0.1,
            n_iter=10,
        )
        adam_params, _ = adam_reg.register_volume(moved)
        adam_error = jnp.sum((adam_params - inv_true) ** 2)

        # GN should achieve lower error with the same iteration budget
        assert float(gn_error) < float(adam_error), \
            f"GN error ({gn_error:.4f}) not less than Adam error ({adam_error:.4f})"


class TestGaussNewtonJIT:
    def test_register_volume_jit(self, template_volume):
        """GaussNewtonRegistration.register_volume should be JIT-compiled."""
        vol_shape = template_volume.shape
        gn_reg = GaussNewtonRegistration(
            template=template_volume,
            vol_shape=vol_shape,
            n_iter=5,
        )
        # First call triggers compilation
        params, registered = gn_reg.register_volume(template_volume)
        assert params.shape == (6,)
        assert registered.shape == vol_shape
        assert not jnp.any(jnp.isnan(params))
        assert not jnp.any(jnp.isnan(registered))

    def test_apply_transform_jit(self, template_volume):
        """apply_transform should be JIT-compiled and produce valid output."""
        vol_shape = template_volume.shape
        gn_reg = GaussNewtonRegistration(
            template=template_volume,
            vol_shape=vol_shape,
            n_iter=5,
        )
        p = jnp.zeros(6)
        result = gn_reg.apply_transform(template_volume, p)
        assert result.shape == vol_shape
        # Identity transform should return near-identical volume
        np.testing.assert_allclose(
            np.asarray(result),
            np.asarray(template_volume),
            atol=1e-4,
        )
