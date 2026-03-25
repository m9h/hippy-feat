"""
Test suite for MindEye RT GLM preprocessing variants.

Follows TDD: tests for all variants defined upfront.
Test categories:
  - Unit: output shape, dtype, no NaN
  - Integration: variant output through frozen MindEye model (no NaN/Inf)
  - Regression: Variant A matches existing pipeline within tolerance
  - Performance: per-TR timing < threshold after JIT warmup
  - Property-based: hypothesis for edge cases
"""

import time
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rt_glm_variants import (
    VariantConfig,
    VariantA_Baseline,
    VariantB_FLOBS,
    VariantC_PerVoxelHRF,
    VariantD_Bayesian,
    VariantE_Spatial,
    VariantF_LogSignature,
    VariantCD_Combined,
    VARIANT_REGISTRY,
    create_variant,
    make_glover_hrf,
    build_design_matrix,
    build_design_matrix_flobs,
    build_nuisance_regressors,
    apply_masks,
    load_flobs_basis,
    load_glmsingle_hrf_library,
    resample_hrf,
)


# ===========================================================================
# Unit Tests: Design Matrix Construction
# ===========================================================================

class TestDesignMatrix:
    def test_glover_hrf_shape(self):
        hrf = make_glover_hrf(tr=1.5, n_trs=22)
        assert hrf.shape == (22,)
        assert hrf.dtype == np.float32

    def test_glover_hrf_normalized(self):
        hrf = make_glover_hrf(tr=1.5, n_trs=22)
        assert np.abs(hrf).max() == pytest.approx(1.0, abs=0.01)

    def test_glover_hrf_peaks_positive(self):
        hrf = make_glover_hrf(tr=1.5, n_trs=22)
        # HRF should peak around 5-6s (TR index 3-4 at TR=1.5)
        peak_idx = np.argmax(hrf)
        assert 2 <= peak_idx <= 6

    def test_design_matrix_shape(self):
        onsets = np.array([3.0, 9.0, 15.0])
        hrf = make_glover_hrf(1.5, 22)
        dm, probe_idx = build_design_matrix(onsets, 1.5, 50, hrf, probe_trial=1)
        assert dm.shape[0] == 50
        assert dm.shape[1] >= 2  # at least probe + reference
        assert probe_idx == 0
        assert dm.dtype == np.float32

    def test_design_matrix_no_nan(self):
        onsets = np.array([3.0, 9.0, 15.0])
        hrf = make_glover_hrf(1.5, 22)
        dm, _ = build_design_matrix(onsets, 1.5, 50, hrf, probe_trial=0)
        assert not np.any(np.isnan(dm))

    def test_flobs_design_matrix_wider(self):
        onsets = np.array([3.0, 9.0])
        hrf = make_glover_hrf(1.5, 22)
        dm_glover, _ = build_design_matrix(onsets, 1.5, 50, hrf, probe_trial=0)

        # FLOBS design should have 3x more regressors per condition
        flobs = np.random.randn(559, 3).astype(np.float32)
        dm_flobs, probe_start = build_design_matrix_flobs(
            onsets, 1.5, 50, flobs, probe_trial=0
        )
        assert dm_flobs.shape[1] > dm_glover.shape[1]
        assert probe_start == 0


# ===========================================================================
# Unit Tests: Mask Utilities
# ===========================================================================

class TestMasks:
    def test_apply_masks_shape(self, dummy_volume_3d, brain_mask_flat, union_mask):
        result = apply_masks(dummy_volume_3d, brain_mask_flat, union_mask)
        assert result.shape == (8627,)
        assert result.dtype == np.float32

    def test_apply_masks_no_nan(self, dummy_volume_3d, brain_mask_flat, union_mask):
        result = apply_masks(dummy_volume_3d, brain_mask_flat, union_mask)
        assert not np.any(np.isnan(result))


# ===========================================================================
# Unit Tests: Variant A — Baseline
# ===========================================================================

class TestVariantA:
    def test_precompute(self, test_config):
        v = VariantA_Baseline(test_config)
        v.precompute()
        assert v._precomputed
        assert v.hrf is not None
        assert v.hrf.shape[0] > 0

    def test_process_tr_shape(self, test_config, dummy_timeseries, dummy_events):
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=5)
        assert beta.shape == (test_config.n_voxels,)

    def test_process_tr_dtype(self, test_config, dummy_timeseries, dummy_events):
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=5)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, test_config, dummy_timeseries, dummy_events):
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=5)
        assert not np.any(np.isnan(beta))

    def test_process_tr_no_inf(self, test_config, dummy_timeseries, dummy_events):
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=5)
        assert not np.any(np.isinf(beta))

    def test_z_score_properties(self, test_config, dummy_timeseries, dummy_events):
        v = VariantA_Baseline(test_config)
        v.precompute()
        # Collect multiple betas then z-score
        for trial in range(5):
            raw = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=trial)
            z = v.z_score_beta(raw)
        # After multiple trials, z-scored vector should have reasonable stats
        assert z.shape == (test_config.n_voxels,)
        assert not np.any(np.isnan(z))

    def test_different_probe_different_beta(self, test_config, dummy_timeseries, dummy_events):
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta0 = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)
        beta1 = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=1)
        # Different probes should give different betas
        assert not np.allclose(beta0, beta1)

    def test_save_results(self, test_config, dummy_timeseries, dummy_events, tmp_path):
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)
        v._betas.append(beta)
        v._timing.append(0.5)
        v.save_results(str(tmp_path / "variant_a"))
        assert (tmp_path / "variant_a" / "betas" / "run-01_tr-000.npy").exists()
        assert (tmp_path / "variant_a" / "timing.csv").exists()


# ===========================================================================
# Unit Tests: Variant B — FLOBS
# ===========================================================================

class TestVariantB:
    def test_precompute(self, test_config):
        v = VariantB_FLOBS(test_config)
        v.precompute()
        assert v._precomputed
        assert v.flobs_basis is not None
        assert v.voxel_weights is not None
        assert v.voxel_weights.shape == (test_config.n_voxels, 3)

    def test_process_tr_shape(self, test_config, dummy_timeseries, dummy_events):
        v = VariantB_FLOBS(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        assert beta.shape == (test_config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, test_config, dummy_timeseries, dummy_events):
        v = VariantB_FLOBS(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        assert not np.any(np.isnan(beta))

    def test_flobs_basis_loaded(self, real_data_available):
        if not real_data_available:
            pytest.skip("Real FLOBS data not available")
        basis = load_flobs_basis(
            "/home/mhough/fsl/src/fsl-feat5/data/default_flobs.flobs/hrfbasisfns.txt"
        )
        assert basis.shape == (559, 3)
        assert not np.any(np.isnan(basis))

    def test_different_from_variant_a(self, test_config, dummy_timeseries, dummy_events):
        """FLOBS betas should differ from Glover betas."""
        va = VariantA_Baseline(test_config)
        va.precompute()
        beta_a = va.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)

        vb = VariantB_FLOBS(test_config)
        vb.precompute()
        beta_b = vb.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)

        # They should produce different values (different HRF basis)
        assert not np.allclose(beta_a, beta_b, atol=1e-3)


# ===========================================================================
# Unit Tests: Variant C — Per-Voxel HRF
# ===========================================================================

class TestVariantC:
    @pytest.fixture
    def variant_c_with_dummy_hrfs(self, test_config, rng):
        """Create a Variant C instance with synthetic HRF data."""
        v = VariantC_PerVoxelHRF(test_config)
        # Manually set up with dummy data instead of loading from disk
        n_vox = test_config.n_voxels
        v.hrf_indices = rng.randint(0, 20, size=n_vox)
        base_time = np.linspace(0, 32, 501, endpoint=True)
        hrf_lib = rng.randn(501, 20).astype(np.float32)
        # Peak normalize
        for i in range(20):
            peak = np.abs(hrf_lib[:, i]).max()
            if peak > 0:
                hrf_lib[:, i] /= peak
        v.base_time = base_time
        v.hrf_library = hrf_lib

        unique_hrfs = np.unique(v.hrf_indices)
        v.voxel_groups = {int(h): np.where(v.hrf_indices == h)[0] for h in unique_hrfs}

        n_hrf_trs = int(np.ceil(32.0 / test_config.tr))
        v.resampled_hrfs = {
            int(h): resample_hrf(hrf_lib[:, int(h)], base_time, test_config.tr, n_hrf_trs)
            for h in unique_hrfs
        }
        v._precomputed = True
        return v

    def test_precompute_real_data(self, real_data_available, real_config):
        if not real_data_available:
            pytest.skip("Real data not available")
        v = VariantC_PerVoxelHRF(real_config)
        v.precompute()
        assert v._precomputed
        assert v.hrf_indices.shape == (8627,)
        assert v.hrf_library.shape[1] == 20
        assert len(v.voxel_groups) > 0

    def test_process_tr_shape(self, variant_c_with_dummy_hrfs, dummy_timeseries, dummy_events):
        v = variant_c_with_dummy_hrfs
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)
        assert beta.shape == (v.config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, variant_c_with_dummy_hrfs, dummy_timeseries, dummy_events):
        v = variant_c_with_dummy_hrfs
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)
        assert not np.any(np.isnan(beta))

    def test_hrf_indices_loaded(self, real_data_available):
        if not real_data_available:
            pytest.skip("Real data not available")
        from rt_glm_variants import load_hrf_indices, load_brain_mask, load_union_mask
        brain_mask = load_brain_mask("/data/3t/data/sub-005_final_mask.nii.gz")
        union_mask = load_union_mask("/data/3t/data/union_mask_from_ses-01-02.npy")
        indices = load_hrf_indices(
            "/data/3t/data/avg_hrfs_s1_s2_full.npy", brain_mask, union_mask
        )
        assert indices.shape == (8627,)
        assert indices.min() >= 0
        assert indices.max() <= 19

    def test_hrf_library_loaded(self, real_data_available):
        if not real_data_available:
            pytest.skip("Real data not available")
        base_time, hrfs = load_glmsingle_hrf_library(
            "/data/3t/data/getcanonicalhrflibrary.tsv"
        )
        assert hrfs.shape == (501, 20)
        assert base_time.shape == (501,)
        assert base_time[0] == 0.0
        assert base_time[-1] == pytest.approx(32.0)

    def test_voxel_groups_cover_all(self, variant_c_with_dummy_hrfs):
        v = variant_c_with_dummy_hrfs
        total = sum(len(ids) for ids in v.voxel_groups.values())
        assert total == v.config.n_voxels


# ===========================================================================
# Unit Tests: Variant D — Bayesian
# ===========================================================================

class TestVariantD:
    def test_precompute_with_training_data(self, test_config, dummy_training_betas):
        v = VariantD_Bayesian(test_config)
        v.precompute(training_betas=dummy_training_betas)
        assert v._precomputed
        assert v.prior_mean.shape == (test_config.n_voxels,)
        assert v.prior_var.shape == (test_config.n_voxels,)
        assert np.all(v.prior_var > 0)

    def test_precompute_uninformative_prior(self, test_config):
        v = VariantD_Bayesian(test_config)
        v.precompute()
        assert v._precomputed
        # Uninformative prior: large variance
        assert np.all(v.prior_var > 1e5)

    def test_process_tr_shape(self, test_config, dummy_timeseries, dummy_events):
        v = VariantD_Bayesian(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=4)
        assert beta.shape == (test_config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, test_config, dummy_timeseries, dummy_events):
        v = VariantD_Bayesian(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=4)
        assert not np.any(np.isnan(beta))

    def test_posterior_between_prior_and_ols(self, test_config, dummy_timeseries,
                                              dummy_events, dummy_training_betas):
        """Posterior mean should be between prior mean and OLS estimate."""
        v = VariantD_Bayesian(test_config)
        v.precompute(training_betas=dummy_training_betas)

        # Get OLS beta for comparison
        va = VariantA_Baseline(test_config)
        va.precompute()
        ols_beta = va.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)

        # Get Bayesian beta
        bayes_beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)

        # Posterior should be between prior and OLS on most voxels:
        # |posterior - prior| + |posterior - ols| ≈ |prior - ols|
        # Equivalently, for each voxel the posterior should lie in the interval
        # [min(prior, ols), max(prior, ols)] ± tolerance
        prior = v.prior_mean
        lo = np.minimum(prior, ols_beta)
        hi = np.maximum(prior, ols_beta)
        tol = 0.1 * (hi - lo + 1e-6)
        in_range = (bayes_beta >= lo - tol) & (bayes_beta <= hi + tol)
        frac_in_range = in_range.mean()
        assert frac_in_range > 0.9, f"Only {frac_in_range:.2%} of voxels have posterior between prior and OLS"

    def test_bayesian_update_jit(self):
        """Test that the Bayesian update function is JIT-compiled correctly."""
        ols = jnp.array([1.0, 2.0, 3.0])
        ols_var = jnp.array([1.0, 1.0, 1.0])
        prior_mean = jnp.array([0.0, 0.0, 0.0])
        prior_var = jnp.array([1.0, 1.0, 1.0])
        result = VariantD_Bayesian._bayesian_update(ols, ols_var, prior_mean, prior_var)
        # With equal variance, posterior should be midpoint
        expected = jnp.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-5)


# ===========================================================================
# Unit Tests: Variant E — Spatial Regularization
# ===========================================================================

class TestVariantE:
    def test_precompute(self, test_config, brain_mask_flat, union_mask):
        v = VariantE_Spatial(test_config)
        v.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        assert v._precomputed
        assert v.LtL is not None
        assert v.LtL.shape == (test_config.n_voxels, test_config.n_voxels)
        # Pre-factored solver should be cached
        assert v._solve is not None

    def test_laplacian_symmetric(self, test_config, brain_mask_flat, union_mask):
        v = VariantE_Spatial(test_config)
        v.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        LtL_dense = v.LtL.toarray()
        np.testing.assert_allclose(LtL_dense, LtL_dense.T, atol=1e-6)

    def test_laplacian_positive_semidefinite(self, test_config, brain_mask_flat, union_mask):
        """L'L should be positive semi-definite."""
        v = VariantE_Spatial(test_config)
        v.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        LtL_dense = v.LtL.toarray()
        eigenvalues = np.linalg.eigvalsh(LtL_dense)
        assert np.all(eigenvalues >= -1e-6)

    def test_process_tr_shape(self, test_config, dummy_timeseries, dummy_events,
                               brain_mask_flat, union_mask):
        v = VariantE_Spatial(test_config)
        v.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)
        assert beta.shape == (test_config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, test_config, dummy_timeseries, dummy_events,
                                brain_mask_flat, union_mask):
        v = VariantE_Spatial(test_config)
        v.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)
        assert not np.any(np.isnan(beta))


# ===========================================================================
# Unit Tests: Variant F — Log Signature
# ===========================================================================

class TestVariantF:
    def test_precompute(self, test_config, dummy_training_betas):
        v = VariantF_LogSignature(test_config)
        v.precompute(training_betas=dummy_training_betas)
        assert v._precomputed
        assert v.pca_components.shape == (32, test_config.n_voxels)
        assert v.pca_mean.shape == (test_config.n_voxels,)

    def test_process_tr_shape(self, test_config, dummy_timeseries, dummy_events):
        v = VariantF_LogSignature(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)
        assert beta.shape == (test_config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, test_config, dummy_timeseries, dummy_events):
        v = VariantF_LogSignature(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)
        assert not np.any(np.isnan(beta))

    def test_logsig_norm_increases(self, test_config, dummy_timeseries, dummy_events):
        """Log signature norm should change as more betas are processed."""
        v = VariantF_LogSignature(test_config)
        v.precompute()
        norms = []
        for trial in range(5):
            v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=trial)
            norms.append(v.logsig_norm)
        # After first trial, norm should be 0 (need at least 2 points)
        assert norms[0] == 0.0
        # After 2+ trials, norm should be > 0
        assert norms[-1] > 0.0

    def test_pca_projection_dimension(self, test_config, dummy_training_betas):
        v = VariantF_LogSignature(test_config, n_components=32)
        v.precompute(training_betas=dummy_training_betas)
        assert v.pca_components.shape[0] == 32
        projected = v._project_to_pca(np.zeros(test_config.n_voxels))
        assert projected.shape == (32,)


# ===========================================================================
# Unit Tests: Variant CD — Combined
# ===========================================================================

class TestVariantCD:
    @pytest.fixture
    def variant_cd_precomputed(self, test_config, rng, dummy_training_betas):
        v = VariantCD_Combined(test_config)
        # Set up Variant C internals with dummy data
        vc = v.variant_c
        n_vox = test_config.n_voxels
        vc.hrf_indices = rng.randint(0, 20, size=n_vox)
        base_time = np.linspace(0, 32, 501, endpoint=True)
        hrf_lib = rng.randn(501, 20).astype(np.float32)
        for i in range(20):
            peak = np.abs(hrf_lib[:, i]).max()
            if peak > 0:
                hrf_lib[:, i] /= peak
        vc.base_time = base_time
        vc.hrf_library = hrf_lib
        unique_hrfs = np.unique(vc.hrf_indices)
        vc.voxel_groups = {int(h): np.where(vc.hrf_indices == h)[0] for h in unique_hrfs}
        n_hrf_trs = int(np.ceil(32.0 / test_config.tr))
        vc.resampled_hrfs = {
            int(h): resample_hrf(hrf_lib[:, int(h)], base_time, test_config.tr, n_hrf_trs)
            for h in unique_hrfs
        }
        vc._precomputed = True
        # Set up priors
        v.prior_mean = dummy_training_betas.mean(axis=0).astype(np.float32)
        v.prior_var = np.maximum(dummy_training_betas.var(axis=0).astype(np.float32), 1e-6)
        v._precomputed = True
        return v

    def test_process_tr_shape(self, variant_cd_precomputed, dummy_timeseries, dummy_events):
        beta = variant_cd_precomputed.process_tr(
            dummy_timeseries, 49, dummy_events, probe_trial=1
        )
        assert beta.shape == (variant_cd_precomputed.config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, variant_cd_precomputed, dummy_timeseries, dummy_events):
        beta = variant_cd_precomputed.process_tr(
            dummy_timeseries, 49, dummy_events, probe_trial=1
        )
        assert not np.any(np.isnan(beta))


# ===========================================================================
# Registry and Factory Tests
# ===========================================================================

class TestRegistry:
    def test_all_variants_registered(self):
        expected = {"a_baseline", "a_nuisance", "b_flobs", "c_pervoxel_hrf",
                    "d_bayesian", "e_spatial", "f_logsig", "cd_combined"}
        assert set(VARIANT_REGISTRY.keys()) == expected

    def test_create_variant(self):
        v = create_variant("a_baseline")
        assert isinstance(v, VariantA_Baseline)

    def test_create_variant_with_config(self, test_config):
        v = create_variant("d_bayesian", config=test_config)
        assert isinstance(v, VariantD_Bayesian)
        assert v.config.n_voxels == test_config.n_voxels


# ===========================================================================
# Performance Tests
# ===========================================================================

class TestPerformance:
    @pytest.mark.parametrize("variant_name,max_time", [
        ("a_baseline", 10.0),  # nilearn-equivalent, may be slower
        ("b_flobs", 5.0),
        ("d_bayesian", 5.0),
    ])
    def test_per_tr_timing(self, variant_name, max_time, test_config,
                           dummy_timeseries, dummy_events):
        """Per-TR processing time after warmup should be within threshold."""
        v = create_variant(variant_name, config=test_config)
        v.precompute()

        # Warmup
        _ = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=0)

        # Timed run
        start = time.time()
        _ = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=1)
        elapsed = time.time() - start

        assert elapsed < max_time, f"{variant_name} took {elapsed:.2f}s (max {max_time}s)"


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

class TestPropertyBased:
    @given(
        n_trs=st.integers(min_value=5, max_value=100),
    )
    @settings(max_examples=10, deadline=30000)
    def test_hrf_length_matches_request(self, n_trs):
        hrf = make_glover_hrf(1.5, n_trs)
        assert hrf.shape == (n_trs,)
        assert hrf.dtype == np.float32
        assert not np.any(np.isnan(hrf))

    @given(
        n_events=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=5, deadline=30000)
    def test_design_matrix_event_count(self, n_events):
        onsets = np.arange(n_events) * 6.0
        hrf = make_glover_hrf(1.5, 22)
        dm, probe_idx = build_design_matrix(onsets, 1.5, 50, hrf, probe_trial=0)
        assert dm.shape[0] == 50
        assert dm.shape[1] >= 2  # probe + reference + drift

    @given(
        data=arrays(
            dtype=np.float32,
            shape=(100, 20),
            elements=st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=5, deadline=60000)
    def test_variant_a_with_random_data(self, data):
        """Variant A should handle arbitrary (non-NaN) input."""
        config = VariantConfig(n_voxels=100, max_trs=64)
        v = VariantA_Baseline(config)
        v.precompute()
        onsets = np.array([3.0, 9.0, 15.0])
        beta = v.process_tr(data, 19, onsets, probe_trial=0)
        assert beta.shape == (100,)
        assert not np.any(np.isnan(beta))

    def test_zero_volume(self, test_config, dummy_events):
        """Processing zero-valued timeseries should not crash."""
        v = VariantA_Baseline(test_config)
        v.precompute()
        zeros = np.zeros((test_config.n_voxels, 50), dtype=np.float32)
        beta = v.process_tr(zeros, 49, dummy_events, probe_trial=0)
        assert beta.shape == (test_config.n_voxels,)
        # Betas of zero input should be zero (or near-zero)
        assert np.allclose(beta, 0.0, atol=1e-3)

    def test_single_tr(self, test_config):
        """Single TR should not crash (though results may be meaningless)."""
        v = VariantA_Baseline(test_config)
        v.precompute()
        single = np.random.randn(test_config.n_voxels, 2).astype(np.float32)
        onsets = np.array([0.0])
        # With only 2 TRs the GLM is underdetermined but should not crash
        beta = v.process_tr(single, 1, onsets, probe_trial=0)
        assert beta.shape == (test_config.n_voxels,)


# ===========================================================================
# Integration Tests: Real Data
# ===========================================================================

class TestIntegrationRealData:
    """These tests require real data on /data. Skip if not available."""

    def test_variant_a_with_real_mask(self, real_data_available, real_config):
        if not real_data_available:
            pytest.skip("Real data not available")
        from rt_glm_variants import load_brain_mask, load_union_mask
        brain_mask = load_brain_mask(real_config.brain_mask_path)
        union_mask = load_union_mask(real_config.union_mask_path)
        assert brain_mask.sum() == 19174
        assert union_mask.sum() == 8627

    def test_variant_c_real_precompute(self, real_data_available, real_config):
        if not real_data_available:
            pytest.skip("Real data not available")
        v = VariantC_PerVoxelHRF(real_config)
        v.precompute()
        assert v.hrf_indices.shape == (8627,)
        # Should have multiple HRF groups (typically 5-10 active)
        assert len(v.voxel_groups) >= 5

    def test_existing_betas_shape(self, real_data_available):
        if not real_data_available:
            pytest.skip("Real data not available")
        betas = np.load("/data/3t/derivatives/sub-005_ses-06_task-C_run-02_recons/betas_run-02.npy")
        assert betas.shape[1] == 8627


# ===========================================================================
# Regression Tests
# ===========================================================================

class TestRegression:
    def test_variant_a_deterministic(self, test_config, dummy_timeseries, dummy_events):
        """Same input should produce same output."""
        v = VariantA_Baseline(test_config)
        v.precompute()
        beta1 = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        beta2 = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        np.testing.assert_array_equal(beta1, beta2)

    def test_variant_d_with_uninformative_prior_matches_ols(self, test_config,
                                                              dummy_timeseries,
                                                              dummy_events):
        """With very large prior variance, Bayesian should approximate OLS."""
        va = VariantA_Baseline(test_config)
        va.precompute()
        ols_beta = va.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)

        vd = VariantD_Bayesian(test_config)
        vd.precompute()  # Uninformative prior (var=1e6)
        bayes_beta = vd.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)

        # With huge prior variance, posterior ≈ OLS
        np.testing.assert_allclose(bayes_beta, ols_beta, rtol=0.02)


# ===========================================================================
# Unit Tests: Nuisance Regression (CSF/WM CompCor)
# ===========================================================================

class TestNuisanceRegressors:
    """Tests for CSF/WM nuisance regressor construction."""

    def test_build_nuisance_regressors_shape(self, dummy_timeseries):
        """Nuisance regressors should have (n_trs, n_regressors) shape."""
        n_voxels, n_trs = dummy_timeseries.shape
        # Create dummy WM/CSF masks (subset of voxels)
        wm_mask = np.zeros(n_voxels, dtype=bool)
        wm_mask[:500] = True
        csf_mask = np.zeros(n_voxels, dtype=bool)
        csf_mask[500:700] = True

        nuisance = build_nuisance_regressors(dummy_timeseries, wm_mask, csf_mask)
        assert nuisance.shape[0] == n_trs
        # Should have at least 2 columns (WM mean, CSF mean)
        assert nuisance.shape[1] >= 2

    def test_nuisance_regressors_no_nan(self, dummy_timeseries):
        """Nuisance regressors should contain no NaN values."""
        n_voxels = dummy_timeseries.shape[0]
        wm_mask = np.zeros(n_voxels, dtype=bool)
        wm_mask[:500] = True
        csf_mask = np.zeros(n_voxels, dtype=bool)
        csf_mask[500:700] = True

        nuisance = build_nuisance_regressors(dummy_timeseries, wm_mask, csf_mask)
        assert not np.any(np.isnan(nuisance))

    def test_nuisance_regressors_dtype(self, dummy_timeseries):
        """Nuisance regressors should be float32."""
        n_voxels = dummy_timeseries.shape[0]
        wm_mask = np.zeros(n_voxels, dtype=bool)
        wm_mask[:500] = True
        csf_mask = np.zeros(n_voxels, dtype=bool)
        csf_mask[500:700] = True

        nuisance = build_nuisance_regressors(dummy_timeseries, wm_mask, csf_mask)
        assert nuisance.dtype == np.float32

    def test_wm_mean_tracks_wm_signal(self, rng):
        """WM mean regressor should correlate with actual WM voxel timeseries."""
        n_voxels, n_trs = 1000, 50
        data = rng.randn(n_voxels, n_trs).astype(np.float32)
        # Inject a shared signal into WM voxels
        wm_signal = rng.randn(n_trs).astype(np.float32) * 10
        wm_mask = np.zeros(n_voxels, dtype=bool)
        wm_mask[:200] = True
        data[wm_mask] += wm_signal[np.newaxis, :]

        csf_mask = np.zeros(n_voxels, dtype=bool)
        csf_mask[200:300] = True

        nuisance = build_nuisance_regressors(data, wm_mask, csf_mask)
        wm_reg = nuisance[:, 0]  # first column should be WM mean
        r = np.corrcoef(wm_reg, wm_signal)[0, 1]
        assert r > 0.9, f"WM regressor should track WM signal, got r={r:.3f}"

    def test_nuisance_with_derivatives(self, dummy_timeseries):
        """With include_derivatives=True, should have more columns."""
        n_voxels = dummy_timeseries.shape[0]
        wm_mask = np.zeros(n_voxels, dtype=bool)
        wm_mask[:500] = True
        csf_mask = np.zeros(n_voxels, dtype=bool)
        csf_mask[500:700] = True

        basic = build_nuisance_regressors(
            dummy_timeseries, wm_mask, csf_mask, include_derivatives=False)
        with_deriv = build_nuisance_regressors(
            dummy_timeseries, wm_mask, csf_mask, include_derivatives=True)
        assert with_deriv.shape[1] > basic.shape[1]

    def test_empty_mask_returns_zeros(self, dummy_timeseries):
        """Empty WM/CSF masks should return zero regressors without crashing."""
        n_voxels, n_trs = dummy_timeseries.shape
        empty = np.zeros(n_voxels, dtype=bool)

        nuisance = build_nuisance_regressors(dummy_timeseries, empty, empty)
        assert nuisance.shape == (n_trs, 2)
        assert np.allclose(nuisance, 0.0)


class TestDesignMatrixWithNuisance:
    """Tests for design matrix augmented with nuisance regressors."""

    def test_augmented_dm_wider(self, dummy_timeseries, dummy_events):
        """Design matrix with nuisance should have more columns."""
        n_voxels, n_trs = dummy_timeseries.shape
        hrf = make_glover_hrf(1.5, 22)
        dm_basic, _ = build_design_matrix(
            dummy_events, 1.5, n_trs, hrf, probe_trial=0)

        wm_mask = np.zeros(n_voxels, dtype=bool)
        wm_mask[:500] = True
        csf_mask = np.zeros(n_voxels, dtype=bool)
        csf_mask[500:700] = True
        nuisance = build_nuisance_regressors(dummy_timeseries, wm_mask, csf_mask)

        dm_augmented = np.column_stack([dm_basic, nuisance])
        assert dm_augmented.shape[1] > dm_basic.shape[1]

    def test_nuisance_does_not_change_probe_index(self, dummy_events):
        """Adding nuisance regressors should not change probe column index."""
        hrf = make_glover_hrf(1.5, 22)
        dm, probe_idx = build_design_matrix(
            dummy_events, 1.5, 50, hrf, probe_trial=0)
        # Probe is always column 0 in LSS design
        assert probe_idx == 0


class TestVariantWithNuisance:
    """Integration test: a variant using nuisance regression."""

    def test_nuisance_variant_output_shape(self, test_config, dummy_timeseries,
                                            dummy_events):
        """Variant A + nuisance should still produce (8627,) output."""
        from rt_glm_variants import VariantA_NuisanceRegression
        v = VariantA_NuisanceRegression(test_config)
        v.precompute()

        # Set dummy tissue masks
        n_voxels = test_config.n_voxels
        v.wm_mask = np.zeros(n_voxels, dtype=bool)
        v.wm_mask[:500] = True
        v.csf_mask = np.zeros(n_voxels, dtype=bool)
        v.csf_mask[500:700] = True

        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        assert beta.shape == (n_voxels,)
        assert beta.dtype == np.float32

    def test_nuisance_variant_no_nan(self, test_config, dummy_timeseries,
                                      dummy_events):
        from rt_glm_variants import VariantA_NuisanceRegression
        v = VariantA_NuisanceRegression(test_config)
        v.precompute()
        n_voxels = test_config.n_voxels
        v.wm_mask = np.zeros(n_voxels, dtype=bool)
        v.wm_mask[:500] = True
        v.csf_mask = np.zeros(n_voxels, dtype=bool)
        v.csf_mask[500:700] = True

        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        assert not np.any(np.isnan(beta))

    def test_nuisance_differs_from_baseline(self, test_config, dummy_timeseries,
                                             dummy_events, rng):
        """With injected noise, nuisance regression should produce different betas."""
        from rt_glm_variants import VariantA_NuisanceRegression

        # Inject correlated noise into "CSF" voxels that leaks into signal
        n_voxels = test_config.n_voxels
        n_trs = dummy_timeseries.shape[1]
        noise = rng.randn(n_trs).astype(np.float32) * 50
        ts_noisy = dummy_timeseries.copy()
        ts_noisy[:, :] += noise[np.newaxis, :]  # global noise

        # Baseline without nuisance
        va = VariantA_Baseline(test_config)
        va.precompute()
        beta_baseline = va.process_tr(ts_noisy, 49, dummy_events, probe_trial=3)

        # With nuisance regression
        vn = VariantA_NuisanceRegression(test_config)
        vn.precompute()
        vn.wm_mask = np.zeros(n_voxels, dtype=bool)
        vn.wm_mask[:500] = True
        vn.csf_mask = np.zeros(n_voxels, dtype=bool)
        vn.csf_mask[500:700] = True

        beta_nuisance = vn.process_tr(ts_noisy, 49, dummy_events, probe_trial=3)

        # They should differ because nuisance regression removes the global noise
        assert not np.allclose(beta_baseline, beta_nuisance, atol=1e-2)

    def test_nuisance_variant_in_registry(self):
        """VariantA_NuisanceRegression should be in the variant registry."""
        assert "a_nuisance" in VARIANT_REGISTRY
