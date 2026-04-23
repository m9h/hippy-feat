"""
Tests for Variant G — Bayesian first-level GLM (conjugate RT path).

Phase 1 scope: G-conjugate wired into the RTPreprocessingVariant framework
using jaxoccoli.bayesian_beta.make_ar1_conjugate_glm. The key novel output
is a per-voxel posterior variance on the probe beta, exposed via
v._last_beta_var and a confidence_mask helper.

NUTS, MRF spatial prior, parametric HRF, and Riera (Level 3) HRF are
Phases 2-4 and are not tested here.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rt_glm_variants import (
    VariantConfig,
    VariantA_Baseline,
    VariantD_Bayesian,
    VariantG_Bayesian,
    VARIANT_REGISTRY,
    create_variant,
    confidence_mask,
)


# ===========================================================================
# Precompute & shape
# ===========================================================================

class TestVariantGPrecompute:
    def test_precompute_with_training_data(self, test_config, dummy_training_betas):
        v = VariantG_Bayesian(test_config)
        v.precompute(training_betas=dummy_training_betas)
        assert v._precomputed
        assert v.prior_mean.shape == (test_config.n_voxels,)
        assert v.prior_var.shape == (test_config.n_voxels,)
        assert np.all(v.prior_var > 0)

    def test_precompute_uninformative_prior(self, test_config):
        v = VariantG_Bayesian(test_config)
        v.precompute()
        assert v._precomputed
        assert np.all(v.prior_var > 1e5)

    def test_process_tr_shape(self, test_config, dummy_timeseries, dummy_events):
        v = VariantG_Bayesian(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=4)
        assert beta.shape == (test_config.n_voxels,)
        assert beta.dtype == np.float32

    def test_process_tr_no_nan(self, test_config, dummy_timeseries, dummy_events):
        v = VariantG_Bayesian(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=4)
        assert not np.any(np.isnan(beta))
        assert not np.any(np.isinf(beta))


# ===========================================================================
# Variance output — the novel feature
# ===========================================================================

class TestVariantGUncertainty:
    def test_emits_beta_var(self, test_config, dummy_timeseries, dummy_events):
        """After process_tr, _last_beta_var must be populated with correct shape."""
        v = VariantG_Bayesian(test_config)
        v.precompute()
        _ = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=4)
        assert hasattr(v, "_last_beta_var")
        assert v._last_beta_var.shape == (test_config.n_voxels,)
        assert v._last_beta_var.dtype == np.float32

    def test_beta_var_all_positive(self, test_config, dummy_timeseries, dummy_events):
        """Posterior variance must be strictly positive, no NaN."""
        v = VariantG_Bayesian(test_config)
        v.precompute()
        _ = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=4)
        assert np.all(v._last_beta_var > 0)
        assert not np.any(np.isnan(v._last_beta_var))
        assert not np.any(np.isinf(v._last_beta_var))

    def test_variance_decreases_with_more_trs(self, test_config, rng, dummy_events):
        """
        Property: more data → tighter posterior. Compare beta_var at TR=20 vs TR=80
        on the SAME underlying generative process. Variance at 80 should be
        strictly smaller (on mean) than at 20.
        """
        # Build long timeseries, then slice
        long_ts = rng.randn(test_config.n_voxels, 100).astype(np.float32) * 50 + 500
        v = VariantG_Bayesian(test_config)
        v.precompute()

        _ = v.process_tr(long_ts[:, :20], 19, dummy_events, probe_trial=2)
        var_short = v._last_beta_var.copy()

        v2 = VariantG_Bayesian(test_config)
        v2.precompute()
        _ = v2.process_tr(long_ts[:, :80], 79, dummy_events, probe_trial=2)
        var_long = v2._last_beta_var.copy()

        # Mean posterior variance should drop as we add data
        assert var_long.mean() < var_short.mean(), (
            f"Expected posterior variance to decrease with more TRs: "
            f"mean var at 20 TRs = {var_short.mean():.3g}, at 80 TRs = {var_long.mean():.3g}"
        )


# ===========================================================================
# Prior behavior — G-conjugate as "D done right"
# ===========================================================================

class TestVariantGPrior:
    def test_uninformative_prior_close_to_ols(self, test_config, dummy_timeseries,
                                               dummy_events):
        """
        With an uninformative prior (huge prior variance), G-conjugate's posterior
        mean should be close to Variant A's OLS estimate on most voxels.

        Not bit-exact — G uses AR(1) prewhitening, A uses OLS. Expect r > 0.9.
        """
        g = VariantG_Bayesian(test_config)
        g.precompute()  # uninformative
        g_beta = g.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)

        a = VariantA_Baseline(test_config)
        a.precompute()
        a_beta = a.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)

        r = np.corrcoef(g_beta, a_beta)[0, 1]
        assert r > 0.9, f"G vs A correlation {r:.3f} below 0.9 — AR(1) drift is too large"

    def test_strong_prior_shrinks_toward_prior(self, test_config, dummy_timeseries,
                                                dummy_events, rng):
        """
        With a very tight prior (low prior variance) at a nonzero prior mean,
        posterior mean should move toward the prior, not toward zero.
        """
        n = test_config.n_voxels
        # Prior centered at a specific value, very tight
        training_betas = rng.randn(200, n).astype(np.float32) * 0.01 + 5.0  # prior ~ N(5, 0.01^2)

        v = VariantG_Bayesian(test_config)
        v.precompute(training_betas=training_betas)
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)

        # With tight prior at 5.0, posterior should be much closer to 5.0 than to OLS
        mean_beta = beta.mean()
        assert abs(mean_beta - 5.0) < abs(mean_beta - 0.0), (
            f"posterior mean {mean_beta:.3f} did not shrink toward prior (5.0)"
        )

    def test_posterior_between_prior_and_ols(self, test_config, dummy_timeseries,
                                              dummy_events, dummy_training_betas):
        """Posterior lies between prior and data for most voxels (mirrors TestVariantD)."""
        v = VariantG_Bayesian(test_config)
        v.precompute(training_betas=dummy_training_betas)

        a = VariantA_Baseline(test_config)
        a.precompute()
        ols_beta = a.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)
        bayes_beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=3)

        prior = v.prior_mean
        lo = np.minimum(prior, ols_beta)
        hi = np.maximum(prior, ols_beta)
        tol = 0.1 * (hi - lo + 1e-6)
        in_range = (bayes_beta >= lo - tol) & (bayes_beta <= hi + tol)
        assert in_range.mean() > 0.9


# ===========================================================================
# Confidence-gating helper
# ===========================================================================

class TestConfidenceMask:
    def test_threshold_zero_excludes_all(self):
        """threshold=0 with positive variances → all voxels excluded (beta_std > 0)."""
        beta_mean = np.array([1.0, 2.0, 3.0])
        beta_var = np.array([0.1, 0.1, 0.1])
        mask = confidence_mask(beta_mean, beta_var, threshold=0.0)
        assert mask.dtype == bool
        assert mask.shape == (3,)
        assert not mask.any()

    def test_threshold_infinity_includes_all(self):
        """threshold=inf → every voxel is high-confidence."""
        beta_mean = np.array([1.0, 2.0, 3.0])
        beta_var = np.array([0.1, 1.0, 100.0])
        mask = confidence_mask(beta_mean, beta_var, threshold=np.inf)
        assert mask.all()

    def test_monotone_in_threshold(self, rng):
        """Higher threshold ⇒ stricter SNR cutoff ⇒ fewer voxels (subset property).

        Matches the ratio semantic documented on test_ratio_threshold_semantics:
        mask = True where |beta_mean| / sqrt(beta_var) > threshold.
        """
        beta_mean = rng.randn(100).astype(np.float32)
        beta_var = np.abs(rng.randn(100).astype(np.float32)) + 1e-3
        m1 = confidence_mask(beta_mean, beta_var, threshold=0.5)
        m2 = confidence_mask(beta_mean, beta_var, threshold=2.0)
        m3 = confidence_mask(beta_mean, beta_var, threshold=10.0)
        # m3 ⊆ m2 ⊆ m1
        assert (m3 & ~m2).sum() == 0
        assert (m2 & ~m1).sum() == 0

    def test_ratio_threshold_semantics(self):
        """
        Default semantics: mask is True where |beta_mean| / sqrt(beta_var) > threshold.
        i.e. threshold is a z-score-like SNR cutoff.
        """
        beta_mean = np.array([1.0, 3.0, 0.1])
        beta_var = np.array([1.0, 1.0, 1.0])  # std = 1.0 each
        # |beta| / std = [1.0, 3.0, 0.1]. Threshold 2.0 keeps only the 3.0 voxel.
        mask = confidence_mask(beta_mean, beta_var, threshold=2.0)
        assert mask.tolist() == [False, True, False]


# ===========================================================================
# AR(1) sanity
# ===========================================================================

class TestVariantGAR1:
    def test_no_crash_when_residuals_white(self, test_config, dummy_timeseries,
                                            dummy_events):
        """The AR(1) estimation path should not crash on near-white residuals."""
        v = VariantG_Bayesian(test_config)
        v.precompute()
        beta = v.process_tr(dummy_timeseries, 49, dummy_events, probe_trial=2)
        assert np.isfinite(beta).all()


# ===========================================================================
# Registry
# ===========================================================================

class TestVariantGRegistry:
    def test_g_bayesian_registered(self):
        assert "g_bayesian" in VARIANT_REGISTRY
        assert VARIANT_REGISTRY["g_bayesian"] is VariantG_Bayesian

    def test_create_variant_g(self, test_config):
        v = create_variant("g_bayesian", config=test_config)
        assert isinstance(v, VariantG_Bayesian)


# ===========================================================================
# Property-based
# ===========================================================================

class TestVariantGProperties:
    @given(
        n_voxels=st.integers(min_value=20, max_value=200),
        n_trs=st.integers(min_value=30, max_value=80),
        probe_trial=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=5, deadline=30000)
    def test_posterior_variance_strictly_positive(self, n_voxels, n_trs, probe_trial):
        """For any valid input, posterior variance is strictly positive."""
        rng = np.random.RandomState(42)
        config = VariantConfig(n_voxels=n_voxels, max_trs=max(n_trs + 10, 64))
        ts = rng.randn(n_voxels, n_trs).astype(np.float32) * 50 + 500
        onsets = np.arange(6.0, 6.0 + 10 * 3.0, 3.0)

        v = VariantG_Bayesian(config)
        v.precompute()
        _ = v.process_tr(ts, n_trs - 1, onsets, probe_trial=probe_trial)
        assert np.all(v._last_beta_var > 0)
        assert np.all(np.isfinite(v._last_beta_var))
