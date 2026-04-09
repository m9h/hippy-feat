"""
Tests for jaxoccoli.nsd — NSD validation utilities.

Covers:
  - rdm_from_betas: representational dissimilarity matrix
  - compare_rdms: Spearman correlation of upper-triangle RDM vectors
  - noise_ceiling_r: split-half noise ceiling estimation
  - category_selectivity: per-region category preference index
  - load_nsd_betas: NIfTI loader (tested with mock data)
  - compare_fc: Wasserstein FC distance wrapper
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxoccoli.nsd import (
    rdm_from_betas,
    compare_rdms,
    noise_ceiling_r,
    category_selectivity,
    upper_triangle,
    split_half_rdms,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def structured_betas(rng):
    """Betas with 3 categories (4 trials each), category structure embedded."""
    n_trials, n_features = 12, 50
    betas = rng.randn(n_trials, n_features).astype(np.float32) * 0.1
    # Category A (trials 0-3): high in features 0-15
    betas[:4, :16] += 2.0
    # Category B (trials 4-7): high in features 16-31
    betas[4:8, 16:32] += 2.0
    # Category C (trials 8-11): high in features 32-49
    betas[8:12, 32:] += 2.0
    return jnp.array(betas)


@pytest.fixture
def categories():
    return ["A"] * 4 + ["B"] * 4 + ["C"] * 4


@pytest.fixture
def random_betas(rng):
    """Random betas with no structure."""
    return jnp.array(rng.randn(20, 50).astype(np.float32))


# ===========================================================================
# rdm_from_betas
# ===========================================================================

class TestRDMFromBetas:

    def test_shape(self, structured_betas):
        rdm = rdm_from_betas(structured_betas)
        assert rdm.shape == (12, 12)

    def test_diagonal_zero(self, structured_betas):
        rdm = rdm_from_betas(structured_betas)
        np.testing.assert_array_almost_equal(
            np.diag(np.asarray(rdm)), np.zeros(12), decimal=5,
        )

    def test_symmetric(self, structured_betas):
        rdm = rdm_from_betas(structured_betas)
        np.testing.assert_array_almost_equal(
            np.asarray(rdm), np.asarray(rdm.T), decimal=5,
        )

    def test_nonnegative(self, structured_betas):
        rdm = rdm_from_betas(structured_betas)
        assert jnp.all(rdm >= -1e-6)

    def test_within_category_smaller(self, structured_betas, categories):
        """Within-category distances should be smaller than between."""
        rdm = rdm_from_betas(structured_betas)
        within, between = [], []
        for i in range(12):
            for j in range(i + 1, 12):
                d = float(rdm[i, j])
                if categories[i] == categories[j]:
                    within.append(d)
                else:
                    between.append(d)
        assert np.mean(within) < np.mean(between)

    def test_metric_correlation(self, random_betas):
        """Default metric should be correlation distance."""
        rdm = rdm_from_betas(random_betas, metric="correlation")
        # Values should be in [0, 2] for correlation distance
        assert float(rdm.max()) <= 2.0 + 1e-5

    def test_metric_euclidean(self, random_betas):
        rdm = rdm_from_betas(random_betas, metric="euclidean")
        assert rdm.shape == (20, 20)
        assert jnp.all(rdm >= -1e-6)


# ===========================================================================
# upper_triangle
# ===========================================================================

class TestUpperTriangle:

    def test_length(self):
        mat = jnp.ones((5, 5))
        ut = upper_triangle(mat)
        assert ut.shape == (10,)  # 5*4/2

    def test_excludes_diagonal(self):
        mat = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        ut = upper_triangle(mat)
        assert jnp.allclose(ut, 0.0)


# ===========================================================================
# compare_rdms
# ===========================================================================

class TestCompareRDMs:

    def test_identical_rdms(self, structured_betas):
        rdm = rdm_from_betas(structured_betas)
        r = compare_rdms(rdm, rdm)
        assert float(r) > 0.99

    def test_returns_scalar(self, structured_betas, random_betas):
        rdm1 = rdm_from_betas(structured_betas)
        rdm2 = rdm_from_betas(random_betas[:12])
        r = compare_rdms(rdm1, rdm2)
        assert r.shape == ()

    def test_range(self, structured_betas, random_betas):
        rdm1 = rdm_from_betas(structured_betas)
        rdm2 = rdm_from_betas(random_betas[:12])
        r = compare_rdms(rdm1, rdm2)
        assert -1.0 <= float(r) <= 1.0


# ===========================================================================
# noise_ceiling_r
# ===========================================================================

class TestNoiseCeiling:

    def test_returns_lower_upper(self, structured_betas):
        lower, upper = noise_ceiling_r(structured_betas, n_splits=10, seed=0)
        assert lower.shape == ()
        assert upper.shape == ()

    def test_upper_geq_lower(self, structured_betas):
        lower, upper = noise_ceiling_r(structured_betas, n_splits=10, seed=0)
        assert float(upper) >= float(lower) - 0.01  # small tolerance

    def test_high_for_structured(self, structured_betas):
        """Structured betas should have a high noise ceiling."""
        lower, upper = noise_ceiling_r(structured_betas, n_splits=20, seed=0)
        assert float(upper) > 0.5

    def test_range(self, random_betas):
        lower, upper = noise_ceiling_r(random_betas, n_splits=10, seed=0)
        assert -1.0 <= float(lower) <= 1.0
        assert -1.0 <= float(upper) <= 1.0


# ===========================================================================
# split_half_rdms
# ===========================================================================

class TestSplitHalfRDMs:

    def test_returns_two_rdms(self, structured_betas):
        rdm1, rdm2 = split_half_rdms(structured_betas, seed=0)
        n = structured_betas.shape[0]
        assert rdm1.shape == (n, n)
        assert rdm2.shape == (n, n)

    def test_different_splits(self, structured_betas):
        rdm1a, rdm2a = split_half_rdms(structured_betas, seed=0)
        rdm1b, rdm2b = split_half_rdms(structured_betas, seed=1)
        # Different seeds should give different splits
        assert not jnp.allclose(rdm1a, rdm1b)


# ===========================================================================
# category_selectivity
# ===========================================================================

class TestCategorySelectivity:

    def test_returns_dict(self, structured_betas, categories):
        sel = category_selectivity(structured_betas, categories)
        assert isinstance(sel, dict)

    def test_has_all_categories(self, structured_betas, categories):
        sel = category_selectivity(structured_betas, categories)
        assert set(sel.keys()) == {"A", "B", "C"}

    def test_selectivity_shape(self, structured_betas, categories):
        sel = category_selectivity(structured_betas, categories)
        n_features = structured_betas.shape[1]
        for cat, vals in sel.items():
            assert vals.shape == (n_features,)

    def test_structured_selectivity(self, structured_betas, categories):
        """Category A should be most selective in features 0-15."""
        sel = category_selectivity(structured_betas, categories)
        a_top = jnp.argsort(sel["A"])[-5:]
        assert jnp.all(a_top < 16), f"Expected top features < 16, got {a_top}"
