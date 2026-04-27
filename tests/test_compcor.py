"""Red-green TDD for jaxoccoli/compcor.py."""
from __future__ import annotations

import numpy as np
import pytest

from jaxoccoli.compcor import (
    acompcor_components,
    append_compcor_to_design,
    tcompcor_components,
)


def test_acompcor_components_recovers_known_signal():
    """A noise pool with a dominant rank-1 temporal signal should yield a top
    component that strongly correlates with the injected signal."""
    rng = np.random.default_rng(0)
    V, T = 200, 80
    physio = np.sin(np.linspace(0, 6 * np.pi, T)).astype(np.float32)
    # CSF/WM mask: voxels 100..199 carry the physio signal + noise
    Y = 0.05 * rng.normal(size=(V, T)).astype(np.float32)
    csfwm = np.zeros(V, dtype=bool); csfwm[100:] = True
    Y[100:] += rng.normal(0.5, 0.2, size=100)[:, None] * physio[None, :]
    comps, ev = acompcor_components(Y, csfwm, n_components=3)
    assert comps.shape == (T, 3)
    # First component should correlate strongly with physio (sign-agnostic)
    r = abs(np.corrcoef(comps[:, 0], physio)[0, 1])
    assert r > 0.9, f"top component did not recover physio (r={r:.3f})"


def test_acompcor_explained_variance_descending():
    rng = np.random.default_rng(0)
    V, T = 50, 60
    Y = rng.normal(size=(V, T)).astype(np.float32)
    csfwm = np.ones(V, dtype=bool)
    _, ev = acompcor_components(Y, csfwm, n_components=5)
    assert np.all(np.diff(ev) <= 0), "explained variance should be descending"


def test_tcompcor_picks_high_variance_voxels():
    rng = np.random.default_rng(0)
    V, T = 1000, 80
    Y = 0.1 * rng.normal(size=(V, T)).astype(np.float32)
    # Make voxels 0..50 very high-variance
    Y[:50] *= 10
    _, _, pool_mask = tcompcor_components(Y, n_components=3,
                                            top_variance_frac=0.05)
    # All 50 high-variance voxels should be in the pool
    assert pool_mask[:50].all()


def test_append_compcor_to_design_stacks_columns():
    T, P, K = 60, 4, 3
    design = np.random.normal(size=(T, P)).astype(np.float32)
    comps = np.random.normal(size=(T, K)).astype(np.float32)
    aug = append_compcor_to_design(design, comps)
    assert aug.shape == (T, P + K)
    np.testing.assert_array_equal(aug[:, :P], design)
    np.testing.assert_array_equal(aug[:, P:], comps)


def test_append_compcor_to_design_rejects_T_mismatch():
    design = np.zeros((60, 3), dtype=np.float32)
    comps = np.zeros((40, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        append_compcor_to_design(design, comps)


def test_acompcor_too_small_pool_raises():
    Y = np.zeros((10, 30), dtype=np.float32)
    csfwm = np.zeros(10, dtype=bool); csfwm[:2] = True   # only 2 voxels
    with pytest.raises(ValueError):
        acompcor_components(Y, csfwm, n_components=5)
