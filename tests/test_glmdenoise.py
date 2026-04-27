"""Red-green TDD for jaxoccoli/glmdenoise.py."""
from __future__ import annotations

import numpy as np
import pytest

from jaxoccoli.glmdenoise import (
    extract_noise_components,
    glmdenoise_fit,
    per_voxel_r2,
    select_noise_pool,
)


# ----------------------------- per_voxel_r2 ---------------------------------

def test_per_voxel_r2_perfect_fit_returns_one():
    rng = np.random.default_rng(0)
    T, V, P = 50, 20, 3
    X = rng.normal(size=(T, P)).astype(np.float32)
    true_betas = rng.normal(size=(V, P)).astype(np.float32)
    Y = true_betas @ X.T                                  # zero residual
    r2 = per_voxel_r2(Y, X, true_betas)
    np.testing.assert_allclose(r2, 1.0, atol=1e-5)


def test_per_voxel_r2_zero_fit_when_betas_random():
    """A random β unrelated to Y should give R² ≪ 1."""
    rng = np.random.default_rng(0)
    T, V, P = 200, 50, 3
    X = rng.normal(size=(T, P)).astype(np.float32)
    Y = rng.normal(size=(V, T)).astype(np.float32)        # uncorrelated
    bad_betas = rng.normal(size=(V, P)).astype(np.float32)
    r2 = per_voxel_r2(Y, X, bad_betas)
    assert r2.mean() < 0.2  # should be near 0 or negative


# --------------------------- select_noise_pool ------------------------------

def test_select_noise_pool_picks_consistently_low_r2_voxels():
    """Voxel 0 has R² ≈ 0 in every run; voxel 1 has high R² always.
    Pool should include voxel 0 and exclude voxel 1.
    """
    n_runs, V = 4, 10
    per_run_r2 = np.full((n_runs, V), 0.5, dtype=np.float32)
    per_run_r2[:, 0] = -0.05     # consistently low
    per_run_r2[:, 1] = 0.8       # consistently high
    pool = select_noise_pool(per_run_r2, threshold=0.0, max_pool_frac=0.5)
    assert pool[0]
    assert not pool[1]


def test_select_noise_pool_capped_by_max_pool_frac():
    n_runs, V = 3, 100
    per_run_r2 = -np.ones((n_runs, V), dtype=np.float32) * 0.1  # all eligible
    pool = select_noise_pool(per_run_r2, threshold=0.0, max_pool_frac=0.2)
    assert pool.sum() <= int(np.floor(V * 0.2))


# ----------------------- extract_noise_components ---------------------------

def test_extract_noise_components_returns_correct_shape_and_unit_norm():
    """Each returned component is a unit-norm temporal vector."""
    rng = np.random.default_rng(0)
    V_pool, T = 100, 60
    Y = rng.normal(size=(V_pool, T)).astype(np.float32)
    comps, ev = extract_noise_components(Y, max_K=5)
    assert comps.shape == (T, 5)
    norms = np.linalg.norm(comps, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_extract_noise_components_recovers_known_low_rank_signal():
    """A rank-2 noise pool should have its top-2 components dominate."""
    rng = np.random.default_rng(0)
    V_pool, T = 80, 100
    # Inject 2 known temporal signals
    sig1 = np.sin(np.linspace(0, 4 * np.pi, T)).astype(np.float32)
    sig2 = np.cos(np.linspace(0, 6 * np.pi, T)).astype(np.float32)
    a1 = rng.normal(size=V_pool).astype(np.float32)
    a2 = rng.normal(size=V_pool).astype(np.float32)
    Y = np.outer(a1, sig1) + np.outer(a2, sig2)
    Y = Y + 0.01 * rng.normal(size=Y.shape).astype(np.float32)
    comps, ev = extract_noise_components(Y, max_K=5)
    # Top 2 components should explain almost all the variance
    assert ev[:2].sum() > 0.95


# ----------------------------- glmdenoise_fit -------------------------------

def _make_synth_runs(n_runs=4, V=80, T=120, n_trials=10, seed=0):
    """Factory for synthetic multi-run data with shared noise structure."""
    rng = np.random.default_rng(seed)
    runs_Y, runs_X = [], []
    # Shared physiological noise template across all runs (different gains)
    physio = np.sin(np.linspace(0, 8 * np.pi, T)).astype(np.float32)
    for r in range(n_runs):
        # Build LSS-like design: probe + nuisance + intercept
        onsets_tr = np.linspace(8, T - 8, n_trials).astype(np.float32)
        probe = np.zeros(T, dtype=np.float32)
        probe[int(onsets_tr[0])] = 1.0
        nuisance = np.zeros(T, dtype=np.float32)
        for o in onsets_tr[1:]:
            nuisance[int(o)] = 1.0
        # No HRF convolution for simplicity — just impulse design
        X = np.stack([probe, nuisance, np.ones(T)], axis=1).astype(np.float32)

        # Active half: signal at probe onset; inactive half: no signal
        active = np.zeros(V, dtype=bool)
        active[: V // 2] = True
        gain = rng.normal(0.4, 0.1, size=V).astype(np.float32) * active
        Y = np.outer(gain, probe + 0 * nuisance)              # task signal
        # Physiology in non-active voxels (CSF/WM)
        physio_amp = rng.normal(0, 0.3, size=V).astype(np.float32) * (~active)
        Y = Y + np.outer(physio_amp, physio)
        # i.i.d. noise
        Y = Y + rng.normal(0, 0.1, size=Y.shape).astype(np.float32)
        runs_Y.append(Y); runs_X.append(X)
    return runs_Y, runs_X, active


def _ols(X, Y):
    XtX_inv = np.linalg.inv(X.T @ X + 1e-6 * np.eye(X.shape[1]))
    return Y @ X @ XtX_inv.T                                   # (V, P)


def test_glmdenoise_fit_picks_K_ge_one_when_noise_structured():
    runs_Y, runs_X, _ = _make_synth_runs(seed=1)
    res = glmdenoise_fit(runs_Y, runs_X, fit_fn=_ols, max_K=8)
    # Structured shared physio across runs → CV should pick K ≥ 1
    assert res.K_chosen >= 1


def test_glmdenoise_fit_returns_fields():
    runs_Y, runs_X, _ = _make_synth_runs(seed=2)
    res = glmdenoise_fit(runs_Y, runs_X, fit_fn=_ols, max_K=5)
    assert res.noise_components.shape[1] == res.K_chosen
    assert res.noise_pool_mask.shape == (runs_Y[0].shape[0],)
    assert res.cv_curve.shape == (6,)              # 0..max_K
    assert res.explained_variance.shape == (5,)
