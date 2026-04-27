"""Red-green TDD for jaxoccoli/fracridge.py."""
from __future__ import annotations

import numpy as np
import pytest

from jaxoccoli.fracridge import fracridge_cv, fracridge_solve


def test_fracridge_solve_at_frac_one_equals_ols():
    rng = np.random.default_rng(0)
    T, P = 100, 5
    X = rng.normal(size=(T, P)).astype(np.float32)
    true_b = rng.normal(size=P).astype(np.float32)
    y = X @ true_b + 0.05 * rng.normal(size=T).astype(np.float32)
    # OLS reference
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    betas, lambdas = fracridge_solve(X, y, np.array([1.0]))
    np.testing.assert_allclose(betas[0], beta_ols, atol=1e-3)
    np.testing.assert_allclose(lambdas[0], 0.0, atol=1e-3)


def test_fracridge_solve_norms_are_monotone_in_frac():
    """For 0 < f1 < f2 ≤ 1, ||β(f1)|| < ||β(f2)|| (excluding pathologies)."""
    rng = np.random.default_rng(0)
    T, P = 80, 4
    X = rng.normal(size=(T, P)).astype(np.float32)
    y = rng.normal(size=T).astype(np.float32)
    fracs = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
    betas, _ = fracridge_solve(X, y, fracs)
    norms = np.linalg.norm(betas, axis=1)
    assert np.all(np.diff(norms) > 0), \
        f"Norms not monotone in frac: {norms}"


def test_fracridge_solve_norm_ratio_matches_frac():
    """||β(f)|| should be close to f · ||β_OLS|| (the fracridge invariant)."""
    rng = np.random.default_rng(0)
    T, P = 200, 6
    X = rng.normal(size=(T, P)).astype(np.float32)
    y = (X @ rng.normal(size=P) + 0.1 * rng.normal(size=T)).astype(np.float32)
    fracs = np.array([0.2, 0.5, 0.8])
    betas, _ = fracridge_solve(X, y, fracs)
    beta_ols_norm = np.linalg.norm(np.linalg.lstsq(X, y, rcond=None)[0])
    for i, f in enumerate(fracs):
        observed_ratio = float(np.linalg.norm(betas[i])) / beta_ols_norm
        np.testing.assert_allclose(observed_ratio, f, atol=1e-2)


def test_fracridge_cv_selects_high_frac_for_well_conditioned():
    """When X is well-conditioned and signal is strong, CV-best f → 1."""
    rng = np.random.default_rng(0)
    T, P = 200, 4
    # Two orthogonal-ish runs
    X1 = rng.normal(size=(T, P)).astype(np.float32)
    X2 = rng.normal(size=(T, P)).astype(np.float32)
    V = 10
    true_betas = rng.normal(size=(V, P)).astype(np.float32) * 2.0
    Y1 = (true_betas @ X1.T + 0.05 * rng.normal(size=(V, T))).astype(np.float32)
    Y2 = (true_betas @ X2.T + 0.05 * rng.normal(size=(V, T))).astype(np.float32)
    f_chosen, cv_score = fracridge_cv(
        Y_train=[Y1], X_train=[X1],
        Y_test=[Y2], X_test=[X2],
        fracs=np.linspace(0.1, 1.0, 10),
    )
    assert f_chosen.mean() >= 0.7, \
        f"Well-conditioned data should pick high f; got {f_chosen.mean()}"
    assert cv_score.mean() > 0.9


def test_fracridge_cv_selects_low_frac_for_noisy_target():
    """When the target is mostly noise, CV-best f → small (heavy shrinkage)."""
    rng = np.random.default_rng(0)
    T, P = 100, 8
    X1 = rng.normal(size=(T, P)).astype(np.float32)
    X2 = rng.normal(size=(T, P)).astype(np.float32)
    V = 10
    Y1 = rng.normal(0, 1, size=(V, T)).astype(np.float32)
    Y2 = rng.normal(0, 1, size=(V, T)).astype(np.float32)  # independent noise
    f_chosen, _ = fracridge_cv(
        Y_train=[Y1], X_train=[X1],
        Y_test=[Y2], X_test=[X2],
        fracs=np.linspace(0.05, 1.0, 20),
    )
    # When there is no real signal, heavier shrinkage should win on CV
    assert f_chosen.mean() < 0.7
