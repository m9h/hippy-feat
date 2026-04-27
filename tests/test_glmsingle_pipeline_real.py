"""Real-data integration tests for GLMdenoise + fracridge + CompCor.

Uses sub-005 ses-01 recognition runs from the RT-MindEye paper checkpoint
(local at /data/derivatives/rtmindeye_paper/). These tests SKIP gracefully
if the data isn't available, so they can run on any developer machine
without forcing the full dataset to be present.

Why real data: synthetic tests pin the math; these pin sensible behavior
on actual fMRI (signal/noise mix, drift structure, condition number),
which the math-only tests can't.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from jaxoccoli.compcor import (
    acompcor_components,
    append_compcor_to_design,
    tcompcor_components,
)
from jaxoccoli.fracridge import fracridge_cv, fracridge_solve
from jaxoccoli.glmdenoise import glmdenoise_fit, per_voxel_r2
from jaxoccoli.realtime import build_lss_design_matrix, make_glover_hrf


PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
FMRIPREP_ROOT = (PAPER_ROOT / "fmriprep_mindeye" / "data_sub-005"
                 / "bids" / "derivatives" / "fmriprep" / "sub-005")
EVENTS_DIR = RT3T / "events"
BRAIN_MASK = RT3T / "sub-005_final_mask.nii.gz"


def _data_available() -> bool:
    return (BRAIN_MASK.exists()
            and (FMRIPREP_ROOT / "ses-01" / "func").exists()
            and (EVENTS_DIR / "sub-005_ses-01_task-C_run-01_events.tsv").exists())


@pytest.fixture(scope="module")
def sub005_ses01_runs():
    """Load 3 runs of sub-005 ses-01 recognition data.

    Returns a list of (Y, design, probe_col, onsets, T) tuples, masked to
    the paper's 19174-voxel brain mask (NOT the 2792-voxel relmask, since
    GLMdenoise needs more voxels to find a noise pool).
    """
    if not _data_available():
        pytest.skip("sub-005 paper data not available locally")
    import nibabel as nib
    import pandas as pd

    flat_brain = (nib.load(BRAIN_MASK).get_fdata() > 0).flatten()
    runs = []
    for run in (1, 2, 3):
        bold_path = (FMRIPREP_ROOT / "ses-01" / "func"
                     / f"sub-005_ses-01_task-C_run-{run:02d}"
                       f"_space-T1w_desc-preproc_bold.nii.gz")
        events_path = (EVENTS_DIR
                       / f"sub-005_ses-01_task-C_run-{run:02d}_events.tsv")
        if not bold_path.exists() or not events_path.exists():
            pytest.skip(f"missing run {run} data for sub-005 ses-01")
        img = nib.load(bold_path)
        vol = img.get_fdata()
        T = vol.shape[-1]
        Y = vol.reshape(-1, T)[flat_brain].astype(np.float32)   # (V, T)

        events = pd.read_csv(events_path, sep="\t")
        onsets = events["onset"].astype(float).values
        onsets = (onsets - onsets[0]).astype(np.float32)
        tr = 1.5
        hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
        design, probe_col = build_lss_design_matrix(
            onsets, tr, T, hrf, probe_trial=0,
        )
        runs.append((Y, design, probe_col, onsets, T))
    return runs


# ----------------------- tCompCor on real data ------------------------------

def test_tcompcor_runs_on_real_data(sub005_ses01_runs):
    """Smoke test: tCompCor returns sensible output on real fMRI.
    Detailed component-content assertions live in the synthetic tests;
    real fMRI varies too much subject-to-subject for tight thresholds."""
    Y, _, _, _, T = sub005_ses01_runs[0]
    comps, ev, pool_mask = tcompcor_components(Y, n_components=5,
                                                 top_variance_frac=0.02)
    assert comps.shape == (T, 5)
    assert not np.isnan(comps).any()
    assert ev[:5].sum() > 0.10, \
        f"top-5 components explain only {ev[:5].sum():.2%} — sanity floor"
    # Components should be unit-norm (per acompcor_components contract)
    norms = np.linalg.norm(comps, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


# ----------------------- GLMdenoise on real data ----------------------------

def _ols(X, Y):
    XtX_inv = np.linalg.inv(X.T @ X + 1e-6 * np.eye(X.shape[1]))
    return Y @ X @ XtX_inv.T


def test_glmdenoise_picks_K_at_least_one_on_real_data(sub005_ses01_runs):
    """On real fMRI with ≥3 runs and structured drift / physio, GLMdenoise
    LOO-CV should pick K ≥ 1 (i.e., adding ≥1 noise component improves
    held-out R² over the no-component baseline)."""
    Ys = [r[0] for r in sub005_ses01_runs]
    Xs = [r[1] for r in sub005_ses01_runs]
    res = glmdenoise_fit(Ys, Xs, fit_fn=_ols, max_K=8)
    assert res.K_chosen >= 1, (
        f"GLMdenoise picked K=0 on real fMRI — either the data has no "
        f"structured noise (unlikely) or our CV is broken. CV curve: {res.cv_curve.tolist()}"
    )


def test_glmdenoise_noise_pool_is_nontrivial_fraction_on_real_data(
        sub005_ses01_runs
):
    Ys = [r[0] for r in sub005_ses01_runs]
    Xs = [r[1] for r in sub005_ses01_runs]
    res = glmdenoise_fit(Ys, Xs, fit_fn=_ols, max_K=5)
    V = Ys[0].shape[0]
    pool_frac = res.noise_pool_mask.sum() / V
    assert 0.05 < pool_frac < 0.55, (
        f"noise pool fraction {pool_frac:.2%} outside reasonable range"
    )


# ----------------------- fracridge on real data -----------------------------

def test_fracridge_solve_at_frac_one_recovers_ols_on_real_data(
        sub005_ses01_runs
):
    """For a randomly-sampled real-data voxel, fracridge at f=1 should match
    OLS within tolerance."""
    Y0, X0, _, _, _ = sub005_ses01_runs[0]
    rng = np.random.default_rng(0)
    v = int(rng.integers(0, Y0.shape[0]))
    y = Y0[v]
    beta_ols = np.linalg.lstsq(X0, y, rcond=None)[0]
    betas, lambdas = fracridge_solve(X0, y, np.array([1.0]))
    np.testing.assert_allclose(betas[0], beta_ols, atol=1e-2)
    assert lambdas[0] < 1e-2


def test_fracridge_cv_runs_on_real_data(sub005_ses01_runs):
    """Smoke: fracridge_cv runs without error on real fMRI, returns valid
    fractions and CV scores. We do NOT assert a specific shrinkage pattern —
    that depends on design conditioning, which differs between our LSS
    (1 probe regressor) and GLMsingle's full-trial design where fracridge
    is decisive. Math correctness is pinned by synthetic tests."""
    if len(sub005_ses01_runs) < 2:
        pytest.skip("need ≥ 2 runs")
    Y_train, X_train, _, _, _ = sub005_ses01_runs[0]
    Y_test, X_test, _, _, _ = sub005_ses01_runs[1]
    rng = np.random.default_rng(0)
    n_sample = 50
    idx = rng.choice(Y_train.shape[0], size=n_sample, replace=False)
    f_chosen, cv_score = fracridge_cv(
        Y_train=[Y_train[idx]], X_train=[X_train],
        Y_test=[Y_test[idx]], X_test=[X_test],
        fracs=np.linspace(0.1, 1.0, 10),
    )
    assert f_chosen.shape == (n_sample,)
    assert cv_score.shape == (n_sample,)
    assert (f_chosen > 0).all() and (f_chosen <= 1).all()
    assert not np.isnan(cv_score).any()
