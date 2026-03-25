"""
Pytest fixtures for the RT GLM variants test suite.

Provides dummy data, masks, configs, and shared utilities for testing
all preprocessing variants.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path so we can import rt_glm_variants
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rt_glm_variants import VariantConfig

# ---------------------------------------------------------------------------
# Constants matching real data dimensions
# ---------------------------------------------------------------------------
VOL_SHAPE = (76, 90, 74)
N_BRAIN_VOXELS = 19174
N_UNION_VOXELS = 8627
TR = 1.5
MAX_TRS = 192


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


@pytest.fixture(scope="session")
def vol_shape():
    return VOL_SHAPE


@pytest.fixture(scope="session")
def brain_mask_flat(rng):
    """
    Simulated brain mask: (506160,) bool with exactly 19174 True values.
    """
    n_total = np.prod(VOL_SHAPE)
    mask = np.zeros(n_total, dtype=bool)
    indices = rng.choice(n_total, size=N_BRAIN_VOXELS, replace=False)
    mask[indices] = True
    return mask


@pytest.fixture(scope="session")
def union_mask(rng):
    """
    Simulated union mask: (19174,) bool with exactly 8627 True values.
    """
    mask = np.zeros(N_BRAIN_VOXELS, dtype=bool)
    indices = rng.choice(N_BRAIN_VOXELS, size=N_UNION_VOXELS, replace=False)
    mask[indices] = True
    return mask


@pytest.fixture(scope="session")
def dummy_volume_3d(rng):
    """Single 3D volume (76, 90, 74) with realistic fMRI-like values."""
    return (rng.randn(*VOL_SHAPE) * 100 + 1000).astype(np.float32)


@pytest.fixture(scope="session")
def dummy_timeseries(rng):
    """
    Masked voxel timeseries: (8627, 50) simulating 50 TRs of data.
    Values are fMRI-like (mean ~1000, std ~100).
    """
    n_trs = 50
    data = rng.randn(N_UNION_VOXELS, n_trs).astype(np.float32) * 100 + 1000
    return data


@pytest.fixture(scope="session")
def dummy_events():
    """
    Simulated event onsets (in seconds) for 10 trials at ~6s intervals.
    """
    return np.array([6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0])


@pytest.fixture(scope="session")
def dummy_training_betas(rng):
    """
    Simulated training betas: (100, 8627) for computing priors.
    """
    return rng.randn(100, N_UNION_VOXELS).astype(np.float32)


@pytest.fixture(scope="session")
def test_config(tmp_path_factory):
    """VariantConfig pointing to temp directories for output."""
    tmp = tmp_path_factory.mktemp("variants")
    return VariantConfig(
        tr=TR,
        n_voxels=N_UNION_VOXELS,
        vol_shape=VOL_SHAPE,
        max_trs=MAX_TRS,
        output_base=str(tmp),
    )


@pytest.fixture(scope="session")
def real_config():
    """VariantConfig with real data paths (for integration tests that need real data)."""
    return VariantConfig()


@pytest.fixture(scope="session")
def real_data_available():
    """Check if real data files exist for integration tests."""
    paths = [
        "/data/3t/data/union_mask_from_ses-01-02.npy",
        "/data/3t/data/avg_hrfs_s1_s2_full.npy",
        "/data/3t/data/getcanonicalhrflibrary.tsv",
        "/data/3t/data/sub-005_final_mask.nii.gz",
        "/home/mhough/fsl/src/fsl-feat5/data/default_flobs.flobs/hrfbasisfns.txt",
    ]
    return all(Path(p).exists() for p in paths)


@pytest.fixture
def small_timeseries(rng):
    """Small timeseries for fast unit tests: (100, 20)."""
    return rng.randn(100, 20).astype(np.float32) * 50 + 500


@pytest.fixture
def small_config():
    """Config with reduced voxel count for fast tests."""
    return VariantConfig(
        n_voxels=100,
        max_trs=64,
    )
