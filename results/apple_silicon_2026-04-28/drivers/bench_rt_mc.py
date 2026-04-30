#!/usr/bin/env python3
"""Benchmark RT motion correction options for the TR=1.5s budget.

Compares per-volume runtime of:
  1. FSL MCFLIRT (single-volume, stages=4, cost=normcorr) — file I/O included
  2. jaxoccoli RigidBodyRegistration (Adam, 6-DOF)        — JAX MPS
  3. jaxoccoli GaussNewtonRegistration (6-DOF, ~10 iter)  — JAX MPS
  4. jaxoccoli motion_phase translation only              — JAX MPS

Reports wall-clock latency per volume and whether within TR=1.5s budget.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore")
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")

# Use one ses-03 raw BOLD volume as the test target
RAW_BIDS = LOCAL / "rt3t" / "data" / "raw_bids" / "sub-005" / "ses-03" / "func"
RUN1_BOLD = RAW_BIDS / "sub-005_ses-03_task-C_run-01_bold.nii.gz"
BOLDREF = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_run-01_space-T1w_boldref.nii.gz"

print(f"loading run-01 BOLD as test source...")
img = nib.load(RUN1_BOLD)
print(f"  shape: {img.shape}, dtype: {img.get_fdata().dtype}")

# Test target: take TR 50 (mid-run), test source: take TR 0 (reference)
n_trs_total = img.shape[3]
ref_vol = img.slicer[..., 0:1].get_fdata().astype(np.float32).squeeze()  # (X,Y,Z)
test_vol = img.slicer[..., 50:51].get_fdata().astype(np.float32).squeeze()
print(f"  ref vol shape: {ref_vol.shape}, test vol shape: {test_vol.shape}")
print(f"  ref-test L2 dist: {np.linalg.norm(ref_vol - test_vol):.0f}")
print(f"  jax backend: {jax.default_backend()}, devices: {jax.devices()}")
print()


# ============================================================================
# 1. FSL MCFLIRT — per-volume, file I/O included
# ============================================================================

def bench_mcflirt(test_vol: np.ndarray, ref_vol: np.ndarray, affine, n_trials: int = 5):
    print(f"=== 1. FSL MCFLIRT (single-volume, stages=4, cost=normcorr) ===")
    times = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Write reference once
        ref_path = td / "ref.nii.gz"
        nib.save(nib.Nifti1Image(ref_vol[..., None], affine), str(ref_path))

        for i in range(n_trials):
            # Write test volume
            test_path = td / f"test_{i}.nii.gz"
            out_path = td / f"out_{i}"
            nib.save(nib.Nifti1Image(test_vol[..., None], affine), str(test_path))

            t0 = time.time()
            result = subprocess.run([
                "/Users/mhough/fsl/bin/mcflirt",
                "-in", str(test_path),
                "-out", str(out_path),
                "-reffile", str(ref_path),
                "-stages", "4",
                "-cost", "normcorr",
                "-plots",
            ], capture_output=True, text=True, timeout=10)
            wall = time.time() - t0
            times.append(wall)
            if result.returncode != 0:
                print(f"  trial {i}: FAILED — {result.stderr[:200]}")
                return None

    times = np.asarray(times)
    print(f"  per-volume latency: {times.mean()*1000:.1f}ms ± {times.std()*1000:.1f}ms "
          f"(min {times.min()*1000:.1f}, max {times.max()*1000:.1f})")
    return times


# ============================================================================
# 2. jaxoccoli RigidBodyRegistration (Adam, 6-DOF)
# ============================================================================

def bench_rigid_adam(test_vol: np.ndarray, ref_vol: np.ndarray, n_trials: int = 5):
    sys.path.insert(0, str(REPO))
    from jaxoccoli.motion import RigidBodyRegistration

    print(f"=== 2. jaxoccoli RigidBodyRegistration (Adam, 6-DOF, 50 iter) ===")
    ref_jax = jnp.asarray(ref_vol)
    test_jax = jnp.asarray(test_vol)
    reg = RigidBodyRegistration(template=ref_jax, vol_shape=ref_vol.shape, n_iter=50)

    # Warm-up JIT
    _ = reg.register_volume(test_jax)
    jax.block_until_ready(_[0])

    times = []
    for _ in range(n_trials):
        t0 = time.time()
        params, registered = reg.register_volume(test_jax)
        jax.block_until_ready(params)
        times.append(time.time() - t0)
    times = np.asarray(times)
    print(f"  per-volume latency: {times.mean()*1000:.1f}ms ± {times.std()*1000:.1f}ms "
          f"(min {times.min()*1000:.1f}, max {times.max()*1000:.1f})")
    return times


# ============================================================================
# 3. jaxoccoli GaussNewtonRegistration (6-DOF, ~10 iter)
# ============================================================================

def bench_rigid_gn(test_vol: np.ndarray, ref_vol: np.ndarray, n_trials: int = 5):
    sys.path.insert(0, str(REPO))
    from jaxoccoli.motion import GaussNewtonRegistration

    print(f"=== 3. jaxoccoli GaussNewtonRegistration (6-DOF, ~10 iter) ===")
    ref_jax = jnp.asarray(ref_vol)
    test_jax = jnp.asarray(test_vol)
    reg = GaussNewtonRegistration(template=ref_jax, vol_shape=ref_vol.shape, n_iter=10)

    _ = reg.register_volume(test_jax)
    jax.block_until_ready(_[0])

    times = []
    for _ in range(n_trials):
        t0 = time.time()
        params, registered = reg.register_volume(test_jax)
        jax.block_until_ready(params)
        times.append(time.time() - t0)
    times = np.asarray(times)
    print(f"  per-volume latency: {times.mean()*1000:.1f}ms ± {times.std()*1000:.1f}ms "
          f"(min {times.min()*1000:.1f}, max {times.max()*1000:.1f})")
    return times


# ============================================================================
# 4. jaxoccoli motion_phase (phase-correlation, translation only, 3-DOF)
# ============================================================================

def bench_phase_corr(test_vol: np.ndarray, ref_vol: np.ndarray, n_trials: int = 10):
    sys.path.insert(0, str(REPO))
    from jaxoccoli.motion_phase import register_translation

    print(f"=== 4. jaxoccoli motion_phase (FFT phase-corr, translation only) ===")
    ref_jax = jnp.asarray(ref_vol)
    test_jax = jnp.asarray(test_vol)

    _ = register_translation(ref_jax, test_jax)
    jax.block_until_ready(_)

    times = []
    for _ in range(n_trials):
        t0 = time.time()
        result = register_translation(ref_jax, test_jax)
        jax.block_until_ready(result)
        times.append(time.time() - t0)
    times = np.asarray(times)
    print(f"  per-volume latency: {times.mean()*1000:.1f}ms ± {times.std()*1000:.1f}ms "
          f"(min {times.min()*1000:.1f}, max {times.max()*1000:.1f})")
    return times


# ============================================================================
# Run all
# ============================================================================

if __name__ == "__main__":
    affine = img.affine

    results = {}
    try:
        results["MCFLIRT"] = bench_mcflirt(test_vol, ref_vol, affine, n_trials=3)
    except Exception as e:
        print(f"  MCFLIRT failed: {e}")
    print()
    try:
        results["RigidBody (Adam)"] = bench_rigid_adam(test_vol, ref_vol, n_trials=5)
    except Exception as e:
        print(f"  RigidBody Adam failed: {e}")
    print()
    try:
        results["GaussNewton"] = bench_rigid_gn(test_vol, ref_vol, n_trials=5)
    except Exception as e:
        print(f"  GaussNewton failed: {e}")
    print()
    try:
        results["PhaseCorr (translation only)"] = bench_phase_corr(test_vol, ref_vol, n_trials=10)
    except Exception as e:
        print(f"  PhaseCorr failed: {e}")

    print()
    print("=== summary (per-volume latency, TR budget = 1500ms) ===")
    print(f"{'method':40s} {'mean (ms)':>10s} {'within TR?':>12s}")
    print("-" * 65)
    for name, times in results.items():
        if times is None: continue
        mean_ms = float(times.mean()) * 1000
        within = "✓" if mean_ms < 1500 else "✗ over"
        print(f"{name:40s} {mean_ms:>10.1f} {within:>12s}")
