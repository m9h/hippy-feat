"""Benchmarking script for jaxoccoli preprocessing variants.

Compares:
  1. Batch Bayesian Variant G (realtime.py)
  2. Streaming Kalman Filter (white noise) (streaming_kalman.py)
  3. Streaming Kalman Filter (AR1 prewhitened) (streaming_kalman.py)
  4. Baseline (Simple average of stim TRs)

Metrics:
  - Category separation in RDM (Within < Between)
  - Voxelwise SNR (|beta|/sqrt(var))
  - Latency per TR (processing time)
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List

from smoke_test_nsd_tribe import (
    NSD_TR, NSD_TRS_PER_TRIAL, NSD_CATEGORIES,
    make_nsd_trial_sequence, NSDTrialProducer, compute_rdm
)
from jaxoccoli.realtime import RTPipeline, RTPipelineConfig
from jaxoccoli.streaming_kalman import (
    init_streaming_kalman, streaming_kalman_update,
    init_streaming_kalman_ar1, streaming_kalman_ar1_update
)
from jaxoccoli.hf_encoder import make_cortical_projection
from jaxoccoli.covariance import corr

import queue

def generate_hard_mode_bold(n_trs, n_vertices, key, rho=0.6, drift_scale=0.5, spike_prob=0.01):
    """Generate representative BOLD with AR(1) noise, drift, and spikes."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # 1. AR(1) Noise: e[t] = rho * e[t-1] + white[t]
    white = jax.random.normal(k1, (n_trs, n_vertices))
    
    def ar_step(carry, w):
        next_val = rho * carry + w
        return next_val, next_val
    
    _, ar_noise = jax.lax.scan(ar_step, jnp.zeros(n_vertices), white)
    
    # 2. Low-frequency drift (sine + cosine)
    t = jnp.arange(n_trs)[:, None]
    drift = drift_scale * jnp.sin(2 * jnp.pi * t / n_trs) + drift_scale * jnp.cos(4 * jnp.pi * t / n_trs)
    
    # 3. Spikes
    spikes = jax.random.bernoulli(k2, p=spike_prob, shape=(n_trs, n_vertices)) * 5.0
    
    return ar_noise + drift + spikes

def run_bakeoff(n_trials=20, n_parcels=50, seed=42, hard_mode=True):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    n_vertices = 20484
    trials = make_nsd_trial_sequence(n_trials=n_trials, seed=seed)
    
    # 1. Setup projection
    proj_params, project_fn = make_cortical_projection(
        n_vertices=n_vertices, n_parcels=n_parcels, key=k1, init="block"
    )
    
    # 2. Custom Producer Logic (replacing NSDTrialProducer to inject hard mode noise)
    # Build category spatial templates
    templates = []
    for i in range(len(NSD_CATEGORIES)):
        k, k_sub = jax.random.split(key)
        # Each category activates a specific vertex range
        v_start = i * (n_vertices // len(NSD_CATEGORIES))
        v_end = (i+1) * (n_vertices // len(NSD_CATEGORIES))
        pattern = jnp.zeros(n_vertices).at[v_start:v_end].set(1.0)
        templates.append(pattern)
    
    n_trs_per_trial = NSD_TRS_PER_TRIAL
    n_total_trs = n_trials * n_trs_per_trial
    
    # Clean signal: (T, V)
    clean_signal = jnp.zeros((n_total_trs, n_vertices))
    categories = [t["category"] for t in trials]
    
    is_stimulus = np.zeros(n_total_trs, dtype=bool)
    trial_ids = np.zeros(n_total_trs, dtype=int)
    
    for i, t in enumerate(trials):
        start = i * n_trs_per_trial
        # 2 TRs of stimulus
        clean_signal = clean_signal.at[start:start+2].set(templates[NSD_CATEGORIES.index(t["category"])])
        is_stimulus[start:start+2] = True
        trial_ids[start:start+n_trs_per_trial] = t["trial_id"]
        
    # Inject Hard Mode Noise
    if hard_mode:
        print(f"Generating Hard Mode BOLD (rho=0.6, drift, spikes)...")
        noise = generate_hard_mode_bold(n_total_trs, n_vertices, k2)
        bold = clean_signal + noise
    else:
        bold = clean_signal + jax.random.normal(k2, (n_total_trs, n_vertices)) * 0.5

    all_frames = []
    for t in range(n_total_trs):
        all_frames.append({
            "bold": np.array(bold[t]),
            "tr_index": t,
            "trial_id": int(trial_ids[t]),
            "is_stimulus": bool(is_stimulus[t])
        })
    
    results = {}
    
    # -----------------------------------------------------------------------
    # Variant: Baseline (Mean)
    # -----------------------------------------------------------------------
    print("\nRunning Baseline (Mean)...")
    t0 = time.time()
    trial_data = {tid: [] for tid in range(n_trials)}
    for f in all_frames:
        if f["is_stimulus"]:
            parc = project_fn(proj_params, jnp.asarray(f["bold"])[None, :])[0]
            trial_data[f["trial_id"]].append(parc)
    
    betas_baseline = jnp.stack([jnp.mean(jnp.stack(trial_data[tid]), axis=0) for tid in range(n_trials)])
    results["Baseline"] = {
        "betas": betas_baseline,
        "time": time.time() - t0,
        "snr": jnp.mean(jnp.abs(betas_baseline) / (jnp.std(betas_baseline, axis=0) + 1e-6))
    }

    # -----------------------------------------------------------------------
    # Variant: Batch Variant G (realtime.py)
    # -----------------------------------------------------------------------
    print("Running Batch Variant G...")
    config = RTPipelineConfig(
        tr=NSD_TR,
        mask=np.ones(n_parcels, dtype=bool),
        onsets_sec=np.array([t["onset_tr"] * NSD_TR for t in trials]),
        max_trs=len(all_frames) + 10
    )
    pipeline = RTPipeline(config)
    pipeline.precompute()
    
    betas_vg = np.zeros((n_trials, n_parcels))
    vars_vg = np.zeros((n_trials, n_parcels))
    
    t0 = time.time()
    for f in all_frames:
        parc = project_fn(proj_params, jnp.asarray(f["bold"])[None, :])[0]
        # Note: RTPipeline expects 3D vol, but we can pass 1D parcel vector if we cheat the mask
        res = pipeline.on_volume(np.asarray(parc), f["tr_index"])
        if res is not None:
            betas_vg[res["probe_trial"]] = res["beta_mean"]
            vars_vg[res["probe_trial"]] = res["beta_var"]
    
    results["Variant G"] = {
        "betas": jnp.asarray(betas_vg),
        "time": time.time() - t0,
        "snr": jnp.mean(jnp.abs(betas_vg) / jnp.sqrt(vars_vg + 1e-10))
    }

    # -----------------------------------------------------------------------
    # Variant: Streaming Kalman (White)
    # -----------------------------------------------------------------------
    print("Running Streaming Kalman (White)...")
    # We need a design matrix row for each TR
    # For simplicity, we'll use a simple LSS-like design: 1 for probe trial stimulus, 0 otherwise
    # In a real bakeoff, we'd use the same DM as Variant G
    
    # Let's build a simple DM: (n_trs, n_trials + 1 for intercept)
    n_trs = len(all_frames)
    dm = np.zeros((n_trs, n_trials + 1))
    dm[:, -1] = 1.0 # intercept
    for f in all_frames:
        if f["is_stimulus"]:
            dm[f["tr_index"], f["trial_id"]] = 1.0
    
    state = init_streaming_kalman(n_trials + 1, n_parcels)
    
    t0 = time.time()
    for f in all_frames:
        parc = project_fn(proj_params, jnp.asarray(f["bold"])[None, :])[0]
        state = streaming_kalman_update(state, jnp.asarray(dm[f["tr_index"]]), jnp.asarray(parc))
    
    # Extract betas (diagonal of the beta_mean matrix for each trial's column)
    betas_kalman = jnp.array([state.beta_mean[:, i] for i in range(n_trials)])
    # Variance = b / (a - 1) * diag(cov)
    sigma2 = state.b_post / (state.a_post - 1.0)
    vars_kalman = jnp.array([sigma2 * state.beta_cov[:, i, i] for i in range(n_trials)])

    results["Kalman (White)"] = {
        "betas": betas_kalman,
        "time": time.time() - t0,
        "snr": jnp.mean(jnp.abs(betas_kalman) / jnp.sqrt(vars_kalman + 1e-10))
    }

    # -----------------------------------------------------------------------
    # Variant: Streaming Kalman (AR1)
    # -----------------------------------------------------------------------
    print("Running Streaming Kalman (AR1)...")
    state_ar1 = init_streaming_kalman_ar1(n_trials + 1, n_parcels)
    
    t0 = time.time()
    for f in all_frames:
        parc = project_fn(proj_params, jnp.asarray(f["bold"])[None, :])[0]
        state_ar1 = streaming_kalman_ar1_update(state_ar1, jnp.asarray(dm[f["tr_index"]]), jnp.asarray(parc))
    
    betas_ar1 = jnp.array([state_ar1.beta_mean[:, i] for i in range(n_trials)])
    sigma2_ar1 = state_ar1.b_post / (state_ar1.a_post - 1.0)
    vars_ar1 = jnp.array([sigma2_ar1 * state_ar1.beta_cov[:, i, i] for i in range(n_trials)])

    results["Kalman (AR1)"] = {
        "betas": betas_ar1,
        "time": time.time() - t0,
        "snr": jnp.mean(jnp.abs(betas_ar1) / jnp.sqrt(vars_ar1 + 1e-10))
    }

    # -----------------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Variant':<20} | {'Category Sep':<15} | {'Mean SNR':<10} | {'Latency/TR':<12}")
    print("-" * 80)
    
    categories = [t["category"] for t in trials]
    
    for name, res in results.items():
        rdm = compute_rdm(res["betas"])
        
        # Calculate Category Separation (Between - Within)
        within = []
        between = []
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                if categories[i] == categories[j]:
                    within.append(float(rdm[i, j]))
                else:
                    between.append(float(rdm[i, j]))
        
        sep = np.mean(between) - np.mean(within)
        lat = (res["time"] / n_trs) * 1000 # ms
        
        print(f"{name:<20} | {sep:15.4f} | {float(res['snr']):10.3f} | {lat:10.2f} ms")
    print("="*80)

if __name__ == "__main__":
    run_bakeoff()
