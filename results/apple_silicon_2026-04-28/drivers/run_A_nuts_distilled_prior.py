#!/usr/bin/env python3
"""(A) NUTS-distilled hierarchical prior for Variant G.

Hierarchical Bayesian model on ses-01 G_fmriprep training betas:
  For each voxel v, training trials t:
    β_train[v, t] ~ N(μ_v, σ_v²)
  μ_v ~ N(μ_pop, τ²)              # voxel-level prior shrinks toward pop mean
  log σ_v ~ N(log σ_pop, ω²)
  μ_pop ~ N(0, 100²)
  log τ, log σ_pop, log ω ~ N(0, 4²)

Run NUTS (blackjax) to get posterior over (μ_v, σ_v²). Use the posterior mean
of μ_v and posterior variance of μ_v as the prior_mean / prior_var for ses-03
Variant G fits — this is the shrinkage-improved version of cell 5 (which used
plain empirical mean/var without shrinkage and got -3pp).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import blackjax

warnings.filterwarnings("ignore")
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
P.RT3T = LOCAL / "rt3t" / "data"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.HRF_INDICES_PATH = str(LOCAL / "rt3t" / "data" / "avg_hrfs_s1_s2_full.npy")
P.HRF_LIB_PATH = str(LOCAL / "rt3t" / "data" / "getcanonicalhrflibrary.tsv")
P.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

CELL = "VariantG_NUTSprior_glover_rtm"
SESSION = "ses-03"
RUNS = list(range(1, 12))
N_WARMUP = 500
N_SAMPLES = 1000


def build_logpdf(beta_train: jnp.ndarray):
    """Hierarchical Gaussian. Per-voxel mean + variance, hyperprior over voxels."""
    T, V = beta_train.shape

    def logpdf(params):
        # Unpack: 4 hyperparams + 2*V voxel params
        mu_pop = params[0]
        log_tau = params[1]
        log_sigma_pop = params[2]
        log_omega = params[3]
        mu_v = params[4:4 + V]
        log_sigma_v = params[4 + V:]

        tau = jnp.exp(log_tau)
        sigma_pop = jnp.exp(log_sigma_pop)
        omega = jnp.exp(log_omega)
        sigma_v = jnp.exp(log_sigma_v)

        # Hyperpriors (broad)
        lp = (-0.5 * (mu_pop / 100.0) ** 2
              - 0.5 * (log_tau / 4.0) ** 2
              - 0.5 * (log_sigma_pop / 4.0) ** 2
              - 0.5 * (log_omega / 4.0) ** 2)
        # Voxel-level priors
        lp += jnp.sum(-0.5 * ((mu_v - mu_pop) / tau) ** 2 - log_tau)
        lp += jnp.sum(-0.5 * ((log_sigma_v - log_sigma_pop) / omega) ** 2 - log_omega)
        # Likelihood: β_train[t, v] ~ N(μ_v, σ_v²)
        # Vectorize over trials and voxels
        lp += jnp.sum(-0.5 * ((beta_train - mu_v[None, :]) / sigma_v[None, :]) ** 2
                       - log_sigma_v[None, :])
        return lp
    return logpdf


def main():
    print("=== (A) NUTS-distilled hierarchical prior ===")

    # 1. Load ses-01 training betas (G_fmriprep)
    train_path = LOCAL / "task_2_1_betas" / "G_fmriprep_ses-01_betas.npy"
    beta_train = np.load(train_path).astype(np.float32)              # (770, 2792)
    print(f"  training betas: {beta_train.shape}")

    # 2. Subsample voxels for speed — fit hyperprior on a representative sample
    # then apply to all 2792. Or just use all if fast enough.
    V = beta_train.shape[1]
    print(f"  fitting NUTS on V={V} voxels, T={beta_train.shape[0]} training trials")

    beta_train_j = jnp.asarray(beta_train)
    logpdf = build_logpdf(beta_train_j)
    n_params = 4 + 2 * V
    init_pos = jnp.concatenate([
        jnp.array([0.0, 0.0, 0.0, 0.0]),
        beta_train.mean(axis=0),                                     # init μ_v at empirical mean
        jnp.log(beta_train.std(axis=0) + 1.0),                       # init log σ_v
    ])
    print(f"  n_params: {n_params}")

    # 3. Window adaptation + NUTS sampling
    rng_key = jax.random.PRNGKey(0)
    warmup = blackjax.window_adaptation(blackjax.nuts, logpdf, target_acceptance_rate=0.65)
    print(f"  warmup ({N_WARMUP} steps)...")
    t0 = time.time()
    (state, parameters), _ = warmup.run(rng_key, init_pos, num_steps=N_WARMUP)
    print(f"  warmup done in {time.time()-t0:.1f}s")

    # Sampling
    nuts = blackjax.nuts(logpdf, **parameters)

    @jax.jit
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info.is_divergent)

    keys = jax.random.split(rng_key, N_SAMPLES)
    print(f"  sampling ({N_SAMPLES} steps)...")
    t0 = time.time()
    final_state, (positions, divergent) = jax.lax.scan(one_step, state, keys)
    print(f"  sampling done in {time.time()-t0:.1f}s, divergent: {int(divergent.sum())}/{N_SAMPLES}")

    # 4. Extract per-voxel posterior moments
    samples = np.asarray(positions)                                   # (N_SAMPLES, n_params)
    mu_v_samples = samples[:, 4:4 + V]                                # (N_SAMPLES, V)
    log_sigma_v_samples = samples[:, 4 + V:]
    mu_v_post_mean = mu_v_samples.mean(axis=0).astype(np.float32)
    mu_v_post_var = mu_v_samples.var(axis=0).astype(np.float32)
    print(f"  posterior μ_v: mean={mu_v_post_mean.mean():.4f}, "
          f"var={mu_v_post_var.mean():.4f}")
    print(f"  shrinkage: empirical-mean std={beta_train.mean(axis=0).std():.4f}, "
          f"posterior-mean std={mu_v_post_mean.std():.4f} "
          f"(ratio = {mu_v_post_mean.std()/beta_train.mean(axis=0).std():.3f})")

    # Save the distilled prior
    np.save(P.OUT_DIR / f"NUTS_prior_mean_{SESSION}.npy", mu_v_post_mean)
    np.save(P.OUT_DIR / f"NUTS_prior_var_{SESSION}.npy", mu_v_post_var + 1.0)  # add reg

    # 5. Run Variant G on ses-03 with this NUTS-distilled prior
    print(f"\n  applying NUTS prior to ses-03 Variant G...")
    sys.argv = [
        "run_A",
        "--cells", "VariantG_glover_rtm_prior",
        "--session", "ses-03",
        "--runs", *[str(r) for r in RUNS],
        "--prior-from-session", "ses-01",
    ]
    # Monkey-patch the prior loading: instead of empirical mean/var, use NUTS posterior
    import prereg_variant_sweep as Pmod
    _orig_main = Pmod.main

    def _patched_main():
        # Pre-load NUTS prior into the variables main() uses
        prior_mean = mu_v_post_mean
        prior_var = np.maximum(mu_v_post_var + 1.0, 1e-3)
        # Run cell directly
        cell_name = CELL
        config = Pmod.CELLS["VariantG_glover_rtm_prior"].copy()
        config["session"] = SESSION
        config["runs"] = RUNS
        config["prior_source"] = f"NUTS posterior over ses-01 G_fmriprep"
        config["NUTS_warmup"] = N_WARMUP
        config["NUTS_samples"] = N_SAMPLES
        print(f"\n  === {cell_name} === {config}")
        t0 = time.time()
        betas, trial_ids = Pmod.run_glm_cell(
            cell_name, mode=config["mode"],
            bold_source=config["bold_source"],
            hrf_strategy=config["hrf_strategy"],
            session=SESSION, runs=RUNS,
            prior_mean=prior_mean, prior_var=prior_var,
            denoise=None,
        )
        Pmod.save_cell(cell_name, betas, trial_ids, SESSION, config)
        print(f"  elapsed: {time.time() - t0:.1f}s")

    _patched_main()


if __name__ == "__main__":
    main()
