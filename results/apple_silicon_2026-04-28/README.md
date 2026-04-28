# Task 2.1 12-cell factorial — Apple Silicon (M-series) results, 2026-04-28

Local replication of `TASK_2_1_PREREGISTRATION.md` 12-cell sweep on a Mac, intended as a reference for the DGX agent's parallel run on the same factorial design. Where the two diverge tells us whether the divergence is **algorithmic** (will reproduce identically on either platform) or **platform/numerical** (jax-mps vs CUDA).

## Hardware + environment

- Apple Silicon (M5 Max), 36 GB RAM, 18 CPU cores, native arm64 throughout
- See `env_jax.txt`, `env_torch.txt`, `env_macos.txt`
- JAX path: **jax 0.9.2 + jax-mps 0.9.13 (MLX-backed)** on Python 3.13. `jax-metal` was abandoned; see project memory `reference_jax_apple_silicon.md`.
- PyTorch path: **torch 2.10.0 + MPS** on Python 3.12.
- FSL: native arm64 install at `~/fsl`, MCFLIRT parallelized via `fsl_sub` shell method.

## Drivers (in `drivers/`)

These are local-path-rebound thin wrappers over `scripts/*.py` in the repo. Each rebinds `PAPER_ROOT` etc. to `~/Workspace/data/rtmindeye_paper/` (laptop layout; DGX layout is `/data/derivatives/rtmindeye_paper/`) without modifying any tracked code.

| Driver | What it ran |
|---|---|
| `mcflirt_ses03_local.sh` + `mcflirt_one_run.sh` | 11 parallel MCFLIRT jobs via `fsl_sub` for ses-03 |
| `mcflirt_par_only.sh` | re-ran MCFLIRT just to capture `.par` motion params (a `cp` typo in v1 missed them) |
| `run_jax_cells_local.py` | cells 1, 2, 4, 6, 7, 8, 9 (JAX-only) |
| `run_g_fmriprep_ses01.py` | ses-01 G_fmriprep prior (cell-5 prerequisite) |
| `run_cell_5_local.py` | cell 5 (VariantG with ses-01 prior) |
| `run_cell_3_local.py` | cell 3 (nilearn AR1 sanity, drift_model=None, high_pass=0.0) |
| `run_cells_10_11_local.py` | cells 10 & 11 (nilearn paper RT replicas) |
| `run_cell_12_local.py` | cell 12 (paper Offline replica) |
| `run_retrieval_local.py` | v1 retrieval eval (with the **leaky** session-level z-score) |
| `run_retrieval_local_v2.py` | **v2 retrieval eval** with causal cumulative z-score |
| `parity_jax_nilearn_v2.py` | AR(1) parity test, JAX vs nilearn with same design matrix |
| `variant_g_pw.py` | Prais-Winsten + iterated AR(1) sidecar (parity attempt) |

## Results

### Cells 1-10: 150-trial denominator (no repeat-averaging)
Each special515 image shown 3× during ses-03 → 50 unique × 3 = 150 raw trials.

### Cells 11-12: 50-trial denominator (post repeat-averaging, paper §2.5)
3 repeats per image collapsed into 1 averaged β → 50 trials.

### Final retrieval (`retrieval_results_v2_cumulative_zscore.json`)

| # | Cell | Trials | Top-1 | Top-5 |
|---|---|---|---|---|
| 1 | `OLS_glover_rtm` | 150 | 56.7% | 80.0% |
| 2 | `AR1freq_glover_rtm` | 150 | **62.7%** | 84.0% |
| 3 | `AR1freq_glover_rtm_nilearn` | 150 | 56.0% | 84.0% |
| 4 | `VariantG_glover_rtm` | 150 | 60.0% | 83.3% |
| 5 | `VariantG_glover_rtm_prior` | 150 | 62.0% | 82.7% |
| 6 | `AR1freq_glmsingleS1_rtm` | 150 | **44.7%** | 74.7% |
| 7 | `AR1freq + GLMdenoise + fracridge` | 150 | 60.0% | 84.7% |
| 8 | `VariantG + GLMdenoise + fracridge` | 150 | 60.0% | 84.7% |
| 9 | `VariantG + aCompCor` | 150 | 59.3% | 87.3% |
| 10 | `RT_paper_replica_partial` | 150 | 56.7% | 82.7% |
| 11 | `RT_paper_replica_full` | 50 | **74.0%** | 92.0% |
| 12 | `Offline_paper_replica_full` | 50 | **76.0%** | 94.0% |

### Bug fix: cumulative z-score
Compare `v1_leaky_zscore.json` to `v2_cumulative_zscore.json`. v1 used session-level z-score over **all 770 trials including the test trials themselves** — data leakage. v2 uses causal cumulative z-score (trial *i* uses statistics from trials 0..*i*−1 only) for cells 1-10, and **skips** the re-z for cells 11-12 (already cum-z'd inside their cell driver, per `cumulative_zscore_with_optional_repeat_avg` in `scripts/rt_paper_full_replica.py`).

The fix moved cell 3 (nilearn AR1) down 4pp, cell 12 (Offline) down 4pp from 80 → **76% (paper exact)**, cell 11 down 2pp.

## Anchor calibration

- **Cell 12 = 76.0% — exactly the paper's Offline number.** Strong calibration evidence.
- **Cell 11 = 74.0% vs paper's RT 66%** — 8pp inflation.
- **Local RT-vs-Offline gap = 2pp** (74→76), not paper's 10pp. The factorial decomposition can't yet read off "fmriprep contribution" + "GLMsingle contribution" because the gap to decompose isn't there.

The most likely reason is the **checkpoint**. We used the only finalmask checkpoint available locally:
`sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth`
The TASK_2_1_STATUS.md canonical is `sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_avgrepeats_finalmask_epochs_150.pth` from `macandro96/mindeye_offline_ckpts` on HF (gated, requires auth). DGX should run with the canonical checkpoint and report whether cell 11 lands at ~66% there.

## Open issues for DGX comparison

### 1. AR(1) parity gap (cell 2 vs cell 3)
JAX `_variant_g_forward(pp_scalar=0, rho_prior_var=1e8)` does **not** match `nilearn FirstLevelModel(noise_model='ar1')` to the prereg's 1e-3 tolerance. Best on Mac:
- Same design matrix passed to both → r=0.953, scale ratio 0.84 (Cochrane-Orcutt single-pass)
- After Prais-Winsten + 2 iterations (`drivers/variant_g_pw.py`) → r=0.995, scale ratio 1.02, max diff 9.5
- 1e-3 still not achievable

See `parity_test_output.txt` for the full numerical readout.

**DGX question**: does this same r=0.995 / 2% scale gap hold on CUDA? If yes → algorithmic difference between JAX and nilearn whitening, fix `_variant_g_forward` or amend prereg tolerance. If DGX shows materially better parity (r>0.999) → jax-mps numerical issue, file separately.

The Prais-Winsten variant is at `drivers/variant_g_pw.py` ready to drop into `_variant_g_forward` if DGX confirms.

### 2. Cell 6 collapse (per-voxel HRF library) — 44.7%
The HRF index field (`avg_hrfs_s1_s2_full.npy`) was computed against the paper's specific MC pipeline. Our local MCFLIRT is recipe-faithful to `scripts/mcflirt_ses03.sbatch` but uses a different `boldref` reference frame possibly. Or HRF library indexing into the convolution is mis-scaled.

**DGX question**: does cell 6 also collapse on the canonical paper pipeline? If yes → HRF library / index alignment is broken in the prereg sweep code itself. If no → it's a Mac-local MCFLIRT difference.

### 3. Cell 11 8pp inflation
Most likely the checkpoint, secondarily the MCFLIRT details.

**DGX question**: with the canonical `sample=10_..._epochs_150` checkpoint, does cell 11 land at ~66%? If yes → checkpoint was the only issue, our v2 z-score fix calibrates Mac to paper. If no → there's a deeper pipeline difference (MCFLIRT-fmriprep subspace residual, repeat-avg semantics, etc.).

## How to compare

For each cell, expect:
- top-1 image retrieval within ±2pp Mac vs DGX → **algorithmic / platform-noise floor**
- larger differences → cells where the bug or platform divergence matters

The bigger DGX-vs-paper anchor question is whether DGX cell 12 lands at 76% and cell 11 lands at 66% with the canonical checkpoint.
