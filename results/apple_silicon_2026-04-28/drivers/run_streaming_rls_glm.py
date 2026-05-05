#!/usr/bin/env python3
"""Streaming/incremental GLM (Ernest Lo's actual persistent-GLM proposal).

For each trial i, at decode_TR (pst-based for Fast/Slow, run-end for EoR):
  - Build design X_i = [HRF-conv trial regressors 1..i, drift basis,
                         motion 6-param, aCompCor K=7] over BOLD rows 0..decode_TR_i
  - Solve ridge-regularized OLS (λ=1e-3·tr(XᵀX)/n)
  - Extract β_i from the i-th trial column

This differs from per-trial LSS (which only fits trial i + lumped reference)
and from end-of-session persistent LSA (which fits all trials at run-end).
The streaming variant fits all trials seen so far at decode time, with
truncated BOLD — the actual real-time analog of growing-design GLM.

Runs 3 tiers: Fast (pst=5), Slow (pst=20), EoR (run-end).
Saves βs as prereg cells `RT_paper_RLS_{Fast,Slow,EoR}_*_inclz` for scoring
through fold-0.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.ndimage import binary_erosion
from nilearn.signal import clean
from nilearn.image import resample_to_img
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT3T = LOCAL / "rt3t/data"
EVENTS_DIR = RT3T / "events"
MC_DIR = LOCAL / "motion_corrected_resampled"
PREREG = LOCAL / "task_2_1_betas/prereg"
TR = 1.5
SESSION = "ses-03"
RUNS = list(range(1, 12))
PVE_THRESH = 0.5
K_ACOMP = 7

# Masks
final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
brain_img = nib.load(RT3T / "sub-005_final_mask.nii.gz")

print("=== building noise-pool mask (CSF∪WM eroded×1) ===", flush=True)
brain_3d = brain_img.get_fdata() > 0
csf = resample_to_img(nib.load(RT3T / "T1_brain_seg_pve_0.nii.gz"),
                      brain_img, interpolation="linear",
                      force_resample=True, copy_header=True).get_fdata()
wm = resample_to_img(nib.load(RT3T / "T1_brain_seg_pve_2.nii.gz"),
                     brain_img, interpolation="linear",
                     force_resample=True, copy_header=True).get_fdata()
csfwm_3d = ((csf > PVE_THRESH) | (wm > PVE_THRESH)) & brain_3d
csfwm_e1 = binary_erosion(csfwm_3d, iterations=1)
mask_noise = csfwm_e1.flatten()
print(f"  noise-pool voxels: {csfwm_e1.sum()}", flush=True)


# === Load BOLD per run, build per-run aCompCor PCs (HP-filtered) ===
print("\n=== loading BOLD + extracting aCompCor PCs ===", flush=True)
bold_per_run = []
acomp_per_run = []
mc_per_run = []
n_TRs_per_run = []
for run in RUNS:
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    frames_relmask = []
    frames_noise = []
    for v in vols:
        flat = nib.load(v).get_fdata().flatten().astype(np.float32)
        frames_relmask.append(flat[final_mask][relmask])
        frames_noise.append(flat[mask_noise])
    bold = np.stack(frames_relmask, axis=0)            # (T, 2792)
    noise_ts = np.stack(frames_noise, axis=0)          # (T, V_noise)
    bold_per_run.append(bold)
    n_TRs_per_run.append(bold.shape[0])

    # HP-filter noise pool, SVD, top-K7. Input to clean must be (T, V).
    # noise_ts is (T_run, V_pool); pass directly to clean.
    ts_c = clean(noise_ts, t_r=TR, high_pass=0.01,
                 detrend=False, standardize=False)            # (T, V_pool)
    # SVD on (V_pool, T) so Vt rows are temporal components
    _, _, Vt = np.linalg.svd(ts_c.T, full_matrices=False)
    acomp_per_run.append(Vt[:K_ACOMP].T.astype(np.float32))    # (T, K7)

    par = MC_DIR / f"{SESSION}_run-{run:02d}_motion.par"
    mc_per_run.append(np.loadtxt(par).astype(np.float32) if par.exists()
                      else np.zeros((bold.shape[0], 6), dtype=np.float32))
    print(f"  run-{run:02d}: BOLD ({bold.shape[0]}, 2792), aCompCor {acomp_per_run[-1].shape}, mc {mc_per_run[-1].shape}", flush=True)

n_TRs_total = sum(n_TRs_per_run)
run_start = [0]
for n in n_TRs_per_run[:-1]:
    run_start.append(run_start[-1] + n)
print(f"  total TRs: {n_TRs_total}, run starts: {run_start}", flush=True)


# === Concatenate BOLD across runs ===
bold_full = np.concatenate(bold_per_run, axis=0)       # (n_TRs_total, 2792)
print(f"  bold_full: {bold_full.shape}", flush=True)


# === Glover HRF (Nilearn convention, TR-spaced) ===
hrf = glover_hrf(TR, oversampling=1, time_length=32, onset=0)
hrf = hrf / np.max(hrf)
hrf_len = len(hrf)
print(f"  HRF: {hrf.shape}", flush=True)


# === Build trial design (HRF-convolved per-trial) + chronological event list ===
print("\n=== building trial design matrix ===", flush=True)
trials = []   # (run_idx, image_name, global_onset_TR)
for run_idx, run in enumerate(RUNS):
    ev = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv", sep="\t")
    run_start_t = float(ev.iloc[0]["onset"])
    for _, row in ev.iterrows():
        name = "blank.jpg" if (pd.isna(row["image_name"])) else str(row["image_name"])
        onset_TR = int(round((float(row["onset"]) - run_start_t) / TR))
        global_TR = run_start[run_idx] + onset_TR
        trials.append((run_idx, name, global_TR, onset_TR))
n_trials = len(trials)
print(f"  total trials (incl. blanks): {n_trials}", flush=True)

# HRF-convolved trial regressors: shape (n_TRs_total, n_trials)
trial_design = np.zeros((n_TRs_total, n_trials), dtype=np.float32)
for j, (run_idx, name, gTR, _) in enumerate(trials):
    end = min(gTR + hrf_len, n_TRs_total)
    trial_design[gTR:end, j] = hrf[:end - gTR]


# === Build per-run nuisance (drift cosine + intercept + motion + aCompCor) ===
def make_nuisance():
    """(n_TRs_total, n_nuisance_cols)"""
    cols = []
    # Per-run intercept
    for run_idx in range(len(RUNS)):
        c = np.zeros(n_TRs_total, dtype=np.float32)
        c[run_start[run_idx]:run_start[run_idx] + n_TRs_per_run[run_idx]] = 1.0
        cols.append(c)
    # Per-run cosine drift order=1 (frequency = 1/(2*N) per nilearn cosine drift)
    for run_idx in range(len(RUNS)):
        N = n_TRs_per_run[run_idx]
        c = np.zeros(n_TRs_total, dtype=np.float32)
        t = np.arange(N)
        c[run_start[run_idx]:run_start[run_idx] + N] = np.cos(np.pi * t / N).astype(np.float32)
        cols.append(c)
    # Motion (6 cols) — concatenated per run
    mc_full = np.concatenate(mc_per_run, axis=0)
    for k in range(6):
        cols.append(mc_full[:, k].astype(np.float32))
    # aCompCor K=7 — concatenated per run
    acomp_full = np.concatenate(acomp_per_run, axis=0)
    for k in range(K_ACOMP):
        cols.append(acomp_full[:, k].astype(np.float32))
    return np.stack(cols, axis=1)


nuisance = make_nuisance()
n_nuisance = nuisance.shape[1]
print(f"  nuisance: {nuisance.shape}", flush=True)


# === Streaming OLS with ridge ===
def streaming_betas_for_tier(decode_pst):
    """For each trial, decode at trial-onset + decode_pst (or run-end for EoR),
    fit ridge OLS over [trials seen so far + nuisance], extract β_i.

    decode_pst = int → fixed offset from onset
    decode_pst = 'EoR' → run end for trial's run
    """
    betas = np.zeros((n_trials, 2792), dtype=np.float32)
    for i, (run_idx, name, gTR, onset_TR_run) in enumerate(trials):
        if isinstance(decode_pst, str) and decode_pst == "EoR":
            decode_TR_local = n_TRs_per_run[run_idx]
        else:
            decode_TR_local = onset_TR_run + decode_pst
        decode_TR_global = run_start[run_idx] + min(decode_TR_local, n_TRs_per_run[run_idx])
        decode_TR_global = min(decode_TR_global, n_TRs_total)

        # Trials seen so far: up to and including i (since trial i has onset
        # at gTR and decode_TR_global > gTR for any pst > 0)
        n_trials_so_far = i + 1
        X_trials = trial_design[:decode_TR_global, :n_trials_so_far]
        X_nuisance = nuisance[:decode_TR_global, :]
        X = np.concatenate([X_trials, X_nuisance], axis=1)        # (T, n_t + n_n)
        y = bold_full[:decode_TR_global]                          # (T, V)

        XtX = X.T @ X
        Xty = X.T @ y
        # Ridge regularization
        n, K = X.shape
        if n < K:
            # underdetermined — heavier ridge
            lam = max(np.trace(XtX) / max(K, 1) * 1e-2, 1e-4)
        else:
            lam = max(np.trace(XtX) / max(K, 1) * 1e-3, 1e-6)
        XtX_reg = XtX + lam * np.eye(K, dtype=np.float32)
        try:
            B = np.linalg.solve(XtX_reg, Xty)                     # (K, V)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(X, y, rcond=None)[0]
        betas[i] = B[i].astype(np.float32)                        # i-th trial col
        if (i + 1) % 50 == 0 or i == n_trials - 1:
            print(f"    [{decode_pst}] trial {i+1}/{n_trials} (decode_TR={decode_TR_global}, "
                  f"design {X.shape}, ridge λ={lam:.2e})", flush=True)
    return betas


# === Inclusive cum-z (matches mindeye.py:771) ===
def inclusive_cumz(arr):
    n = arr.shape[0]
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i+1].mean(axis=0)
        sd = arr[:i+1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


# Generate trial_ids (770 entries, matches existing prereg cells)
trial_ids = np.asarray([t[1] for t in trials])

CONFIGS = [
    ("RT_paper_RLS_Fast_pst5_K7CSFWM_HP_e1_inclz", 5),
    ("RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz", 20),
    ("RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz", "EoR"),
]

for cell_name, decode_pst in CONFIGS:
    out_betas = PREREG / f"{cell_name}_ses-03_betas.npy"
    if out_betas.exists():
        print(f"\n=== {cell_name} already exists — skip ===", flush=True)
        continue
    print(f"\n========== {cell_name} (decode_pst={decode_pst}) ==========", flush=True)
    t0 = time.time()
    betas = streaming_betas_for_tier(decode_pst)
    print(f"\n  raw βs: {betas.shape}, applying inclusive cum-z", flush=True)
    betas_z = inclusive_cumz(betas)
    print(f"  saved βs in cum-z'd form: {betas_z.shape}", flush=True)

    np.save(out_betas, betas_z.astype(np.float32))
    np.save(PREREG / f"{cell_name}_ses-03_trial_ids.npy", trial_ids)
    config = {
        "cell": cell_name, "session": SESSION, "runs": RUNS,
        "decode_pst": decode_pst,
        "tr": TR,
        "method": "streaming/RLS GLM — growing-design ridge OLS at decode time",
        "hrf_model": "glover",
        "drift_model": "cosine_order_1_per_run",
        "motion_cols": 6,
        "acompcor_K": K_ACOMP,
        "acompcor_pool": "CSF ∪ WM via FAST PVEs > 0.5, eroded x1",
        "acompcor_HP": "0.01 Hz",
        "ridge_lambda_scale": "1e-3 × tr(XᵀX)/K (or 1e-2 if underdetermined)",
        "cum_z_formula": "inclusive (arr[:i+1])",
        "n_trials_post": n_trials,
    }
    with open(PREREG / f"{cell_name}_ses-03_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  elapsed {(time.time()-t0)/60:.1f}m", flush=True)

print("\n=== streaming RLS GLM extraction complete ===", flush=True)
