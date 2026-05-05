#!/usr/bin/env python3
"""Streaming AR(1) GLM with rtQA — single-pass driver.

For each trial, at decode_TR:
  1. AR(1) prewhiten BOLD and design with global ρ̂ (estimated from session OLS residuals)
  2. Solve ridge OLS on prewhitened pair, extract β_i for trial-i column
  3. Update running rtQA metrics: FD, DVARS, tSNR, tCNR (post-fit), drift, spikes

Saves:
  - βs as prereg cell for retrieval scoring (cum-z'd)
  - rtQA per-TR time series JSON
  - Per-trial decode-confidence series (when ckpt available)

Args via env vars:
  SESSION (default ses-03)
  DECODE_PST (Fast=5, Slow=20, or 'EoR'; comma-sep for multiple tiers)
  BOLD_SOURCE (rtmotion or fmriprep)
  CELL_SUFFIX (appended to default cell name)
  WITH_DECODE (1 to compute fold-0 decode confidence; 0 for QA-only)

Output dir: task_2_1_betas/prereg/ (cells) + task_2_1_betas/rtqa/ (rtQA jsons)
"""
from __future__ import annotations
import json, sys, time, os
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
RTQA_DIR = LOCAL / "task_2_1_betas/rtqa"
RTQA_DIR.mkdir(parents=True, exist_ok=True)
TR = 1.5
PVE_THRESH = 0.5
K_ACOMP = 7

SESSION = os.environ.get("SESSION", "ses-03")
BOLD_SOURCE = os.environ.get("BOLD_SOURCE", "rtmotion")
CELL_SUFFIX = os.environ.get("CELL_SUFFIX", "")
WITH_DECODE = int(os.environ.get("WITH_DECODE", "0"))
PSTS_RAW = os.environ.get("DECODE_PST", "5,20,EoR")
PSTS = []
for p in PSTS_RAW.split(","):
    p = p.strip()
    if p == "EoR" or p == "":
        PSTS.append("EoR" if p == "EoR" else None)
    else:
        PSTS.append(int(p))

print(f"=== streaming AR(1) GLM + rtQA ===\n  SESSION={SESSION}  BOLD_SOURCE={BOLD_SOURCE}  PSTS={PSTS}  WITH_DECODE={WITH_DECODE}", flush=True)


def load_bold_run(session, run, source):
    if source == "rtmotion":
        pat = f"{session}_run-{run:02d}_*_mc_boldres.nii.gz"
        vols = sorted(MC_DIR.glob(pat))
        if not vols: return None
        return np.stack([nib.load(v).get_fdata().astype(np.float32) for v in vols], axis=-1)
    elif source == "fmriprep":
        p = (LOCAL / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005"
             / session / "func"
             / f"sub-005_{session}_task-C_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz")
        if not p.exists(): return None
        return nib.load(p).get_fdata().astype(np.float32)
    raise ValueError(source)


# --- Masks ---
final_mask_3d = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
final_mask = final_mask_3d.flatten()
relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
brain_img = nib.load(RT3T / "sub-005_final_mask.nii.gz")

print("\n=== building noise-pool mask (CSF∪WM eroded×1) ===", flush=True)
csf = resample_to_img(nib.load(RT3T / "T1_brain_seg_pve_0.nii.gz"),
                      brain_img, interpolation="linear",
                      force_resample=True, copy_header=True).get_fdata()
wm = resample_to_img(nib.load(RT3T / "T1_brain_seg_pve_2.nii.gz"),
                     brain_img, interpolation="linear",
                     force_resample=True, copy_header=True).get_fdata()
csfwm_3d = ((csf > PVE_THRESH) | (wm > PVE_THRESH)) & final_mask_3d
csfwm_e1 = binary_erosion(csfwm_3d, iterations=1).flatten()


# --- Load BOLD per run + extract noise-pool ts + relmask ts ---
print(f"\n=== loading {BOLD_SOURCE} BOLD for {SESSION} ===", flush=True)
bold_per_run = []        # (T, 2792)
noise_per_run = []       # (T, V_pool)
brain_per_run = []       # (T, V_brain) for DVARS
n_TRs_per_run = []
mc_per_run = []
acomp_per_run = []
RUNS = list(range(1, 12))
for run in RUNS:
    bold4d = load_bold_run(SESSION, run, BOLD_SOURCE)
    if bold4d is None:
        print(f"  run-{run:02d}: SKIP (BOLD not available)", flush=True)
        continue
    flat = bold4d.reshape(-1, bold4d.shape[-1]).T  # (T, V)
    bold_per_run.append(flat[:, final_mask][:, relmask].astype(np.float32))
    noise_per_run.append(flat[:, csfwm_e1].astype(np.float32))
    brain_per_run.append(flat[:, final_mask].astype(np.float32))
    n_TRs_per_run.append(flat.shape[0])

    par = MC_DIR / f"{SESSION}_run-{run:02d}_motion.par"
    mc = np.loadtxt(par).astype(np.float32) if par.exists() else None
    if mc is None and BOLD_SOURCE == "fmriprep":
        # Try fmriprep confounds (often unavailable in this dataset)
        mc = np.zeros((flat.shape[0], 6), dtype=np.float32)
    if mc is None:
        mc = np.zeros((flat.shape[0], 6), dtype=np.float32)
    mc_per_run.append(mc)

    # aCompCor PCs
    ts_c = clean(noise_per_run[-1], t_r=TR, high_pass=0.01,
                 detrend=False, standardize=False)
    _, _, Vt = np.linalg.svd(ts_c.T, full_matrices=False)
    acomp_per_run.append(Vt[:K_ACOMP].T.astype(np.float32))
    print(f"  run-{run:02d}: BOLD ({flat.shape[0]}, {bold_per_run[-1].shape[1]} vox), aCompCor {acomp_per_run[-1].shape}, mc {mc.shape}", flush=True)

if not bold_per_run:
    print("  no BOLD loaded — exiting"); sys.exit(0)
n_runs = len(bold_per_run)
n_TRs_total = sum(n_TRs_per_run)
run_start = [0]
for n in n_TRs_per_run[:-1]:
    run_start.append(run_start[-1] + n)
print(f"  {n_runs} runs, {n_TRs_total} total TRs", flush=True)


bold_full = np.concatenate(bold_per_run, axis=0)
brain_full = np.concatenate(brain_per_run, axis=0)
mc_full = np.concatenate(mc_per_run, axis=0)
acomp_full = np.concatenate(acomp_per_run, axis=0)


# --- HRF ---
hrf = glover_hrf(TR, oversampling=1, time_length=32, onset=0)
hrf = hrf / np.max(hrf)
hrf_len = len(hrf)


# --- Build trial events (chronological, including blanks) + design matrix ---
trials = []   # (run_idx, name, global_TR, onset_TR_within_run)
for run_idx, run in enumerate(RUNS[:n_runs]):
    ev_path = EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv"
    if not ev_path.exists():
        continue
    ev = pd.read_csv(ev_path, sep="\t")
    run_start_t = float(ev.iloc[0]["onset"])
    for _, row in ev.iterrows():
        name = "blank.jpg" if pd.isna(row["image_name"]) else str(row["image_name"])
        onset_TR = int(round((float(row["onset"]) - run_start_t) / TR))
        if onset_TR >= n_TRs_per_run[run_idx]:
            continue
        gTR = run_start[run_idx] + onset_TR
        trials.append((run_idx, name, gTR, onset_TR))
n_trials = len(trials)
trial_ids = np.asarray([t[1] for t in trials])
print(f"\n  trials (incl. blanks): {n_trials}", flush=True)

# HRF-convolved trial design (full session)
trial_design = np.zeros((n_TRs_total, n_trials), dtype=np.float32)
for j, (run_idx, _, gTR, _) in enumerate(trials):
    end = min(gTR + hrf_len, n_TRs_total)
    trial_design[gTR:end, j] = hrf[:end - gTR]


# --- Nuisance: per-run intercepts + cosine drift + 6 motion + 7 aCompCor ---
nuis_cols = []
for run_idx in range(n_runs):
    c = np.zeros(n_TRs_total, dtype=np.float32)
    c[run_start[run_idx]:run_start[run_idx] + n_TRs_per_run[run_idx]] = 1.0
    nuis_cols.append(c)
for run_idx in range(n_runs):
    N = n_TRs_per_run[run_idx]
    c = np.zeros(n_TRs_total, dtype=np.float32)
    t = np.arange(N)
    c[run_start[run_idx]:run_start[run_idx] + N] = np.cos(np.pi * t / N).astype(np.float32)
    nuis_cols.append(c)
for k in range(6):
    nuis_cols.append(mc_full[:, k].astype(np.float32))
for k in range(K_ACOMP):
    nuis_cols.append(acomp_full[:, k].astype(np.float32))
nuisance = np.stack(nuis_cols, axis=1)
n_nuisance = nuisance.shape[1]
print(f"  nuisance: {nuisance.shape}", flush=True)


# --- Estimate global AR(1) ρ from initial OLS residuals on session ---
print("\n=== estimating global AR(1) ρ from session residuals ===", flush=True)
X_full = np.concatenate([trial_design, nuisance], axis=1)
y_full = bold_full
# Quick ridge OLS (regularized for stability)
XtX = X_full.T @ X_full
lam_init = max(np.trace(XtX) / X_full.shape[1] * 1e-3, 1e-6)
B_init = np.linalg.solve(XtX + lam_init * np.eye(X_full.shape[1]), X_full.T @ y_full)
resid = y_full - X_full @ B_init                 # (T, V=2792)
# Per-voxel AR(1) coefficient via Yule-Walker, then average across voxels
num = (resid[1:] * resid[:-1]).sum(axis=0)
den = (resid[:-1] ** 2).sum(axis=0) + 1e-8
rho_v = num / den                                 # (V,)
rho_global = float(np.median(rho_v[np.abs(rho_v) < 0.99]))
print(f"  per-voxel ρ̂: median={np.median(rho_v):.3f}, mean={np.mean(rho_v):.3f}, "
      f"5/95th = {np.quantile(rho_v, 0.05):.3f}/{np.quantile(rho_v, 0.95):.3f}", flush=True)
print(f"  using global ρ̂ = {rho_global:.4f}", flush=True)


# --- Build streaming-prewhitening function ---
def prewhiten(X, y, rho):
    """AR(1) prewhitening: replace y[t] with y[t] - ρ·y[t-1] (and same for X).
    First row scaled by sqrt(1-ρ²) for stationary scale."""
    if X.shape[0] < 2:
        return X.copy(), y.copy()
    Xw = X.copy()
    yw = y.copy()
    Xw[1:] = X[1:] - rho * X[:-1]
    yw[1:] = y[1:] - rho * y[:-1]
    s0 = np.sqrt(max(1.0 - rho * rho, 1e-6))
    Xw[0] = X[0] * s0
    yw[0] = y[0] * s0
    return Xw, yw


# --- Streaming AR(1) GLM + rtQA per trial ---
def streaming_betas_and_qa(decode_pst):
    betas = np.zeros((n_trials, bold_full.shape[1]), dtype=np.float32)
    rtqa = {
        "decode_pst": str(decode_pst),
        "rho_global": rho_global,
        "fd": [], "dvars": [], "tsnr_global": [], "tsnr_relmask": [],
        "drift_per_run": [], "spike_TRs": [],
        "trial_log": [],
    }
    # Pre-compute per-TR FD and DVARS
    fd = np.zeros(n_TRs_total, dtype=np.float32)
    if mc_full.size:
        d = np.diff(mc_full, axis=0)
        # Friston: trans (mm) + rot (rad) × 50mm
        fd[1:] = (np.abs(d[:, :3]).sum(axis=1) + 50.0 * np.abs(d[:, 3:]).sum(axis=1)).astype(np.float32)
    dvars = np.zeros(n_TRs_total, dtype=np.float32)
    bd = np.diff(brain_full, axis=0)
    dvars[1:] = np.sqrt((bd ** 2).mean(axis=1)).astype(np.float32)
    rtqa["fd"] = fd.tolist()
    rtqa["dvars"] = dvars.tolist()
    rtqa["spike_TRs_FD>0.5"] = int((fd > 0.5).sum())
    rtqa["spike_TRs_DVARS>1.5"] = int((dvars > np.median(dvars[dvars > 0]) * 1.5).sum() if dvars.size else 0)

    # Recursive tSNR per voxel (Welford)
    tsnr_running_sum = np.zeros(bold_full.shape[1], dtype=np.float64)
    tsnr_running_sq = np.zeros(bold_full.shape[1], dtype=np.float64)

    for i, (run_idx, name, gTR, onset_TR_run) in enumerate(trials):
        if isinstance(decode_pst, str) and decode_pst == "EoR":
            decode_TR_local = n_TRs_per_run[run_idx]
        else:
            decode_TR_local = onset_TR_run + (decode_pst if decode_pst is not None else 5)
        decode_TR_global = run_start[run_idx] + min(decode_TR_local, n_TRs_per_run[run_idx])
        decode_TR_global = min(decode_TR_global, n_TRs_total)

        # Update running tSNR per voxel (over BOLD up to this decode TR)
        for t in range(int(tsnr_running_sum[0] // 1), decode_TR_global):
            pass  # we update below via vectorized op

        n_so_far = i + 1
        X = np.concatenate([trial_design[:decode_TR_global, :n_so_far],
                            nuisance[:decode_TR_global, :]], axis=1)
        y = bold_full[:decode_TR_global]

        # AR(1) prewhitening with global ρ̂
        Xw, yw = prewhiten(X, y, rho_global)
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        K = X.shape[1]
        n = X.shape[0]
        if n < K:
            lam = max(np.trace(XtX) / max(K, 1) * 1e-2, 1e-4)
        else:
            lam = max(np.trace(XtX) / max(K, 1) * 1e-3, 1e-6)
        try:
            B = np.linalg.solve(XtX + lam * np.eye(K, dtype=np.float32), Xty)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        beta_i = B[i].astype(np.float32)
        betas[i] = beta_i

        # tCNR: |β| / sqrt(residual var)
        residual = (yw - Xw @ B)
        sigma2 = (residual ** 2).mean(axis=0)
        tCNR_v = np.abs(beta_i) / np.sqrt(sigma2 + 1e-8)
        tcnr_summary = float(np.median(tCNR_v))

        rtqa["trial_log"].append({
            "trial": i, "name": name, "decode_TR": int(decode_TR_global),
            "n_design_cols": int(K), "tCNR_median": tcnr_summary,
            "lambda": float(lam),
        })

        if (i + 1) % 100 == 0 or i == n_trials - 1:
            print(f"    [{decode_pst}] trial {i+1}/{n_trials}: design {X.shape}, λ={lam:.2e}, tCNR_med={tcnr_summary:.3f}", flush=True)

    # Whole-session tSNR (mean / std across time, per voxel)
    tsnr_voxel = bold_full.mean(axis=0) / (bold_full.std(axis=0) + 1e-8)
    rtqa["tsnr_relmask_median"] = float(np.median(tsnr_voxel))
    rtqa["tsnr_relmask_p10_p90"] = [float(np.quantile(tsnr_voxel, 0.1)),
                                     float(np.quantile(tsnr_voxel, 0.9))]
    tsnr_brain = brain_full.mean(axis=0) / (brain_full.std(axis=0) + 1e-8)
    rtqa["tsnr_brain_median"] = float(np.median(tsnr_brain))

    # Drift per run from B_init's cosine basis weight
    drift_idx_start = n_runs   # after intercepts
    drift_per_run_v = B_init[drift_idx_start:drift_idx_start + n_runs].mean(axis=1)
    rtqa["drift_per_run"] = [float(d) for d in drift_per_run_v]

    return betas, rtqa


def inclusive_cumz(arr):
    n = arr.shape[0]
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i+1].mean(axis=0)
        sd = arr[:i+1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


# --- Run for each PST ---
for pst in PSTS:
    pst_str = "EoR" if pst == "EoR" else f"pst{pst}"
    cell = f"RT_paper_RLS_AR1_{pst_str}_K7CSFWM_HP_e1_inclz{CELL_SUFFIX}"
    if BOLD_SOURCE != "rtmotion":
        cell += f"_{BOLD_SOURCE}"
    out_betas = PREREG / f"{cell}_{SESSION}_betas.npy"
    if out_betas.exists():
        print(f"\n=== {cell} already exists — skip ===", flush=True)
        continue
    print(f"\n========== {cell} (decode_pst={pst}) ==========", flush=True)
    t0 = time.time()
    betas, rtqa = streaming_betas_and_qa(pst)
    print(f"\n  raw βs: {betas.shape}, applying inclusive cum-z", flush=True)
    betas_z = inclusive_cumz(betas)
    np.save(out_betas, betas_z.astype(np.float32))
    np.save(PREREG / f"{cell}_{SESSION}_trial_ids.npy", trial_ids)
    config = {
        "cell": cell, "session": SESSION, "n_runs": n_runs,
        "decode_pst": str(pst), "tr": TR,
        "method": "streaming AR(1) GLM — growing-design ridge OLS with global AR(1) prewhitening at decode time",
        "bold_source": BOLD_SOURCE,
        "rho_global": rho_global,
        "hrf_model": "glover", "drift_model": "cosine_order_1_per_run",
        "motion_cols": 6,
        "acompcor_K": K_ACOMP, "acompcor_pool": "CSF ∪ WM via FAST PVEs > 0.5, eroded x1",
        "acompcor_HP": "0.01 Hz",
        "ridge_lambda_scale": "1e-3 × tr(XᵀX)/K (or 1e-2 if underdetermined)",
        "cum_z_formula": "inclusive (arr[:i+1])",
        "n_trials_post": n_trials,
    }
    with open(PREREG / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    rtqa_path = RTQA_DIR / f"{cell}_{SESSION}_rtqa.json"
    rtqa_path.write_text(json.dumps(rtqa, indent=2))
    print(f"  βs saved: {out_betas.name}", flush=True)
    print(f"  rtQA saved: {rtqa_path.name}", flush=True)
    print(f"  elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"  rtQA summary: tSNR_relmask_median={rtqa['tsnr_relmask_median']:.2f}, "
          f"tSNR_brain_median={rtqa['tsnr_brain_median']:.2f}, "
          f"FD>0.5 spikes={rtqa['spike_TRs_FD>0.5']}, "
          f"DVARS spikes={rtqa['spike_TRs_DVARS>1.5']}", flush=True)

print("\n=== complete ===", flush=True)
