#!/usr/bin/env python3
"""Post-hoc fix the FD column-order bug in rtQA jsons.

Friston FD = Σ|Δrot|·50mm + Σ|Δtrans|·1mm
MCFLIRT .par columns: [rot_x, rot_y, rot_z (rad), trans_x, trans_y, trans_z (mm)]

Original buggy code: Σ|Δd[:,:3]| + 50·Σ|Δd[:,3:]|  (treats translations as the multiplied term)
Correct: 50·Σ|Δd[:,:3]| + Σ|Δd[:,3:]|             (rotations get the 50mm radius factor)

Also produces a cross-session summary doc.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RTQA_DIR = LOCAL / "task_2_1_betas/rtqa"
MC_DIR = LOCAL / "motion_corrected_resampled"

# Recompute FD where motion params exist
def recompute_fd(session, n_TRs_per_run, run_start, n_runs):
    fd = np.zeros(n_TRs_per_run[-1] + run_start[-1], dtype=np.float32)
    have_par = True
    mc_full = []
    for run in range(1, n_runs + 1):
        par = MC_DIR / f"{session}_run-{run:02d}_motion.par"
        if par.exists():
            mc = np.loadtxt(par).astype(np.float32)
            mc_full.append(mc)
        else:
            have_par = False
            break
    if not have_par or not mc_full:
        return None
    mc_full = np.concatenate(mc_full, axis=0)
    d = np.diff(mc_full, axis=0)
    # CORRECT: rotations × 50mm + translations × 1
    fd[1:] = (50.0 * np.abs(d[:, :3]).sum(axis=1) + np.abs(d[:, 3:]).sum(axis=1)).astype(np.float32)
    return fd


for jp in sorted(RTQA_DIR.glob("*_rtqa.json")):
    j = json.loads(jp.read_text())
    name = jp.stem
    # Extract session from filename
    if "_ses-" in name:
        ses = "ses-" + name.split("_ses-")[-1].replace("_rtqa", "")
    else:
        continue

    # Try to recompute FD from MCFLIRT .par
    n_runs = 11
    n_TRs_per_run = [192] * n_runs
    run_start = [i * 192 for i in range(n_runs)]
    if "ses-06" in name:
        n_runs = 5
        n_TRs_per_run = [192] * 5
        run_start = [i * 192 for i in range(5)]

    fd_new = recompute_fd(ses, n_TRs_per_run, run_start, n_runs)
    if fd_new is not None:
        j["fd"] = fd_new.tolist()
        j["fd_max"] = float(fd_new.max())
        j["fd_mean"] = float(fd_new.mean())
        j["spike_TRs_FD>0.5"] = int((fd_new > 0.5).sum())
        j["spike_TRs_FD>0.3"] = int((fd_new > 0.3).sum())
        j["fd_status"] = "recomputed (rotations × 50mm + translations × 1)"
    else:
        j["fd"] = []
        j["fd_max"] = None
        j["fd_mean"] = None
        j["spike_TRs_FD>0.5"] = None
        j["fd_status"] = f"motion .par files not available for {ses}"

    jp.write_text(json.dumps(j, indent=2))
    ts = j.get('tsnr_relmask_median', '?')
    ts_str = f"{ts:.2f}" if isinstance(ts, (int, float)) else str(ts)
    print(f"  {jp.name}: tSNR_relmask={ts_str}", flush=True)


# Cross-session summary
print("\n========== CROSS-SESSION rtQA SUMMARY ==========")
print(f"{'Session':10s} {'BOLD':10s} {'tSNR_relmask':>13s} {'tSNR_brain':>11s} {'DVARS spikes':>13s} {'FD>0.5 spikes':>14s} {'ρ_global':>9s} {'drift mean':>11s}")
sessions = {}
for jp in sorted(RTQA_DIR.glob("*_rtqa.json")):
    j = json.loads(jp.read_text())
    name = jp.stem
    if "qaonly" in name:
        bold = "fmriprep"
    else:
        bold = "rtmotion"
    if "_ses-" in name:
        ses = "ses-" + name.split("_ses-")[-1].replace("_rtqa", "")
    if "EoR" not in name:
        continue   # only compare EoR cells
    rho = j.get("rho_global", "?")
    drift = j.get("drift_per_run", [])
    drift_mean = np.mean(drift) if drift else None
    drift_str = f"{drift_mean:11.4f}" if drift_mean is not None else f"{'---':>11s}"
    fd_spikes = j.get('spike_TRs_FD>0.5', '?')
    fd_spikes_str = f"{fd_spikes:14d}" if isinstance(fd_spikes, int) else f"{str(fd_spikes):>14s}"
    rho_str = f"{rho:9.4f}" if isinstance(rho, (int, float)) else f"{str(rho):>9s}"
    print(f"{ses:10s} {bold:10s} {j.get('tsnr_relmask_median', 0):13.2f} {j.get('tsnr_brain_median', 0):11.2f} "
          f"{j.get('spike_TRs_DVARS>1.5', '?'):>13} {fd_spikes_str} "
          f"{rho_str} {drift_str}")
