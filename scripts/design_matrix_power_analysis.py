#!/usr/bin/env python3
"""FSL-FEAT-style power analysis for RT vs offline design matrices.

Builds three design matrices for sub-005 ses-03 task-C:

  1. **RT LSS Fast** — single-trial probe-vs-reference design windowed at
     onset+5 TRs (Fast RT tier). Tiny matrix (~5 TRs × few cols),
     poorly conditioned because so few rows.

  2. **RT LSS EoR** — same LSS structure but full-run BOLD (192 TRs).
     This is the End-of-Run RT row.

  3. **Offline LSR** — single concatenated design across all 11 runs of the
     session: one regressor per non-blank trial (693 columns), per-run
     intercept dummies (10 + grand intercept), per-run cosine drift basis.
     Mirrors what GLMsingle Stage 3 fracridge effectively conditions on.

For each design X we compute:

  * **Efficiency** = 1 / mean(diag((X'X)^-1))  (higher = more separable
    signal per regressor; FSL FEAT's fundamental criterion).
  * **Condition number** of X'X (higher = more ill-conditioned).
  * **Per-regressor VIF** = 1 / (1 - R²_j), reported as the mean and 95-pct
    of all regressors. VIF > 5 is borderline; > 10 is multicollinear.
  * **Per-run regressor diagnostics** for the offline design (does the
    inclusion of per-run intercepts inflate VIF on adjacent regressors?).

Outputs:
  /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/design_*.png
  /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/design_power.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

# ---------------------------------------------------------------------------
EVENTS_DIR = Path("/data/derivatives/rtmindeye_paper/rt3t/data/events")
OUT_DIR = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")
SUBJ, SES = "sub-005", "ses-03"
TASK = "task-C"
T_R = 1.5
N_TRS_PER_RUN = 192
N_RUNS = 11
HIGH_PASS = 0.01
HRF_MODEL = "glover"
DRIFT_MODEL = "cosine"
DRIFT_ORDER = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run_events(run: int) -> pd.DataFrame:
    """Load events.tsv, drop blanks, keep onset/duration/trial_type."""
    f = EVENTS_DIR / f"{SUBJ}_{SES}_{TASK}_run-{run:02d}_events.tsv"
    df = pd.read_csv(f, sep="\t")
    df = df[df["image_name"] != "blank.jpg"].reset_index(drop=True)
    return df[["onset", "duration", "image_name"]].copy()


# ---------------------------------------------------------------------------
# Design-matrix builders
# ---------------------------------------------------------------------------
def build_rt_lss(run: int, trial_idx: int, n_post_stim_TRs: int) -> pd.DataFrame:
    """One LSS design matrix: probe (single trial) + reference (rest)
    + cosine drift + intercept.  BOLD window: onset_TR + n_post_stim_TRs.

    Returns the design matrix as a (n_TRs, n_cols) DataFrame matching what
    nilearn.FirstLevelModel constructs.
    """
    ev = load_run_events(run)
    if trial_idx >= len(ev):
        raise ValueError(f"trial_idx {trial_idx} > n_trials {len(ev)}")
    onset_TR = int(round(ev.loc[trial_idx, "onset"] / T_R))
    decode_TR = min(onset_TR + n_post_stim_TRs, N_TRS_PER_RUN - 1)
    n_trs = decode_TR + 1

    lss = ev.copy()
    lss["trial_type"] = ["probe" if i == trial_idx else "reference"
                          for i in range(len(lss))]
    # Drop trials whose onset is past the BOLD window
    lss = lss[lss["onset"] <= n_trs * T_R].reset_index(drop=True)

    frame_times = np.arange(n_trs) * T_R
    return make_first_level_design_matrix(
        frame_times, lss,
        hrf_model=HRF_MODEL, drift_model=DRIFT_MODEL,
        drift_order=DRIFT_ORDER, high_pass=HIGH_PASS,
    )


def build_offline_lsr() -> tuple[pd.DataFrame, list[int]]:
    """Concatenated all-11-runs design with per-trial regressors + per-run
    intercept dummies + per-run cosine drift basis.

    Returns (design_matrix, run_starts) where run_starts[i] is the offset of
    run i+1 in the concatenated TR axis.
    """
    Xs, run_starts = [], [0]
    for run in range(1, N_RUNS + 1):
        ev = load_run_events(run).copy()
        # Each trial gets a unique label so it becomes its own column
        ev["trial_type"] = [f"r{run:02d}_t{i:03d}" for i in range(len(ev))]
        # Shift onset to global session time so concat works cleanly
        ev["onset"] = ev["onset"] + run_starts[-1] * T_R
        frame_times = (np.arange(N_TRS_PER_RUN)
                        + run_starts[-1]) * T_R
        X = make_first_level_design_matrix(
            frame_times, ev,
            hrf_model=HRF_MODEL, drift_model=DRIFT_MODEL,
            drift_order=DRIFT_ORDER, high_pass=HIGH_PASS,
        )
        # Rename per-run drift cols so they don't collide between runs
        rename = {c: f"run{run:02d}_{c}" for c in X.columns
                   if c.startswith("drift") or c == "constant"}
        X = X.rename(columns=rename)
        Xs.append(X)
        run_starts.append(run_starts[-1] + N_TRS_PER_RUN)

    # Outer-join columns; missing trial-cols → 0 in other runs
    X_all = pd.concat(Xs, axis=0).fillna(0.0)
    X_all = X_all.reset_index(drop=True)
    return X_all, run_starts[:-1]


# ---------------------------------------------------------------------------
# FEAT-style power analysis
# ---------------------------------------------------------------------------
def power_metrics(X: np.ndarray, name: str) -> dict:
    """Compute FEAT-style efficiency / condition / VIF for a design matrix."""
    n_TRs, n_cols = X.shape
    Xtx = X.T @ X
    rcond = np.linalg.cond(Xtx)
    # Use pseudo-inverse so we don't crash on rank deficiency
    inv = np.linalg.pinv(Xtx)
    diag_inv = np.diag(inv)
    eff = 1.0 / np.mean(diag_inv[diag_inv > 0]) if (diag_inv > 0).any() else 0.0

    # Per-regressor VIF: regress each col on others, VIF = 1/(1-R²).
    # For huge matrices this is O(p^3); skip if > 200 cols and just report
    # the diag(inv) distribution as a proxy.
    if n_cols <= 200:
        vifs = []
        for j in range(n_cols):
            x_j = X[:, j]
            X_other = np.delete(X, j, axis=1)
            try:
                beta, *_ = np.linalg.lstsq(X_other, x_j, rcond=None)
                resid = x_j - X_other @ beta
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((x_j - x_j.mean()) ** 2)
                if ss_tot < 1e-12:
                    vifs.append(np.inf)
                else:
                    r2 = 1 - ss_res / ss_tot
                    vifs.append(1 / max(1e-12, 1 - r2))
            except np.linalg.LinAlgError:
                vifs.append(np.inf)
        vifs = np.asarray(vifs)
        vif_finite = vifs[np.isfinite(vifs)]
        vif_summary = {
            "median": float(np.median(vif_finite)) if len(vif_finite) else None,
            "p95":    float(np.percentile(vif_finite, 95)) if len(vif_finite) else None,
            "max":    float(np.max(vif_finite)) if len(vif_finite) else None,
            "n_inf":  int(np.sum(~np.isfinite(vifs))),
        }
    else:
        vif_summary = {"skipped": "n_cols > 200; diag(inv) used as proxy"}

    return {
        "name": name,
        "n_TRs": n_TRs,
        "n_regressors": n_cols,
        "efficiency": float(eff),
        "condition_number": float(rcond),
        "diag_inv_median": float(np.median(diag_inv[diag_inv > 0]))
                           if (diag_inv > 0).any() else None,
        "vif": vif_summary,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_design(X: pd.DataFrame, title: str, out_path: Path,
                 run_starts: list[int] = None) -> None:
    fig, ax = plt.subplots(figsize=(min(14, max(6, X.shape[1] / 8)), 7))
    arr = X.values
    if arr.shape[1] > 200:
        # subsample columns for visualization to keep figure readable
        keep = np.linspace(0, arr.shape[1] - 1, 200, dtype=int)
        arr = arr[:, keep]
    vmax = np.nanpercentile(np.abs(arr), 99) if arr.size else 1.0
    ax.imshow(arr, aspect="auto", cmap="RdBu_r",
              vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Regressor index")
    ax.set_ylabel("TR")
    ax.set_title(title)
    if run_starts is not None:
        for r in run_starts[1:]:
            ax.axhline(r - 0.5, color="0.3", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}", flush=True)


def plot_xtx(X: pd.DataFrame, title: str, out_path: Path) -> None:
    """Heatmap of (X'X)^-1 — diagonal = regressor variance, off-diag = correl."""
    fig, ax = plt.subplots(figsize=(7, 6))
    Xtx = X.values.T @ X.values
    inv = np.linalg.pinv(Xtx)
    if inv.shape[0] > 200:
        keep = np.linspace(0, inv.shape[0] - 1, 200, dtype=int)
        inv = inv[np.ix_(keep, keep)]
    vmax = np.nanpercentile(np.abs(inv), 99) if inv.size else 1.0
    ax.imshow(inv, aspect="auto", cmap="RdBu_r",
              vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title + r" — $(X^TX)^{-1}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--rt-trial", type=int, default=1,
                     help="trial index for RT LSS designs (default: 1, "
                          "the first special515 in run-01)")
    ap.add_argument("--rt-run", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Power analysis for {SUBJ}_{SES}_{TASK}", flush=True)
    print(f"  TR={T_R}s, n_TRs/run={N_TRS_PER_RUN}, n_runs={N_RUNS}, "
          f"high_pass={HIGH_PASS} Hz, drift={DRIFT_MODEL}", flush=True)

    print("\n[1] RT LSS — Fast tier (decode_TR = onset + 5 TRs)", flush=True)
    X_fast = build_rt_lss(args.rt_run, args.rt_trial, n_post_stim_TRs=5)
    print(f"  shape={X_fast.shape}  cols={list(X_fast.columns)[:6]}...",
          flush=True)
    m_fast = power_metrics(X_fast.values, "RT_LSS_Fast")
    plot_design(X_fast, f"RT LSS Fast — run {args.rt_run} trial {args.rt_trial}  "
                f"(onset+5 TRs)", out_dir / "design_rt_lss_fast.png")
    plot_xtx(X_fast, "RT LSS Fast", out_dir / "design_rt_lss_fast_XTXinv.png")

    print("\n[2] RT LSS — End-of-Run tier (full 192 TRs)", flush=True)
    X_eor = build_rt_lss(args.rt_run, args.rt_trial,
                          n_post_stim_TRs=N_TRS_PER_RUN)
    print(f"  shape={X_eor.shape}  cols={list(X_eor.columns)[:6]}...",
          flush=True)
    m_eor = power_metrics(X_eor.values, "RT_LSS_EoR")
    plot_design(X_eor, f"RT LSS EoR — run {args.rt_run} trial {args.rt_trial}  "
                f"(full run)", out_dir / "design_rt_lss_eor.png")
    plot_xtx(X_eor, "RT LSS EoR", out_dir / "design_rt_lss_eor_XTXinv.png")

    print("\n[3] Offline LSR — concatenated 11 runs (per-trial regressors + "
          "per-run drift)", flush=True)
    X_off, run_starts = build_offline_lsr()
    print(f"  shape={X_off.shape}  trial_cols={(X_off.columns.str.match(r'r\d+_t\d+')).sum()}  "
          f"drift_cols={(X_off.columns.str.contains('drift')).sum()}  "
          f"intercept_cols={(X_off.columns.str.contains('constant')).sum()}",
          flush=True)
    m_off = power_metrics(X_off.values, "Offline_LSR_concatenated")
    plot_design(X_off, "Offline LSR — 11 runs concatenated, "
                "per-trial regressors + per-run intercepts/drift",
                out_dir / "design_offline_lsr.png", run_starts=run_starts)
    plot_xtx(X_off, "Offline LSR (subsampled)",
              out_dir / "design_offline_lsr_XTXinv.png")

    summary = {
        "subj": SUBJ, "session": SES, "task": TASK,
        "TR": T_R, "n_TRs_per_run": N_TRS_PER_RUN, "n_runs": N_RUNS,
        "high_pass_hz": HIGH_PASS, "drift_model": DRIFT_MODEL,
        "drift_order": DRIFT_ORDER, "hrf_model": HRF_MODEL,
        "designs": [m_fast, m_eor, m_off],
    }
    out_path = out_dir / "design_power.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"\n{'design':<28} {'TRs':>5} {'cols':>6} {'efficiency':>12} "
          f"{'cond':>10}", flush=True)
    for m in (m_fast, m_eor, m_off):
        print(f"{m['name']:<28} {m['n_TRs']:>5} {m['n_regressors']:>6} "
              f"{m['efficiency']:>12.4g} {m['condition_number']:>10.4g}",
              flush=True)


if __name__ == "__main__":
    main()
