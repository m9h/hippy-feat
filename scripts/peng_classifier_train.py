#!/usr/bin/env python3
"""Train Peng's per-pair classifiers from a sub's recognition runs.

Replicates `data_preprocess/prepare_coActivation_fig2c/clfTraining/clfTraining.py`
from the KailongPeng/real_time_neurofeedback repo, adapted to:
  - read the public `run_*_bet.nii` (Dryad ships these instead of the
    pre-extracted `brain_run*.npy` their pipeline uses internally)
  - drop the FSL-side unwarp pre-step (Dryad data is already MC + skull-stripped)
  - default to single-subject for the proof-of-concept

Outputs:
  /data/datasets/peng_2024_neurofeedback/derivatives/{sub}/clf/
    bedchair_bedtable.joblib  (LogReg trained to discriminate {bed, table}
                               on the bed-vs-chair pair's training data, etc.)
    train_summary.json         {clf_name: {loo_mean_acc, n_train_trials}, ...}

Item code map (Peng's convention):
  A=bed, B=chair, C=table, D=bench
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression


PENG_ROOT = Path("/data/datasets/peng_2024_neurofeedback/subjects")
DERIV_ROOT = Path("/data/datasets/peng_2024_neurofeedback/derivatives")

# Classifier name template that matches Peng's convention exactly so the
# downstream feedback-replication script can drop in unchanged. Peng's
# naming is `{pair}_{target}{other}` where `pair` is the underlying
# 2-class training pair and the target/other are how the classifier is
# applied at feedback time.
ITEM_OBJECTS = ["bed", "bench", "chair", "table"]


def normalize(X: np.ndarray) -> np.ndarray:
    """Voxelwise z-score (per session) across TRs; NaN-safe."""
    Z = zscore(X, axis=0)
    Z[np.isnan(Z)] = 0
    return Z


def load_recognition_run(sub: str, ses: str, run: int, mask_3d: np.ndarray
                          ) -> tuple[np.ndarray, pd.DataFrame]:
    """Load motion-corrected BOLD and behavior for one recognition run.

    Returns:
        bold_masked: (T, n_voxels) float32 — voxels reduced to chosenMask.
        behav: per-trial DataFrame with TR index and item label.
    """
    sub_dir = PENG_ROOT / sub / sub / ses
    bold_path = sub_dir / "recognition" / f"run_{run}_bet.nii"
    behav_path = sub_dir / "recognition" / f"behav_run{run}.csv"
    img = nib.load(bold_path)
    vol = img.get_fdata()                           # (X, Y, Z, T)
    T = vol.shape[-1]
    flat = vol.reshape(-1, T).T                     # (T, V_full)
    mask_flat = mask_3d.flatten() != 0
    bold = flat[:, mask_flat].astype(np.float32)    # (T, n_mask_vox)
    behav = pd.read_csv(behav_path)
    return bold, behav


def collect_session(sub: str, ses: str, runs: list[int],
                    mask_3d: np.ndarray, item_to_label: dict[str, str],
                    hrf_delay_TRs: int = 2
                    ) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-run: load BOLD → pull HRF-shifted trial TRs → z-score across trials.

    Matches Peng's `noUnwarpPreprocess.py` ordering: extract trial volumes
    FIRST (with the +2 TR HRF shift), THEN normalize across trials within
    a run. Z-scoring across all-TR timeseries before extraction would
    express features in z-units of the run-mean (mostly noise voxels);
    z-scoring across trials makes each feature reflect deviation from the
    trial-mean per voxel, which is what the classifier was trained against.
    """
    trial_Xs = []
    trial_metas = []
    for new_run_idx, run in enumerate(runs, start=1):
        bold, behav = load_recognition_run(sub, ses, run, mask_3d)  # (T, V)
        # +2 TR HRF shift
        T = bold.shape[0]
        trs = behav["TR"].astype(int).values + hrf_delay_TRs
        valid = trs < T
        if not valid.all():
            n_drop = (~valid).sum()
            print(f"  WARN: dropping {n_drop} trials in run-{run} whose "
                  f"HRF-shifted TR exceeds run length {T}")
            trs = trs[valid]
            behav = behav[valid].reset_index(drop=True)
        per_trial = bold[trs]                       # (n_trials, V)
        # Z-score across trials within this run, per voxel
        mu = per_trial.mean(axis=0, keepdims=True)
        sd = per_trial.std(axis=0, keepdims=True) + 1e-8
        per_trial = (per_trial - mu) / sd
        per_trial = np.nan_to_num(per_trial)
        trial_Xs.append(per_trial)
        behav = behav.copy()
        behav["run_num"] = new_run_idx
        behav["label"] = behav["Item"].map(item_to_label)
        trial_metas.append(behav)
    return np.concatenate(trial_Xs, axis=0), pd.concat(trial_metas, ignore_index=True)


def features_at_trial_TRs(bold_concat: np.ndarray, behav: pd.DataFrame,
                          run_lengths: list[int],
                          hrf_delay_TRs: int = 2
                          ) -> tuple[np.ndarray, pd.DataFrame]:
    """Slice HRF-shifted trial TRs from the concatenated BOLD.

    `behav['TR']` indexes the stimulus-onset TR within its run. Peng's
    `noUnwarpPreprocess.py:360` shifts BOLD forward by **2 TRs (4 sec)**
    before extraction — `Brain_TR = np.arange(...) + 2; Brain_TR[behav.TR]`.
    That puts the feature at the canonical Glover HRF peak. Without this
    shift the LogReg LOO accuracy collapses to chance (~25 %).
    """
    cumulative = np.concatenate([[0], np.cumsum(run_lengths)[:-1]])
    abs_trs = []
    for _, row in behav.iterrows():
        run_idx = int(row["run_num"]) - 1
        t = int(row["TR"]) + hrf_delay_TRs
        abs_trs.append(cumulative[run_idx] + t)
    abs_trs = np.asarray(abs_trs, dtype=int)
    valid = abs_trs < bold_concat.shape[0]
    if not valid.all():
        n_drop = (~valid).sum()
        print(f"  WARN: dropping {n_drop} trials whose HRF-shifted TR index "
              f"exceeds the concatenated run length")
        abs_trs = abs_trs[valid]
        behav = behav[valid].reset_index(drop=True)
    return bold_concat[abs_trs], behav


def train_pairwise_clfs(X: np.ndarray, behav: pd.DataFrame
                        ) -> tuple[dict, dict, float]:
    """Train Peng's full set of pairwise LogReg classifiers + LOO acc per name.

    Naming convention: `{pair[0]}{pair[1]}_{obj}{altobj}`. The classifier
    is trained on the pair's two classes (obj vs altobj) with leave-one-run-out.
    We refit on ALL data after LOO to ship the production model.

    Returns:
        clf_dict: {name: fitted LogisticRegression on all data}
        loo_acc:  {name: float} mean LOO accuracy
        fourway_acc: held-out 4-way accuracy mean over LOO
    """
    runs = np.unique(behav["run_num"])
    print(f"  pairwise classifiers — {len(runs)} runs in LOO")

    # 4-way LogReg LOO accuracy (sanity)
    fourway_accs = []
    for test_run in runs:
        tr_mask = behav["run_num"] != int(test_run)
        te_mask = behav["run_num"] == int(test_run)
        # multi_class arg was removed in sklearn 1.7; multinomial is now
        # the default for multiclass solvers like lbfgs.
        clf4 = LogisticRegression(penalty="l2", C=1, solver="lbfgs",
                                  max_iter=1000)
        clf4.fit(X[tr_mask], behav.loc[tr_mask, "label"])
        fourway_accs.append(clf4.score(X[te_mask], behav.loc[te_mask, "label"]))
    fourway_mean = float(np.mean(fourway_accs))
    print(f"  4-way LOO mean acc = {fourway_mean:.3f}")

    # Pairwise classifiers — Peng's convention
    clf_dict = {}
    loo_acc = {}
    for pair in itertools.combinations(ITEM_OBJECTS, 2):
        for obj in pair:
            for altobj in [a for a in ITEM_OBJECTS if a not in pair]:
                name = f"{pair[0]}{pair[1]}_{obj}{altobj}"
                # Subset to trials where the label is in {obj, altobj}
                sub_mask = behav["label"].isin([obj, altobj])
                Xs = X[sub_mask]
                ys = behav.loc[sub_mask, "label"].values
                runs_s = behav.loc[sub_mask, "run_num"].values
                # LOO accuracy
                run_accs = []
                for test_run in np.unique(runs_s):
                    tr = runs_s != test_run
                    te = runs_s == test_run
                    if (tr.sum() == 0) or (te.sum() == 0):
                        continue
                    if len(np.unique(ys[tr])) < 2:
                        continue
                    c = LogisticRegression(penalty="l2", C=1, solver="lbfgs",
                                           max_iter=1000)
                    c.fit(Xs[tr], ys[tr])
                    run_accs.append(c.score(Xs[te], ys[te]))
                # Production model fit on ALL data
                production = LogisticRegression(penalty="l2", C=1, solver="lbfgs",
                                                max_iter=1000)
                production.fit(Xs, ys)
                clf_dict[name] = production
                loo_acc[name] = float(np.mean(run_accs)) if run_accs else float("nan")

    return clf_dict, loo_acc, fourway_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub003")
    ap.add_argument("--ses", default="ses1")
    ap.add_argument("--runs", nargs="+", type=int, default=None,
                    help="Recognition runs to use; defaults to all available")
    args = ap.parse_args()

    # Item code map — Peng's convention
    item_to_label = {"A": "bed", "B": "chair", "C": "table", "D": "bench"}

    # chosenMask lives at ses1 only; reused for all sessions
    mask_path = (PENG_ROOT / args.sub / args.sub / "ses1"
                 / "recognition" / "mask" / "chosenMask.npy")
    mask_3d = np.load(mask_path)
    n_vox = int((mask_3d != 0).sum())
    print(f"[{args.sub}/{args.ses}] chosenMask: shape={mask_3d.shape}, "
          f"{n_vox} voxels")

    # Identify available runs
    rec_dir = PENG_ROOT / args.sub / args.sub / args.ses / "recognition"
    if args.runs:
        runs = args.runs
    else:
        runs = sorted([
            int(f.stem.split("_")[1]) for f in rec_dir.glob("run_*_bet.nii")
        ])
    print(f"[{args.sub}/{args.ses}] recognition runs: {runs}")

    # collect_session now returns per-trial features directly (after +2 TR
    # shift and z-score-across-trials within each run, matching Peng's order).
    print(f"[{args.sub}/{args.ses}] loading + extracting trials + z-scoring ...")
    X, behav = collect_session(args.sub, args.ses, runs, mask_3d,
                               item_to_label, hrf_delay_TRs=2)
    print(f"  trial-feature matrix: {X.shape}, n_trials={len(behav)}")

    # Class distribution
    print("  class counts:")
    for k, v in behav["label"].value_counts().items():
        print(f"    {k}: {v}")

    # Train classifiers
    clf_dict, loo_acc, fourway_mean = train_pairwise_clfs(X, behav)

    # Save
    out_dir = DERIV_ROOT / args.sub / "clf"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, clf in clf_dict.items():
        joblib.dump(clf, out_dir / f"{name}.joblib")
    summary = {
        "sub": args.sub,
        "ses": args.ses,
        "runs": runs,
        "n_voxels": n_vox,
        "n_trials": int(len(behav)),
        "fourway_loo_acc": fourway_mean,
        "pairwise_loo_acc": loo_acc,
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {len(clf_dict)} classifiers saved to {out_dir}")
    print(f"  summary: {out_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
