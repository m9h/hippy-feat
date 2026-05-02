#!/usr/bin/env python3
"""Forensic: how does z-score policy × rep-avg interact for the canonical
Offline anchor? Tests the four combinations:

  cum-z       + first-rep  (n=50)   — our current scoring
  cum-z       + rep-avg    (n=50)
  session-z   + first-rep  (n=50)
  session-z   + rep-avg    (n=50)   — Mac's protocol (got 76% top-1)
  no z        + first-rep  (n=50)
  no z        + rep-avg    (n=50)

Saves these as separate cells so the existing retrieval pass can score
them all in one run.

Paper reports Offline 3T = 76% top-1 (single-trial first-rep, n=50).
Goal: identify which policy combination matches that.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib

GS = Path("/data/derivatives/rtmindeye_paper/glmsingle/glmsingle_sub-005_ses-03_task-C")
RT3T = Path("/data/derivatives/rtmindeye_paper/rt3t/data")
EVENTS = RT3T / "events"
PREREG = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")


def load_canonical_to_relmask() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(GS / "TYPED_FITHRF_GLMDENOISE_RR.npz", allow_pickle=True)
    betas_full = z["betasmd"].squeeze().astype(np.float32)            # (V_canon, T)
    canon_brain = nib.load(GS / "sub-005_ses-03_task-C_brain.nii.gz").get_fdata() > 0
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    me_positions = np.where(final_mask.flatten())[0][relmask]
    canon_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_brain_idx[me_positions]
    betas_me = betas_full[me_in_canon, :]                              # (2792, T)
    out: list[str] = []
    for run in range(1, 12):
        e = pd.read_csv(EVENTS / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv",
                         sep="\t")
        e = e[e["image_name"] != "blank.jpg"]
        out += e["image_name"].astype(str).tolist()
    return betas_me.T.astype(np.float32), np.asarray(out)              # (T, 2792)


def causal_cumulative_zscore(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[0]
    for i in range(n):
        if i < 2:
            mu = arr[:max(i, 1)].mean(0, keepdims=True) if i > 0 else 0.0
            sd = 1.0
        else:
            mu = arr[:i].mean(0, keepdims=True)
            sd = arr[:i].std(0, keepdims=True) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out.astype(np.float32)


def session_zscore(arr: np.ndarray) -> np.ndarray:
    """Non-causal: uses statistics from ALL trials (Mac's protocol)."""
    mu = arr.mean(0, keepdims=True)
    sd = arr.std(0, keepdims=True) + 1e-6
    return ((arr - mu) / sd).astype(np.float32)


def first_rep(betas: np.ndarray, ids: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    seen = set()
    keep = np.zeros(len(ids), dtype=bool)
    for i, t in enumerate(ids):
        if t.startswith("all_stimuli/special515/") and t not in seen:
            keep[i] = True
            seen.add(t)
    return betas[keep], ids[keep]


def rep_avg(betas: np.ndarray, ids: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    spec_mask = np.array([t.startswith("all_stimuli/special515/") for t in ids])
    sb = betas[spec_mask]
    si = ids[spec_mask]
    unique = sorted(set(si))
    out = []
    for img in unique:
        idxs = [i for i, t in enumerate(si) if t == img]
        out.append(sb[idxs].mean(0))
    return np.stack(out, 0).astype(np.float32), np.asarray(unique)


def save_cell(name: str, betas: np.ndarray, ids: np.ndarray):
    np.save(PREREG / f"{name}_ses-03_betas.npy", betas)
    np.save(PREREG / f"{name}_ses-03_trial_ids.npy", ids)
    print(f"  saved {name}: {betas.shape} ids={len(ids)}")


def main():
    PREREG.mkdir(parents=True, exist_ok=True)
    raw, ids = load_canonical_to_relmask()
    print(f"raw canonical: {raw.shape}  ids: {len(ids)}")

    cumz  = causal_cumulative_zscore(raw)
    sessz = session_zscore(raw)

    # 6 combos × scoring policy. Save with names that make the retrieval
    # pass treat them as raw (skip its own z, since we've handled it).
    for tag, arr in [("cumz", cumz), ("sessz", sessz), ("noz", raw)]:
        # First-rep (n=50)
        b_fr, i_fr = first_rep(arr, ids)
        save_cell(f"Probe_canonical_{tag}_firstrep", b_fr, i_fr)
        # Rep-avg (n=50)
        b_ra, i_ra = rep_avg(arr, ids)
        save_cell(f"Probe_canonical_{tag}_repavg", b_ra, i_ra)


if __name__ == "__main__":
    main()
