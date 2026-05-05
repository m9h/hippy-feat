#!/usr/bin/env python3
"""Two related diagnostics for the Offline-3T-first-rep gap:

1) GLMsingle stage ablation. From a glmsingle_persistent/{rtmotion,fmriprep}/
   directory loads the 4 progressive-stage outputs:
     - TYPEA_ONOFF.npy            (assumes-canonical HRF, no library, no denoise, no fracridge)
     - TYPEB_FITHRF.npy           (Stage 1: HRF library)
     - TYPEC_FITHRF_GLMDENOISE.npy(Stages 1+2: + GLMdenoise)
     - TYPED_FITHRF_GLMDENOISE_RR.np?(Stages 1+2+3: + fracridge — canonical)
   Projects to MindEye 2792 voxels, applies causal cum-z, saves prereg cells:
     {bold}_TYPEA_partial / _full
     {bold}_TYPEB_partial / _full
     {bold}_TYPEC_partial / _full
     {bold}_TYPED_partial / _full   (full == repeat-avg)

2) k-rep average diagnostic on canonical Princeton TYPED βs (ses-03):
     k=1 (first-rep), k=2 (avg of reps 1+2), k=3 (avg of all 3 — = full row).
   Tests whether paper Table 1's "Offline 3T (single first-rep) = 76%" is
   actually a 2-rep average (≈ √2 noise reduction would land ~73-76%).

Outputs are saved to the standard prereg/ dir so prereg_retrieval_pass.py
picks them up via its existing --cells argument.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
EVENTS = RT3T / "events"
GS_PERSIST_ROOT = PAPER_ROOT / "task_2_1_betas" / "glmsingle_persistent"
GS_CANON_ROOT = PAPER_ROOT / "glmsingle"
PREREG = PAPER_ROOT / "task_2_1_betas" / "prereg"
SESSION = "ses-03"


def load_finalmask_canonical_idx() -> np.ndarray:
    """Indices into canonical-brain-flat (length 183408) → 2792 MindEye voxels."""
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    canon_brain = nib.load(
        GS_CANON_ROOT / f"glmsingle_sub-005_{SESSION}_task-C"
                     / f"sub-005_{SESSION}_task-C_brain.nii.gz"
    ).get_fdata() > 0
    me_positions = np.where(final_mask.flatten())[0][relmask]
    canon_flat_to_brain = -np.ones(canon_brain.size, dtype=np.int64)
    canon_flat_to_brain[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_flat_to_brain[me_positions]
    assert (me_in_canon >= 0).all()
    return me_in_canon


def load_finalmask_3d_idx() -> tuple[np.ndarray, np.ndarray]:
    """For TYPEA/B/C (which are saved in T1w-3D space, not canonical brain
    space): return flat 3D index → 2792 MindEye voxels."""
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    me_positions = np.where(final_mask.flatten())[0][relmask]            # (2792,)
    return me_positions, final_mask.shape


def reconstruct_trial_ids() -> np.ndarray:
    out: list[str] = []
    for run in range(1, 12):
        e = pd.read_csv(EVENTS / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                          sep="\t")
        e = e[e["image_name"] != "blank.jpg"]
        out += e["image_name"].astype(str).tolist()
    return np.asarray(out)


def causal_cumulative_zscore(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[0]
    for i in range(n):
        if i == 0:
            out[i] = arr[i]
        elif i == 1:
            out[i] = arr[i] - arr[0]
        else:
            mu = arr[:i].mean(axis=0)
            sd = arr[:i].std(axis=0) + 1e-6
            out[i] = (arr[i] - mu) / sd
    return out


def repeat_avg(betas: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen: dict[str, list[int]] = {}
    for i, img in enumerate(ids):
        seen.setdefault(img, []).append(i)
    out_betas, out_ids = [], []
    for img, idxs in seen.items():
        out_betas.append(betas[idxs].mean(axis=0))
        out_ids.append(img)
    return np.stack(out_betas), np.asarray(out_ids)


def k_rep_avg(betas: np.ndarray, ids: np.ndarray, k: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """For each unique image with ≥ k reps, average the FIRST k reps."""
    seen: dict[str, list[int]] = {}
    for i, img in enumerate(ids):
        seen.setdefault(img, []).append(i)
    out_betas, out_ids = [], []
    for img, idxs in seen.items():
        if len(idxs) < k:
            continue
        out_betas.append(betas[idxs[:k]].mean(axis=0))
        out_ids.append(img)
    return np.stack(out_betas), np.asarray(out_ids)


def load_typed_array(path: Path) -> np.ndarray:
    """Load a TYPE{A,B,C,D} GLMsingle output. Returns (V, T) — flat-voxel × trial."""
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        arr = z["betasmd"]
    else:
        z = np.load(path, allow_pickle=True).item()
        arr = z["betasmd"]
    arr = np.asarray(arr).squeeze().astype(np.float32)
    # Reshape (X, Y, Z, T) → (V, T) if needed
    if arr.ndim == 4:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr                                                            # (V_flat, T)


def project_typed_to_2792(arr_flat_T: np.ndarray, source: str) -> np.ndarray:
    """Project (V_flat, T) → (T, 2792). 'persistent' uses 3D-flat; 'canonical' uses canonical-brain."""
    if source == "persistent":
        me_idx, _ = load_finalmask_3d_idx()
    elif source == "canonical":
        me_idx = load_finalmask_canonical_idx()
    else:
        raise ValueError(source)
    return arr_flat_T[me_idx, :].T.astype(np.float32)                     # (T, 2792)


def save_cell(name: str, betas: np.ndarray, ids: np.ndarray, cfg: dict) -> None:
    PREREG.mkdir(parents=True, exist_ok=True)
    np.save(PREREG / f"{name}_{SESSION}_betas.npy", betas)
    np.save(PREREG / f"{name}_{SESSION}_trial_ids.npy", ids)
    import json
    with open(PREREG / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: betas {betas.shape}  ids {ids.shape}")


def stage_ablation(bold: str) -> None:
    """Score TYPEA, TYPEB, TYPEC, TYPED for one BOLD source."""
    print(f"\n=== STAGE ABLATION ({bold}) ===")
    bold_dir = GS_PERSIST_ROOT / bold
    type_files = {
        "TYPEA": bold_dir / "TYPEA_ONOFF.npy",
        "TYPEB": bold_dir / "TYPEB_FITHRF.npy",
        "TYPEC": bold_dir / "TYPEC_FITHRF_GLMDENOISE.npy",
        "TYPED": bold_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy",
    }
    trial_ids = reconstruct_trial_ids()
    for tag, p in type_files.items():
        if not p.exists():
            alt = p.with_suffix(".npz")
            if alt.exists():
                p = alt
            else:
                print(f"  SKIP {tag}: {p} missing")
                continue
        try:
            arr = load_typed_array(p)
        except Exception as e:
            print(f"  SKIP {tag}: load failed {e}")
            continue
        print(f"  {tag} loaded: {arr.shape}")
        if arr.shape[0] < 100000 or arr.shape[1] != len(trial_ids):
            # TYPEA may have masked output (V_brain, T) instead of flat-3D
            if arr.shape[1] == len(trial_ids):
                # Probably already brain-only — try canonical projection instead
                # (TYPEA from a brain-masked GLMsingle is V_brain × T)
                print(f"    (V × T) shape with V={arr.shape[0]} ≠ flat3D — assuming "
                      f"V_brain ordering; cannot project, skipping.")
                continue
            print(f"    shape mismatch with {len(trial_ids)} trials, skipping.")
            continue
        # Project flat 3D → 2792
        betas_T = project_typed_to_2792(arr, "persistent")                 # (T, 2792)
        del arr
        # Cumulative z-score (causal)
        betas_z = causal_cumulative_zscore(betas_T)
        # Cell A: partial (first-rep filter happens at retrieval pass)
        save_cell(f"GLMsingle_{bold}_{tag}_partial", betas_z, trial_ids,
                   {"bold": bold, "stage": tag, "do_repeat_avg": False})
        # Cell B: full (repeat-avg post cum-z)
        ra_betas, ra_ids = repeat_avg(betas_z, trial_ids)
        save_cell(f"GLMsingle_{bold}_{tag}_full", ra_betas, ra_ids,
                   {"bold": bold, "stage": tag, "do_repeat_avg": True})


def k_rep_diagnostic() -> None:
    """k-rep average diagnostic on canonical Princeton TYPED βs."""
    print(f"\n=== k-REP AVERAGE DIAGNOSTIC (canonical TYPED) ===")
    canon_path = (GS_CANON_ROOT / f"glmsingle_sub-005_{SESSION}_task-C"
                                / "TYPED_FITHRF_GLMDENOISE_RR.npz")
    arr = load_typed_array(canon_path)                                    # (V_canon, T)
    betas_T = project_typed_to_2792(arr, "canonical")                     # (T, 2792)
    del arr
    trial_ids = reconstruct_trial_ids()
    assert betas_T.shape[0] == len(trial_ids), \
        f"trial mismatch: {betas_T.shape[0]} vs {len(trial_ids)}"
    betas_z = causal_cumulative_zscore(betas_T)
    # Filter to special515 only — that's the only set with multiple reps
    mask = np.array([t.startswith("all_stimuli/special515/") for t in trial_ids])
    betas_s = betas_z[mask]
    ids_s = trial_ids[mask]
    print(f"  special515 trials: {betas_s.shape[0]}  (50 unique × 3 reps)")
    for k in [1, 2, 3]:
        ka_betas, ka_ids = k_rep_avg(betas_s, ids_s, k)
        cell = f"Canonical_GLMsingle_kavg_{k}rep"
        save_cell(cell, ka_betas, ka_ids,
                   {"k_rep_avg": k, "source": "canonical_TYPED",
                    "n_trials": int(ka_betas.shape[0])})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold", choices=["rtmotion", "fmriprep", "both", "none"],
                     default="both",
                     help="which BOLD source's stage ablation to import")
    ap.add_argument("--krep", action="store_true", default=True,
                     help="also run the k-rep diagnostic on canonical TYPED βs")
    ap.add_argument("--no-krep", action="store_false", dest="krep")
    args = ap.parse_args()

    bold_sources = {"both": ["rtmotion", "fmriprep"],
                     "rtmotion": ["rtmotion"],
                     "fmriprep": ["fmriprep"],
                     "none": []}[args.bold]
    for b in bold_sources:
        if (GS_PERSIST_ROOT / b).exists():
            stage_ablation(b)
        else:
            print(f"\n=== STAGE ABLATION ({b}) === SKIPPED (no glmsingle_persistent/{b})")

    if args.krep:
        k_rep_diagnostic()


if __name__ == "__main__":
    main()
