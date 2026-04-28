#!/usr/bin/env python3
"""Streaming-decode stability demo on sub-005 ses-03 run-01.

For each per-TR arrival, recomputes the per-trial probe beta using the
buffer-so-far via TWO methods:

    OLS         standard OLS GLM on the same Glover-HRF LSS design.
    Variant G   AR(1)-prewhitened Bayesian conjugate GLM.

Each method's β is fed through the SAME paper MindEye ridge → backbone →
CLIP and scored against the trial's GT image cosine similarity. We record
the decoded top-1 image probability and top-1 image rank at each TR within
each trial's feedback window.

Compares:
  - within-trial stability (CV of top-1 prob across TRs after onset+5s)
  - latency-to-correct (#TRs after onset before top-1 == GT)
  - cross-method paired bootstrap of those metrics

The point is to show — on the same already-working sub-005 pipeline used
in the bake-off — that Variant G's running posterior β is a smoother,
more confident classifier input than OLS. This is the methodological
claim Princeton CML cares about, decoupled from the Peng-data pipeline
gap that requires their FSL preprocessing chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

# torch is imported lazily — only the decode phase needs it
try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

# JAX is needed for the GLM phase (CPU venv); the decoder phase (docker container)
# only needs PyTorch. Optional-import so this script runs cleanly in both.
try:
    import jax.numpy as jnp
    from rt_glm_variants import (
        _variant_g_forward,
        _ols_fit,
        make_glover_hrf,
        build_design_matrix,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# PyTorch imports — only used in the decoder phase
try:
    from mindeye_retrieval_eval import (
        load_mindeye,
        cosine_sim_tokens,
    )
    TORCH_DECODER_AVAILABLE = True
except ImportError:
    TORCH_DECODER_AVAILABLE = False


PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
BRAIN_MASK_PATH = RT3T / "sub-005_final_mask.nii.gz"          # 19174-vox brain
RELMASK_PATH = RT3T / "sub-005_ses-01_task-C_relmask.npy"     # 2792-true rel
EVENTS_PATH = (RT3T / "events"
               / "sub-005_ses-03_task-C_run-01_events.tsv")
MC_DIR = Path("/data/3t/derivatives/motion_corrected_resampled")
GT_CACHE = PAPER_ROOT / "task_2_1_betas" / "gt_cache"
STIMULI = PAPER_ROOT / "rt3t" / "data" / "all_stimuli"
OUT_DIR = PAPER_ROOT / "task_2_1_betas" / "streaming_demo"


def load_paper_finalmask() -> tuple[np.ndarray, np.ndarray]:
    flat_brain = (nib.load(BRAIN_MASK_PATH).get_fdata() > 0).flatten()
    rel = np.load(RELMASK_PATH)
    assert flat_brain.sum() == 19174
    assert rel.shape == (19174,) and rel.dtype == bool and rel.sum() == 2792
    return flat_brain, rel


def stream_run01_volumes() -> list[np.ndarray]:
    """Sorted list of 3-D volumes for sub-005 ses-03 run-01.

    The mc_boldres files are int16-quantized 3-D NIfTI per frame.
    """
    files = sorted(MC_DIR.glob("ses-03_run-01_*_mc_boldres.nii.gz"))
    if not files:
        raise FileNotFoundError("no ses-03_run-01_*_mc_boldres frames on disk")
    print(f"  loaded {len(files)} mc_boldres frames")
    return [nib.load(f).get_fdata().astype(np.float32) for f in files]


def voxelwise_zscore_running(buffer: np.ndarray) -> np.ndarray:
    """Z-score the FULL buffer voxelwise (across TR axis 1).

    Real-time RT-cloud usage uses cumulative z-score; this matches the
    paper's per-TR running-z-score convention.
    """
    if buffer.shape[1] < 2:
        return buffer
    mu = buffer.mean(axis=1, keepdims=True)
    sd = buffer.std(axis=1, keepdims=True) + 1e-8
    return (buffer - mu) / sd


def lookup_gt(image_id: str) -> np.ndarray | None:
    if "blank" in image_id.lower():
        return None
    p = STIMULI / Path(image_id).relative_to("all_stimuli")
    if not p.exists():
        cands = list(STIMULI.rglob(Path(image_id).name))
        if not cands:
            return None
        p = cands[0]
    key = GT_CACHE / f"{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}.npy"
    return np.load(key) if key.exists() else None


def predict_clip_through_paper_decoder(beta_2792: np.ndarray, model, device: str
                                        ) -> np.ndarray:
    """Push a (V,)=2792 beta vector through ridge → backbone → clip_voxels."""
    b = torch.from_numpy(beta_2792.astype(np.float32)).to(device)
    b = b.unsqueeze(0).unsqueeze(1)               # (1, 1, V)
    with torch.no_grad(), torch.amp.autocast(
        "cuda" if device == "cuda" else "cpu", dtype=torch.float16
    ):
        latent = model.ridge(b, 0)
        bk_out = model.backbone(latent)
        clip_voxels = (bk_out[1] if isinstance(bk_out, tuple)
                       else bk_out).float().cpu().numpy()
    return clip_voxels[0]                          # (256, 1664)


def phase_glm(args):
    """CPU phase: precompute β_OLS and β_VG for every (trial, tr_delta) pair."""
    if not JAX_AVAILABLE:
        raise RuntimeError(
            "Phase 'glm' requires JAX (run from the host venv, not the docker "
            "container which has only PyTorch)."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_brain, rel = load_paper_finalmask()
    events = pd.read_csv(EVENTS_PATH, sep="\t")
    onsets_session = events["onset"].astype(float).values
    onsets = onsets_session - onsets_session[0]
    image_ids = events["image_name"].astype(str).tolist()
    tr = 1.5

    print("[glm 1] Loading + masking volumes")
    vols = stream_run01_volumes()
    masked = np.stack([v.flatten()[flat_brain][rel] for v in vols], axis=1)
    n_trs_total = masked.shape[1]
    print(f"  masked shape: {masked.shape}")

    # Trials we have GT for (resolved later in decoder phase, but here we
    # compute betas for ALL trials so the decoder can pick).
    print("[glm 2] Computing per-(trial, Δ) betas via OLS + Variant G")
    hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    n_pad = args.max_trs

    rows = []
    betas_ols, betas_vg, vars_vg = [], [], []
    trials_to_run = list(range(len(onsets)))
    if args.n_trials_cap:
        trials_to_run = trials_to_run[:args.n_trials_cap]
    for trial_i in trials_to_run:
        onset_tr = int(round(onsets[trial_i] / tr))
        for delta in range(args.feedback_window_trs + 1):
            cur_tr = onset_tr + delta
            if cur_tr >= n_trs_total:
                break
            n_trs_now = cur_tr + 1
            dm, probe_col = build_design_matrix(
                onsets, tr, n_trs_now, hrf, probe_trial=trial_i,
            )
            Y_now = voxelwise_zscore_running(masked[:, :n_trs_now])
            try:
                bols = _ols_fit(jnp.asarray(dm), jnp.asarray(Y_now))
                bols = np.asarray(bols[:, probe_col], dtype=np.float32)
            except Exception as e:
                print(f"    OLS fail trial {trial_i} Δ{delta}: {e}")
                continue
            dm_pad = np.zeros((n_pad, dm.shape[1]), dtype=np.float32)
            dm_pad[:n_trs_now] = dm
            Y_pad = np.zeros((Y_now.shape[0], n_pad), dtype=np.float32)
            Y_pad[:, :n_trs_now] = Y_now
            bvg, vvg = _variant_g_forward(
                jnp.asarray(dm_pad), jnp.asarray(Y_pad),
                jnp.asarray(n_trs_now, dtype=jnp.int32),
            )
            bvg = np.asarray(bvg[:, probe_col], dtype=np.float32)
            vvg = np.maximum(np.asarray(vvg[:, probe_col], dtype=np.float32),
                             1e-10)
            rows.append({
                "trial": trial_i,
                "tr_delta": delta,
                "onset_tr": onset_tr,
                "image_id": image_ids[trial_i],
            })
            betas_ols.append(bols)
            betas_vg.append(bvg)
            vars_vg.append(vvg)
        print(f"  trial {trial_i:>3}  onset_tr={onset_tr:>3}")

    df = pd.DataFrame(rows)
    np.save(OUT_DIR / "betas_ols.npy", np.stack(betas_ols, axis=0))
    np.save(OUT_DIR / "betas_vg.npy", np.stack(betas_vg, axis=0))
    np.save(OUT_DIR / "vars_vg.npy", np.stack(vars_vg, axis=0))
    df.to_csv(OUT_DIR / "betas_index.csv", index=False)
    print(f"\n  wrote {OUT_DIR}/betas_*.npy + betas_index.csv ({len(df)} rows)")


def phase_decode(args):
    """GPU phase: load pre-computed betas, push through MindEye decoder, score."""
    if not TORCH_DECODER_AVAILABLE:
        raise RuntimeError("Phase 'decode' requires the MindEye PyTorch stack.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ---- Load pre-computed betas (from phase 'glm') ----
    print("[1] Loading pre-computed betas")
    df = pd.read_csv(OUT_DIR / "betas_index.csv")
    betas_ols_arr = np.load(OUT_DIR / "betas_ols.npy")
    betas_vg_arr = np.load(OUT_DIR / "betas_vg.npy")
    print(f"  {len(df)} (trial, Δ) pairs from {df.trial.nunique()} trials")

    # ---- Load decoder ----
    print("[2] Loading paper MindEye checkpoint")
    model, ss, se = load_mindeye(Path(args.checkpoint), n_voxels=2792, device=device)
    image_ids = df["image_id"].astype(str).tolist()

    # ---- Pre-cache GT embeddings for trials we have stimuli for ----
    print("[3] Resolving GT embeddings")
    # df rows are (trial, tr_delta); group by trial for GT lookup
    unique_trials = df["trial"].unique().tolist()
    trial_gt = {}
    trial_image = {}
    for t in unique_trials:
        img = df[df.trial == t].iloc[0]["image_id"]
        trial_image[t] = img
        gt = lookup_gt(img)
        if gt is not None:
            trial_gt[t] = gt
    print(f"  {len(trial_gt)}/{len(unique_trials)} trials have cached GT")

    # ---- Build a candidate set: all unique GT-resolved images ----
    candidate_imgs = []
    candidate_emb = []
    seen = set()
    for t in unique_trials:
        if t not in trial_gt:
            continue
        img = trial_image[t]
        if img in seen:
            continue
        seen.add(img)
        candidate_imgs.append(img)
        candidate_emb.append(trial_gt[t])
    candidate_emb = np.stack(candidate_emb, axis=0)
    img_to_idx = {img: i for i, img in enumerate(candidate_imgs)}
    print(f"  {len(candidate_imgs)} unique candidate images")

    # ---- Iterate (trial, Δ) rows, decode each via OLS and VG ----
    print("[4] Decoding pre-computed betas: OLS vs Variant G")
    records = []
    for row_idx, row in df.iterrows():
        trial_i = int(row["trial"])
        if trial_i not in trial_gt:
            continue
        gt_idx = img_to_idx[trial_image[trial_i]]
        for method, beta_arr in (("ols", betas_ols_arr), ("vg", betas_vg_arr)):
            beta = beta_arr[row_idx]
            pred = predict_clip_through_paper_decoder(beta, model, device)
            sim = cosine_sim_tokens(pred[None, :, :], candidate_emb)[0]
            exp = np.exp(sim - sim.max())
            probs = exp / exp.sum()
            top1 = int(np.argmax(probs))
            rank = int(1 + (np.argsort(-probs) == gt_idx).argmax())
            records.append({
                "trial": trial_i,
                "tr_delta": int(row["tr_delta"]),
                "onset_tr": int(row["onset_tr"]),
                "method": method,
                "image_id": row["image_id"],
                "gt_idx": gt_idx,
                "top1_idx": top1,
                "top1_correct": int(top1 == gt_idx),
                "gt_rank": rank,
                "gt_prob": float(probs[gt_idx]),
                "top1_prob": float(probs[top1]),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "streaming_decode_per_tr.csv", index=False)
    print(f"\n  wrote {OUT_DIR / 'streaming_decode_per_tr.csv'} ({len(df)} rows)")

    # ---- Stability metrics ----
    print("[5] Stability metrics")
    summary = {}
    for method in ("ols", "vg"):
        m = df[df.method == method]
        # Within-trial std of gt_prob over the feedback window (delta >= 3, ie
        # past the HRF rise — only consider TRs where the signal could have
        # peaked).
        per_trial_std = m[m.tr_delta >= 3].groupby("trial")["gt_prob"].std()
        per_trial_cv = (per_trial_std
                        / (m[m.tr_delta >= 3].groupby("trial")["gt_prob"].mean() + 1e-8))
        # Latency: smallest tr_delta where top1_correct == 1, else max+1
        def latency(group):
            hits = group.sort_values("tr_delta")
            for _, row in hits.iterrows():
                if row["top1_correct"]:
                    return int(row["tr_delta"])
            return int(group["tr_delta"].max() + 1)
        per_trial_latency = m.groupby("trial").apply(latency, include_groups=False)
        summary[method] = {
            "n_trials": int(m.trial.nunique()),
            "gt_prob_within_trial_std_mean": float(per_trial_std.mean()),
            "gt_prob_within_trial_cv_mean": float(per_trial_cv.mean()),
            "latency_to_correct_mean": float(per_trial_latency.mean()),
            "latency_to_correct_median": float(per_trial_latency.median()),
            "final_tr_top1_acc": float(
                m[m.tr_delta == m.tr_delta.max()]["top1_correct"].mean()
            ),
        }

    # Paired bootstrap of within-trial std difference
    trials = sorted(df.trial.unique())
    diffs = []
    for t in trials:
        for col in ("gt_prob",):
            v_ols = df[(df.trial == t) & (df.method == "ols") & (df.tr_delta >= 3)
                       ][col].std()
            v_vg = df[(df.trial == t) & (df.method == "vg") & (df.tr_delta >= 3)
                      ][col].std()
            if not (np.isnan(v_ols) or np.isnan(v_vg)):
                diffs.append(v_vg - v_ols)
    diffs = np.asarray(diffs, dtype=np.float32)
    rng = np.random.default_rng(0)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                     for _ in range(2000)])
    summary["paired_diff_within_trial_std_vg_minus_ols"] = {
        "mean": float(diffs.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_diff_ge_0": float((boot >= 0).mean()),
        "n_trials": int(len(diffs)),
    }

    print("\n=== Stability summary ===")
    print(json.dumps(summary, indent=2))
    with open(OUT_DIR / "stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {OUT_DIR / 'stability_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("glm", "decode", "all"),
                    default=("glm" if JAX_AVAILABLE and not TORCH_DECODER_AVAILABLE
                             else "decode" if TORCH_DECODER_AVAILABLE and not JAX_AVAILABLE
                             else "all"),
                    help="'glm' (CPU venv): compute betas from raw BOLD. "
                         "'decode' (GPU container): score betas through MindEye. "
                         "'all': both (only works if both stacks available).")
    ap.add_argument("--max-trs", type=int, default=200,
                    help="Variant G pad-to length (static JIT shape)")
    ap.add_argument("--feedback-window-trs", type=int, default=10,
                    help="How many TRs after each onset to track decoded prob")
    ap.add_argument("--n-trials-cap", type=int, default=None,
                    help="Cap trials processed (debug)")
    ap.add_argument("--checkpoint",
                    default=str(PAPER_ROOT / "checkpoints"
                                / "data_scaling_exp/concat_glmsingle/checkpoints"
                                / "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_"
                                  "avgrepeats_finalmask_epochs_150.pth"))
    args = ap.parse_args()

    print(f"[phase] {args.phase}  (jax={JAX_AVAILABLE}, torch_decoder={TORCH_DECODER_AVAILABLE})")
    if args.phase in ("glm", "all"):
        phase_glm(args)
    if args.phase in ("decode", "all"):
        phase_decode(args)


if __name__ == "__main__":
    main()
