import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.mindeye_retrieval_eval import (
    MindEyeModule,
    RidgeRegression,
    load_mindeye,
    cosine_sim_tokens,
    load_condition_betas,
    filter_to_special515,
    compute_ground_truth_embeddings,
    predict_clip,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["prereg/LogSig_AR1freq_glover_rtm", "prereg/Offline_paper_replica_full"])
    ap.add_argument("--session", default="ses-03")
    ap.add_argument(
        "--checkpoint",
        default="/data/derivatives/rtmindeye_paper/checkpoints/"
                "data_scaling_exp/concat_glmsingle/checkpoints/"
                "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_"
                "avgrepeats_finalmask_epochs_150.pth",
    )
    ap.add_argument(
        "--stimuli-dir",
        default="/data/derivatives/rtmindeye_paper/rt3t/data/all_stimuli/special515",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Load MindEye checkpoint once
    print(f"\n[1/4] Loading MindEye checkpoint (Fold-10)")
    model, ss, se = load_mindeye(Path(args.checkpoint), n_voxels=2792, device=device)

    # Compute/Load GT once (we know ses-03 uses special515)
    print(f"\n[2/4] Building image index + GT embeddings")
    # We need a sample beta to get unique images
    sample_betas, sample_ids = load_condition_betas(args.conditions[0], args.session)
    _, ids_test, unique_images = filter_to_special515(sample_betas, sample_ids)
    image_paths = [Path(args.stimuli_dir) / Path(n).name for n in unique_images]
    gt_emb = compute_ground_truth_embeddings(image_paths, device=device)
    print(f"  gt shape: {gt_emb.shape}")

    img_to_idx = {str(u): i for i, u in enumerate(unique_images)}

    final_results = {}

    for condition in args.conditions:
        print(f"\n[3/4] Processing {condition}...")
        betas_all, trial_ids_all = load_condition_betas(condition, args.session)
        betas_test, ids_test, _ = filter_to_special515(betas_all, trial_ids_all)
        trial_idx = np.array([img_to_idx[t] for t in ids_test])

        pred_emb = predict_clip(model, betas_test, device=device,
                                clip_seq_dim=ss, clip_emb_dim=se)
        
        sim = cosine_sim_tokens(pred_emb, gt_emb)  # (N_trials, N_images)
        topk = np.argsort(-sim, axis=1)

        # [A] All-trials (n=150)
        hits1_all = topk[:, 0] == trial_idx
        hits5_all = np.array([trial_idx[i] in topk[i, :5] for i in range(len(sim))])
        
        # [B] First-repetition only (n=50)
        first_rep_idx = []
        seen = set()
        for i, img in enumerate(ids_test):
            if img not in seen:
                first_rep_idx.append(i)
                seen.add(img)
        first_rep_idx = np.array(first_rep_idx)
        
        hits1_first = hits1_all[first_rep_idx]
        hits5_first = hits5_all[first_rep_idx]

        # [C] Brain retrieval
        brain_sim = sim.T
        brain_hits = []
        for img_idx in range(len(unique_images)):
            ranked_trials = np.argsort(-brain_sim[img_idx])
            top1_trial = ranked_trials[0]
            brain_hits.append(trial_idx[top1_trial] == img_idx)
        top1_brain = float(np.mean(brain_hits))

        res = {
            "top1_image_all": float(hits1_all.mean()),
            "top1_image_first_rep": float(hits1_first.mean()),
            "top5_image_all": float(hits5_all.mean()),
            "top5_image_first_rep": float(hits5_first.mean()),
            "top1_brain": top1_brain,
        }
        final_results[condition] = res
        print(f"  {condition} results:")
        print(json.dumps(res, indent=2))

    # Save summary
    out = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/fold10_repro_summary.json")
    with open(out, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
