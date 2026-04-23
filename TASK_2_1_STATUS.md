# Task 2.1 — fMRIPrep vs GLMsingle contributions to RT gap

**Assigned to Morgan Hough** per Rishab Iyer's Discord post 2026-03-06.
Goal: decompose the 10 pp image-retrieval gap between paper Offline 3T
(76 %) and End-of-run RT (66 %) into a fMRIPrep contribution and a
GLMsingle contribution.

**Target audience for this doc:** me tomorrow morning / remote contributors
who want to resume the work without re-deriving state.

---

## Current state

### What's on disk (all produced by sbatch jobs listed below)

```
/data/derivatives/rtmindeye_paper/
├── fmriprep_mindeye/          # ses-01/02/03/06 desc-preproc_bold.nii.gz (38 runs)
├── rt3t/data/                 # model/, events/, special515 stimuli (515 imgs),
│                              # sub-005_final_mask.nii.gz, relmask.npy, HRF libs
├── checkpoints/data_scaling_exp/concat_glmsingle/checkpoints/
│   └── sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_...finalmask_epochs_150.pth   ← PAPER CANONICAL
├── repos/rtcloud-projects/mindeye/   # canonical model.py etc.
└── task_2_1_betas/
    ├── A_fmriprep_glover_ses-03_betas.npy       (770, 2792) float32
    └── A_fmriprep_glover_ses-03_trial_ids.npy   (770,) string image_name
```

### Mask clarification (important)

The file named `sub-005_final_mask.nii.gz` is **misleading** — it's actually
the **brain mask at 19174 voxels**, not the paper's finalmask.

The paper's **finalmask** = brain mask **∩** reliability mask:
1. Load `sub-005_final_mask.nii.gz` → 19174 brain voxels
2. Load `sub-005_ses-01_task-C_relmask.npy` → `(19174,)` bool, 2792 True
3. Sequential apply: `vol.flatten()[brain_mask][relmask]` → **2792 voxels**

`scripts/task_2_1_factorial.py:load_paper_finalmask()` encodes this.

### What's running or about to run

| Job | What | Status | ETA |
|---|---|---|---|
| 893 | Pull `nvcr.io/nvidia/pytorch:26.03-py3` → `/data/derivatives/containers/pytorch_26.03.sif` | Running cpu, 64 GB | ~30 min |
| 896 | `overnight-dlbs` (user's separate job) | Queued gpu | 14 h |
| 897 | `mindeye_retrieval_eval` (Task 2.1 condition A retrieval) | Queued gpu behind 896 | Starts ~09:00 Thu |

---

## What each script does

| Script | Purpose |
|---|---|
| `scripts/download_fmriprep_mindeye.sbatch` | Pull `rishab-iyer1/fmriprep_mindeye` from HF (fMRIPrep'd sub-005 ses-01/02/03/06, 38 runs, ~11 GB) |
| `scripts/download_rt3t_selective.sbatch` | Pull `rishab-iyer1/3t` selectively (model/, events/, special515/, masks; skips 20k general stimuli) |
| `scripts/download_paper_checkpoint.sbatch` | Pull 19 `.pth` checkpoints from `macandro96/mindeye_offline_ckpts` (bulk, ~150 GB, 2h limit → gets samples 1-9) |
| `scripts/download_paper_checkpoint_sample10.sbatch` | Targeted pull of the one canonical checkpoint missed by the bulk job |
| `scripts/download_special515_stimuli.sbatch` | Pull the 515 special515 test images from rt3t |
| `scripts/clone_rtcloud_mindeye.sbatch` | Git clone `brainiak/rtcloud-projects` for the canonical `models.py` |
| `scripts/pull_pytorch_ngc.sbatch` | Apptainer pull NGC PyTorch 26.03 arm64 SIF (needs **64 GB** memory for mksquashfs) |
| `scripts/task_2_1_factorial.py` / `.sbatch` | Produce per-trial betas under each factorial condition. Currently condition A only; condition B requires RT motion correction. |
| `scripts/mindeye_retrieval_eval.py` / `.sbatch` | Load paper checkpoint, forward betas → predicted CLIP embeddings, compare against OpenCLIP GT, report top-k image/brain retrieval. |

## What's still to do (in priority order)

### 1. Get condition A's retrieval number (auto, overnight)

Job 897 queued. Log at `/data/derivatives/rtmindeye_paper/logs/mindeye-retrieval-eval-<jobid>.out`.
Writes `/data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_A_fmriprep_glover.json`
with fields: `top1_image_retrieval`, `top5_image_retrieval`, `top1_brain_retrieval`.

Compare against paper Table 1:
- End-of-run RT: image 66 %, brain 62 %
- Offline 3T:   image 76 %, brain 64 %
- **Our A** (fMRIPrep motion + Glover): somewhere between — the delta from 66 % is the **fMRIPrep contribution**.

### 2. Add condition B (RT motion + GLMsingle) — blocked until someone writes RT motion

Two paths:

**(a) FSL MCFLIRT** — simplest, native arm64 installed at `/home/mhough/fsl/bin/mcflirt`. Run on raw BIDS BOLD to produce `mc_boldres` files, then feed into `task_2_1_factorial.py` with `B_rtmotion_glmsingle` condition enabled. Matches paper's RT pipeline exactly.

**(b) Custom jaxoccoli GN-MC** — faster on GPU, but needs ~200 LOC to compose HMC + BOLD→boldref + resample. See earlier scoping in memory `project_preprocessing_strategy.md`.

Recommend (a) for Thursday. Write a `scripts/mcflirt_ses03.sbatch` that iterates raw BIDS BOLD for each run, writes per-TR `ses-03_run-XX_NNNN_mc_boldres.nii.gz` into `/data/3t/derivatives/motion_corrected_resampled/`. Then re-submit `task_2_1_factorial.sbatch` with `--conditions A_fmriprep_glover B_rtmotion_glmsingle` and re-run `mindeye_retrieval_eval.sbatch` for both.

### 3. Generate the decomposition report

Once A + B numbers land:
```
fMRIPrep contribution  = retrieval(A) - retrieval(End-of-run RT = 66 %)
GLMsingle contribution = retrieval(B) - retrieval(End-of-run RT = 66 %)
Interaction            = retrieval(Offline 3T = 76 %)
                       - retrieval(End-of-run RT)
                       - fMRIPrep contrib
                       - GLMsingle contrib
```

## How to resume from scratch

```bash
cd /home/mhough/dev/hippy-feat
squeue -u $USER                    # what's running
ls /data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_*.json  # what results landed
```

If job 897 completed overnight:
```bash
cat /data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_A_fmriprep_glover.json
```

If 897 failed, check:
```bash
tail -100 /data/derivatives/rtmindeye_paper/logs/mindeye-retrieval-eval-*.out
# Most likely: model state_dict key mismatch (inline MindEye2 class defs
# in mindeye_retrieval_eval.py may need small alignment with paper checkpoint).
# See the `missing keys` / `unexpected keys` report printed by load_mindeye().
```

## Known risks / open questions

1. **Checkpoint state_dict alignment.** `mindeye_retrieval_eval.py` defines `RidgeRegression` and `MindEyeModule` inline (matching the rtcloud-projects inline classes to avoid importing SDXL deps). Parameter names may differ slightly from the paper's checkpoint. The script uses `strict=False` load and prints missing/unexpected keys for debugging.
2. **Condensed-model hyperparameters.** `hidden_dim=1024, clip_seq_dim=256, clip_emb_dim=1664`, `blurry_recon=False`, `n_blocks=4`. These are educated guesses from the paper text (Section 2.5). If the checkpoint disagrees, fix the values in `load_mindeye()`.
3. **Z-scoring convention.** Paper z-scores cumulatively during RT, across training images for Offline. Our condition A betas are **raw** (not z-scored) from the GLM fit. The ridge was trained on z-scored betas, so for a fair retrieval number we should z-score per-voxel across the 770 trials before passing to the ridge. **Not yet done** — may need to add in `mindeye_retrieval_eval.py` before the forward pass.
4. **Test-set filtering.** Script filters to `all_stimuli/special515/*` which should yield 150 trials (50 images × 3 repeats). Verify `n_test_trials` in the output JSON matches 150.

## Related infrastructure

- **Raramuri arm64 SIF** at `/data/derivatives/containers/raramuri-arm64.sif` (built from source, 8 GB, native GB10). Not directly relevant to Task 2.1; available for TRIBEv2-accelerated video→BOLD experiments (Boris's task).
- **FreeSurfer 8.2 arm64 deb** at `/home/mhough/dev/debian/freesurfer/freesurfer_8.2.0-1_arm64.deb`. Unblocks a future arm64 fMRIPrep build. Not needed for Task 2.1 since fMRIPrep'd data is already on HF.
- **Variant G** (`scripts/rt_glm_variants.py`, `VariantG_Bayesian`). Separate research track (AR(1) conjugate GLM with posterior variance). Not part of Task 2.1 but useful for the June-July neurofeedback pilot.
