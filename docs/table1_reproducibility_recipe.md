# Iyer et al. ICML 2026 — Table 1 Reproducibility Recipe

A self-contained description of what each Table 1 row is comparing and
how to reproduce it. Aimed at a reader who has access to the standard
artifacts but no prior context on the project.

## What's being compared

All Table 1 rows hold these constant:

- **Subject**: sub-005 (a 3T author of the study, fine-tune participant)
- **Test session**: ses-03 (held-out from training)
- **Test set**: 50 unique "special515" images from ses-03, each presented 3 times = 150 trial occurrences. (These 50 images are disjoint from ses-01/02 special515 — confirmed by inspection of events.tsv.)
- **Default eval**: single-trial first-rep — score the **first** of each image's 3 reps (n=50). The "(avg. 3 reps.)" rows instead average each image's 3 βs into one (also n=50).
- **Decoder**: MindEye2 (condensed: shared-subject latent dim 1024, omitting low-level submodule + img2img refinement + caption refinement). Fine-tuned on **ses-01** training betas (single session, ~1 hour, paper §2.6).
- **Retrieval metric**: 50-way top-1 image retrieval (chance = 2%), deterministic — predicted CLIP embedding (projector head output) → cosine sim → argmax against 50 candidate GT image embeddings.
- **Reconstruction metrics** (PixCorr, SSIM, AlexNet 2/5, Inception, CLIP, EfficientNet, SwAV): each averaged over **5 random diffusion-prior seeds** per trial.

What **varies between rows**:

| Row | BOLD source | GLM | Per-trial window | Repeat handling |
|---|---|---|---|---|
| Offline NSD (avg 3 reps) | NSD subj01 7T (separate dataset) | GLMsingle Stages 1+2+3, full session | full-run | avg of 3 reps |
| Offline NSD | same | same | full-run | first-rep only |
| **Offline 3T (avg 3 reps)** | sub-005 fmriprep | GLMsingle Stages 1+2+3, full session | full-run | **avg of 3 reps** |
| **Offline 3T (single first-rep)** | same | same | full-run | **first-rep only** |
| End-of-run RT | rtmotion (per-TR MCFLIRT + flirt cross-session) | nilearn `FirstLevelModel(noise_model='ar1', hrf_model='glover', drift_model='cosine', drift_order=1, high_pass=0.01, signal_scaling=False)` LSS, refit per trial | full run (~190 TRs) | first-rep only |
| Slow RT | same | same | onset_TR + ~20 TRs (~30 s post-stim) | first-rep only |
| Fast RT | same | same | onset_TR + ~5 TRs (~8 s post-stim) | first-rep only |

The two NSD rows are upper-bound references on a different subject (7T NSD subj01 with ~30 sessions of training). The five 3T rows are the actual contribution: same model + same test images, varying preprocessing pipeline and per-trial window.

## Required artifacts on disk

| Artifact | What | Path on this rig |
|---|---|---|
| **MindEye decoder checkpoint** | Single-session ses-01 fine-tune, fold 10, 150 epochs, finalmask (2792 voxels) | `/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth` (pulled from `macandro96/mindeye_offline_ckpts` on HF) |
| Brain mask (3D) | finalmask, ~19k voxels in T1w-aligned space | `/data/derivatives/rtmindeye_paper/rt3t/data/sub-005_final_mask.nii.gz` |
| Reliability mask | bool selector (length 19174) → 2792 voxels = decoder input dim | `/data/derivatives/rtmindeye_paper/rt3t/data/sub-005_ses-01_task-C_relmask.npy` |
| Per-voxel HRF index | from training sessions, used by GLMsingle Stage 1 | `/data/3t/data/avg_hrfs_s1_s2_full.npy` |
| GLMsingle HRF library | 20 candidate HRFs | `/data/3t/data/getcanonicalhrflibrary.tsv` |
| Canonical Offline βs | Princeton-published GLMsingle TYPED_FITHRF_GLMDENOISE_RR | `/data/derivatives/rtmindeye_paper/glmsingle/glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz` (from `rishab-iyer1/glmsingle` on HF) |
| Saved RT betas | per-decode-delay outputs of paper RT pipeline | `/data/derivatives/rtmindeye_paper/rt3t/data/real_time_betas/all_betas_ses-03_all_runs_delay{0,1,3,5,10,15,20,63}.npy` |
| Pre-computed CLIP embeddings | special515 GT image embeddings (256×1664 each) | `/data/derivatives/rtmindeye_paper/task_2_1_betas/gt_cache/` |
| CLIP image embedder | for computing GT on-the-fly | `/data/rt_all_data/clip_img_embedder` |
| SDXL diffusion engine | reconstruction generation | `/data/rt_all_data/diffusion_engine` + `cache/unclip6_epoch0_step110000.ckpt` |
| BOLD: rtmotion | per-TR motion-corrected + flirt-cross-session aligned | `/data/3t/derivatives/motion_corrected_resampled/{ses-03_run-NN_NNNN_mc_boldres.nii.gz}` |
| BOLD: fmriprep | T1w-aligned preprocessed BOLD | `/data/derivatives/rtmindeye_paper/fmriprep_mindeye/.../sub-005_ses-NN_task-C_run-NN_space-T1w_desc-preproc_bold.nii.gz` |
| Events (timing + image labels) | TSV per run | `/data/derivatives/rtmindeye_paper/rt3t/data/events/sub-005_ses-NN_task-C_run-NN_events.tsv` |

## Per-row reproducibility recipe

### Offline 3T (avg 3 reps) → 90% top-1

1. fmriprep (24.0.1) on sub-005 ses-01-03 BOLD, output in `space-T1w`
2. Run cvnlab/GLMsingle on ses-03 fmriprep BOLD with default options (`wantlibrary=1, wantglmdenoise=1, wantfracridge=1`), one column per unique image, multiple TRs marking reps
3. Get TYPED_FITHRF_GLMDENOISE_RR.npz with `betasmd` shape (V_brain, 693)
4. Project to 2792-voxel finalmask via `betas[final_mask][relmask]`
5. Z-score voxelwise using **training images only from the entire session** (paper §2.5.1)
6. For each special515 image, average its 3 reps → 50 averaged βs
7. Forward through fold-10 ckpt: `betas → ridge → backbone → clip_voxels` (projector head output)
8. Cosine similarity vs 50 GT image CLIP embeddings → top-1

### Offline 3T (single first-rep) → 76% top-1

Same as above through step 5; then:
6. Filter trials to **first occurrence per special515 image** (50 trials)
7-8. Same forward pass + retrieval

### End-of-run RT → 66% top-1

1. rtmotion BOLD: per-TR FSL MCFLIRT against run-01-vol-0, then `applywarp` cross-session
2. Per-trial nilearn LSS at end of run: fit `FirstLevelModel(t_r=1.5, slice_time_ref=0, hrf_model='glover', drift_model='cosine', drift_order=1, high_pass=0.01, noise_model='ar1', signal_scaling=False)`, with `mc_params` (6 motion regressors from MCFLIRT) as confounds. Probe = current trial; reference = all other trials in the same run.
3. Causal cumulative z-score across trial βs as session progresses (paper §2.5.2: "cumulatively as the session progresses")
4. Filter to first-rep special515 (50 trials)
5. Same forward pass + retrieval

### Slow RT → 58% top-1

Same as End-of-run RT but step 2 fits LSS on BOLD cropped to `imgs[:onset_TR + ~20]` (i.e., ~30 s of BOLD post stimulus, not the full run). Specifically: at each non-blank TR `t`, the GLM is fit using `imgs[:t+1]` and `events[onset <= t*tr]` per Rishab's notebook cell 19. Decode happens after a fixed delay; for "Slow" the delay is ~30 s.

### Fast RT → 36% top-1

Same as Slow RT but window is ~8 s (~5 TRs) post stimulus. Same per-TR LSS, just earlier decode point.

### Offline NSD / Offline NSD (avg 3 reps) → 78% / 100% top-1

Same MindEye2 architecture, fine-tuned on NSD 7T subj01 (one session of NSD data) instead of sub-005 ses-01. Test on the same 50 special515 images using NSD subj01's βs (~30 sessions of NSD data are available; reconstruction performance on this subject is near-saturated). These rows are upper-bound references — the 100% confirms the model itself doesn't bottleneck on canonical NSD data, framing the 3T rows as the actual contribution.

## Pitfalls and clarifications

1. **The decoder checkpoint that produced Table 1 is `repeats_3split_10_..._epochs_150` (fold 10), NOT fold 0.** Multiple `_ses-01_task-C_..._3split_N_avgrepeats_finalmask` checkpoints exist on HF; only fold 10 + epochs_150 reproduces the paper. The `sample=N` checkpoints in `data_scaling_exp/` are for the data-scaling appendix, not Table 1.
2. **`pcnum=0` for sub-005 ses-03** in the canonical .npz means GLMsingle Stage 2 (GLMdenoise) selected zero PCs via bootstrap CV on this subject — the offline lift comes from Stages 1 + 3, not Stage 2. Per-subject `pcnum` ranges 0–6 across the 9 published .npz files; **K=10 was never selected** anywhere.
3. **GLMsingle `sub-005_ses-03` uses `_all_task-C_` BOLD** (concatenates ses-01-03) at the GLMsingle pipeline level, but the fine-tune step that produces the decoder checkpoint only sees ses-01.
4. **The deployed RT pipeline (`mindeye.py` from `rtcloud-projects/mindeye`) uses the `unionmask` 8627-voxel ses-01-03 finetune checkpoint** — different mask, different fine-tune scope, different test session (ses-06 with MST pairs). That pipeline is for the live scanner pilot; **Table 1 simulates the same algorithm retrospectively on ses-03 with finalmask + ses-01-only checkpoint**.
5. **z-score policy differs by tier.** Paper §2.5.1 Offline: "voxelwise using the training images from the entire session" — session-wide, training-only (excludes the 150 special515 test reps from z stats). §2.5.2 RT: "cumulatively as the session progresses" — causal, past-only. Both end up giving similar numbers on this data because special515 is only 22% of the 693 trials.
6. **5-seed averaging is a Table 1 caption convention, NOT a Scotti 2023/2024 inheritance** — the original MindEye papers don't explicitly average over diffusion seeds. This is per Iyer paper Table 1: "Reconstruction metrics are averaged over 5 random seeds; retrieval is deterministic."
7. **Retrieval is deterministic.** Both predicted CLIP voxels and GT image embeddings are computed once; cosine similarity → argmax has no random component.
8. **The "first-rep" filter** picks the first occurrence (in TR order across runs) of each unique special515 image. We verified ses-03 contains exactly 50 unique special515 images, so first-rep = 50 trials.

## Cross-references

- Pre-registration: `TASK_2_1_PREREGISTRATION.md` and `TASK_2_1_AMENDMENT_2026-04-28.md`
- Live findings + open questions: `TASK_2_1_FINDINGS.md`
- Self-contained writeup for Rishab: `docs/task_2_1_for_rishab.md`
- Princeton pilot recommendations: `docs/princeton_pilot_recommendations.md`
- Glossary: `docs/glossary.md`
- Backup full 10-metric scorer: `scripts/score_full_metrics.py` (untested; needs `generative_models` import resolved)
