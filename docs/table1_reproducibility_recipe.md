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

### End-of-run RT → 66% top-1 (corresponds to `delay=63` in saved RT betas)

1. rtmotion BOLD: per-TR FSL MCFLIRT against run-01-vol-0, then `applywarp` cross-session
2. Per-trial nilearn LSS at end of run: fit `FirstLevelModel(t_r=1.5, slice_time_ref=0, hrf_model='glover', drift_model='cosine', drift_order=1, high_pass=0.01, noise_model='ar1', signal_scaling=False)`, with `mc_params` (6 motion regressors from MCFLIRT) as confounds. Probe = current trial; reference = all other trials in the same run.
3. Causal cumulative z-score across trial βs as session progresses (paper §2.5.2: "cumulatively as the session progresses")
4. Filter to first-rep special515 (50 trials)
5. Same forward pass + retrieval

### Slow RT → 58% top-1 (corresponds to `delay=5` in saved RT betas)

Same as End-of-run RT but the per-trial GLM fits BOLD only up to **~30 s after each trial's stim onset**. Per-trial windowing: at each non-blank TR `t`, the GLM is fit using `imgs[:t+1]` and `events[onset <= t*tr]` per Rishab's notebook cell 19.

### Fast RT → 36% top-1 (corresponds to `delay=0` in saved RT betas — the default)

Same as Slow RT but the per-trial GLM fits BOLD only up to **~7.5–7.9 s after stim onset** (HRF peak — the minimum useful window). This is the **default** delay in the paper RT pipeline.

### How the `delay` parameter actually works (per Rishab's clarification)

The `delay=N` parameter on the saved `all_betas_ses-03_all_runs_delay{N}.npy` files is **N TRIALS** (NOT N TRs and NOT N seconds — this isn't documented in the paper text and is the single most confusing convention).

- **delay=0 = "default" = ~7.5 s post-stim of *current* trial**. This is the Fast RT point — fit GLM as soon as the HRF peaks for the current trial.
- **delay=N = "wait N more trials before fitting"**. Each trial is 4 s (3 s stim + 1 s ISI), so the GLM is fit at `7.5 + 4N` seconds after the current trial's stim onset.
- **delay=63 = "wait until last trial of the 63-trial run" = End-of-Run**. Every trial in the run gets near-full-run BOLD evidence by the time it's decoded at `delay=63`.

Conversion to TR units (TR=1.5 s):

| `delay` (trials) | post-stim s | post-stim TRs | maps to |
|---|---|---|---|
| 0 | 7.5 | 5 | Fast RT (default) |
| 1 | 11.5 | ~8 | (intermediate) |
| 3 | 19.5 | ~13 | (intermediate) |
| 5 | 27.5 | ~18 | **Slow RT** (paper Table 2: 29.5 ± 2.6 s) |
| 10 | 47.5 | ~32 | (intermediate) |
| 15 | 67.5 | ~45 | (intermediate) |
| 20 | 87.5 | ~58 | (intermediate) |
| 63 | 259.5 (or capped at run end) | ~190 capped | **End-of-Run RT** |

For trial `i` decoded at delay=N:

```
decode_TR_for_trial_i = onset_TR_{i+N} + ~5 TRs  (HRF peak of the future trial we wait for)
GLM fit on:  BOLD[0 : decode_TR_for_trial_i + 1]
events:      events[onset <= decode_TR_for_trial_i * 1.5s]
```

That's why the End-of-run row in Table 2 has stim-delay σ=80 s: at delay=63 the wait is calendar-fixed to the last trial, so trial 0 in a run gets ~250 s of post-stim wait while trial 62 gets only ~5 s, averaging ~130 s.

### Offline NSD / Offline NSD (avg 3 reps) → 78% / 100% top-1

Same MindEye2 architecture, fine-tuned on NSD 7T subj01 (one session of NSD data) instead of sub-005 ses-01. Test on the same 50 special515 images using NSD subj01's βs (~30 sessions of NSD data are available; reconstruction performance on this subject is near-saturated). These rows are upper-bound references — the 100% confirms the model itself doesn't bottleneck on canonical NSD data, framing the 3T rows as the actual contribution.

## Pitfalls and clarifications

1. **The `_3split_N_` index in checkpoint names = training random seed, not k-fold split.** Per `run_all_batch.slurm:57`, the slurm array index `N` is passed both to the model name and to `--seed=N`. Same data, same train/test split; varies only model initialization + batch shuffle. Different seeds give different deterministic retrieval numbers (~12pp variance observed across folds). Multiple `_3split_N_` checkpoints exist on HF (N ∈ {0, 3, 5, 7, 10}); fold-0 reproduces paper's "Offline 3T" 76% Image exactly with `filter_and_average_repeats`, and fold-10 reproduces "Offline 3T (avg. 3 reps.)" 88-90% Image. The two paper Offline rows likely correspond to two different training seeds. The `sample=N` checkpoints in `data_scaling_exp/` are for the data-scaling appendix, not Table 1.
2. **`pcnum=0` for sub-005 ses-03** in the canonical .npz means GLMsingle Stage 2 (GLMdenoise) selected zero PCs via bootstrap CV on this subject — the offline lift comes from Stages 1 + 3, not Stage 2. Per-subject `pcnum` ranges 0–6 across the 9 published .npz files; **K=10 was never selected** anywhere.
3. **GLMsingle `sub-005_ses-03` uses `_all_task-C_` BOLD** (concatenates ses-01-03) at the GLMsingle pipeline level, but the fine-tune step that produces the decoder checkpoint only sees ses-01.
4. **The deployed RT pipeline (`mindeye.py` from `rtcloud-projects/mindeye`) uses the `unionmask` 8627-voxel ses-01-03 finetune checkpoint** — different mask, different fine-tune scope, different test session (ses-06 with MST pairs). That pipeline is for the live scanner pilot; **Table 1 simulates the same algorithm retrospectively on ses-03 with finalmask + ses-01-only checkpoint**.
5. **z-score policy differs by tier.** Paper §2.5.1 Offline: "voxelwise using the training images from the entire session" — session-wide, training-only (excludes the 150 special515 test reps from z stats). §2.5.2 RT: "cumulatively as the session progresses" — causal, past-only. Both end up giving similar numbers on this data because special515 is only 22% of the 693 trials.
6. **5-seed averaging is a Table 1 caption convention, NOT a Scotti 2023/2024 inheritance** — the original MindEye papers don't explicitly average over diffusion seeds. This is per Iyer paper Table 1: "Reconstruction metrics are averaged over 5 random seeds; retrieval is deterministic."
7. **Retrieval is deterministic.** Both predicted CLIP voxels and GT image embeddings are computed once; cosine similarity → argmax has no random component.
8. **The "first-rep" filter** picks the first occurrence (in TR order across runs) of each unique special515 image. We verified ses-03 contains exactly 50 unique special515 images, so first-rep = 50 trials.
9. **`delay` parameter is in TRIALS, not TRs and not seconds.** Per Rishab's clarification (paper text doesn't document this): delay=0 is the "default" = ~7.5 s post-stim (HRF peak); delay=N waits N more trials × 4 s/trial; delay=63 = end-of-63-trial-run. Mapping: delay=0 → Fast, delay=5 → Slow (≈30 s post-stim), delay=63 → End-of-run. This is the single most confusing convention in the paper-deployed pipeline and is worth clarifying in the resubmission.
10. **"Resampling" has two meanings** in this codebase. The paper mindeye.py FLIRT-cross-session-aligns each TR to fmriprep's boldref, producing files in `motion_corrected_resampled/` — that's *registration*, not spatial-resolution resampling. Rishab's "resampling" in the Discord clarification refers to *spatial* interpolation to a different voxel grid (e.g., MNI 2 mm vs 1 mm), which is a separate, optional step that doesn't apply to the Table 1 simulation.
11. **Checkpoint per row is partially open.** Per Cesar's Discord clarification (2026-05-02), all Table 1 rows are intended to use the same checkpoint family (offline-preproc fine-tune `_avgrepeats_finalmask_epochs_150`). However, our reproduction shows fold-0 reproduces "Offline 3T" 76% Image exactly via `filter_and_average_repeats`, while fold-10 reproduces "Offline 3T (avg. 3 reps.)" 88-90% Image. If the paper used a single fold for both Offline rows, there must be a different mechanism producing the 76 vs 90 difference — most likely different rep-aggregation (pre-model β-avg vs post-model output-avg). Worth confirming with authors before camera-ready.
12. **HF metrics.csv at `realtime-dump/sub-005_ses-01_task-C-offline_ft_split=repeats3_epochs=150_delay=0/metrics.csv`** is the published single-seed delay=0 (Fast RT) row from the offline-preproc fine-tune ckpt. 1-rep top-1 fwd = **0.36** there matches our reproduction (0.38) within 2 pp ✓. 3-rep avg top-1 fwd = 0.72 in this delay=0 file is *not* the same row as paper Table 1's "Offline 3T (avg 3 reps) = 0.90" — that 0.90 row uses the **fmriprep + nilearn GLM** β path (not delay=0 RT-preproc βs). Don't confuse the two.
13. **Paper's RT eval mechanism is `subset0`/`subset1` from `mindeye.py:947-955`, not pure first-rep filtering.** When `is_repeat==True` (lines 773-784), `mindeye.py` stores the running-average of all accumulated z-scored βs for that image — so the prediction stored at the time of the 2nd-rep trial is `mean(β_rep0, β_rep1)`, not just β_rep1. The eval at lines 947-955 reads `duplicated[:,0]` (= 1st-rep prediction = single-rep) and `duplicated[:,1]` (= 2nd-rep prediction = avg-of-2 by running average). For the 50 special515 with 3 reps in ses-03, this gives 100 evaluation entries (50 primary + 50 secondary indices). Paper Table 1's RT-tier rows likely report the subset1 (avg-of-2) numbers, NOT pure first-rep. Verified empirically: subset1 of `RT_paper_EoR_K7_CSFWM_HP_e1_inclz` on fold-0 = 66% Image, exactly matching paper's EoR row.
14. **Brain retrieval column on Offline 3T row (64%) is suspect.** The same eval that produces Image=76% (fold-0 + `filter_and_average_repeats`) gives Brain=88% in our reproduction, not 64%. The 88% matches the paper's "(avg 3 reps)" Brain column exactly. Either the paper has a typo on Offline 3T Brain, or the row reports Image and Brain from different rep-aggregation modes. Worth confirming with authors before camera-ready.
15. **`utils.filter_and_average_repeats`** at `PrincetonCompMemLab/mindeye_offline:avg_betas/utils.py:800` is the canonical implementation of the avg-of-3 collapse for Offline rows. Used in `recon_inference-multisession.ipynb` (avg_betas branch) when `train_test_split == 'repeats_3'`. Reproducers should call this verbatim rather than rolling their own averaging.

## Project extensions

These extend beyond paper-Table-1 reproduction with new methods or analyses
developed during the project. Each lives in `results/apple_silicon_2026-04-28/`
on the `results/apple-silicon-2026-04-28` branch.

- **`STREAMING_RLS_GLM.md`** — growing-design ridge OLS at decode time, an
  alternative to per-trial LSS for the RT tiers. Implements Ernest Lo's
  "persistent GLM" proposal cleanly. Gives +14pp on Slow subset1 / +4pp on
  EoR subset1 over per-trial AR(1) LSS at matched nuisance regressors.
  Driver: `local_drivers/run_streaming_rls_glm.py` (Mac), `scripts/run_streaming_rls_glm.py` (DGX).

- **`STREAMING_GLM_AR1_ABLATION.md`** — 3-way ablation (per-trial AR(1) LSS
  vs streaming OLS GLM vs streaming AR(1) GLM) confirms the +14pp Slow gain
  is from the joint growing-design, not from missing AR(1) prewhitening. Streaming
  with or without AR(1) gives identical numbers; AR(1) prewhitening is redundant
  once the joint trial-design absorbs the relevant temporal autocorrelation.

- **`FAST_DISTILLATION.md`** — cross-latency distillation Fast ← streaming-Slow
  via per-voxel scalar refiner. Pushes Fast tier from 36% → 42% Image (+6pp)
  and 34% → 48% Brain (+14pp) on the training session. Brain gain transfers
  cross-session; Image is more BOLD-source-dependent.

- **`COMBINED_PIPELINE_DEPLOYMENT.md`** — held-out deployment of the streaming
  Slow GLM + Fast refiner on ses-01 (different special515 image set than ses-03).
  Streaming Slow generalizes cleanly (78% Image at avg-of-3 on ses-01, matches
  ses-03). Refiner Brain gain transfers; Image gain is BOLD-source-fragile.
  Includes per-trial decoder confidence > tSNR finding for deployment alerting.

- **`HEUNIS_COVERAGE.md`** — 137-cell preprocessing factorial mapped against
  Heunis et al (2020) RT-fMRI denoising taxonomy. Documents what we tried,
  what worked, and the categories not covered (physio/RETROICOR, ICA-AROMA,
  spatial smoothing, GSR, 24-param Friston motion).

- **`EOR_OFFLINE_GAP.md`** — analysis of what specifically drives the
  Offline-vs-EoR retrieval gap when comparing at matched subset semantics.
  Resolves the apparent 10pp gap as primarily subset-mismatch (Offline subset2
  vs EoR subset1) plus modest β-quality differences that vanish at avg-of-3.

- **`RISHAB_LADDER_REPORT.md`** — corrected version of the original four-tier
  ladder report. Earlier headline ("Offline 3T 76% reproduces single-rep")
  was mislabeled — the 76% number was actually from `filter_and_average_repeats`,
  not first-rep. This rewrite is paper-author-facing.

- **`FACTORIAL_BOTH_MODES.md`** — the 130-cell factorial scored under both
  first-rep and avg-of-3 modes on fold-0. Shows ~20-30pp gap between modes
  for most cells; useful when judging whether a preprocessing change is
  load-bearing or noise.

## Cross-references

- Pre-registration: `TASK_2_1_PREREGISTRATION.md` and `TASK_2_1_AMENDMENT_2026-04-28.md`
- Live findings + open questions: `TASK_2_1_FINDINGS.md`
- Self-contained writeup for Rishab: `docs/task_2_1_for_rishab.md`
- Princeton pilot recommendations: `docs/princeton_pilot_recommendations.md`
- Glossary: `docs/glossary.md`
- Backup full 10-metric scorer: `scripts/score_full_metrics.py` (untested; needs `generative_models` import resolved)
