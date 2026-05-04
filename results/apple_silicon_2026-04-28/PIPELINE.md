# End-to-end pipeline reference for the realtime-mindeye paper anchors

This document traces every stage from MRI scanner → preprocessing → segmentation
→ GLM → MindEye2 inference → final retrieval percentages. For each stage:
inputs, outputs, file paths on this Mac, and which Table 1 number it feeds.

Use this when:
- Asking "which file does this number come from?"
- Tracing why a reproduction differs from the paper
- Onboarding a new analyst to the project
- Designing a deployment pipeline that mirrors any subset of the offline
  pipeline

**Subject**: sub-005, session ses-03 (test session). Same shape applies to
ses-01 (training session) for fine-tuning data construction.

---

## Stage 0 — Acquisition

### Hardware
- 3T Siemens Prisma + 64-channel head coil at Princeton Neuroscience Institute
- T2*-weighted multiband EPI: TR=1500ms, TE=33ms, voxel=2.0mm isotropic, MB factor=4, 52 slices
- Partial volumes covering occipital and temporal lobes
- Whole-brain T1 MPRAGE: TR=2300ms, TE=2.98ms, 1.0mm iso, 176 slices, GRAPPA=2

### Trial structure
- 11 functional runs per session, 192 TRs/run = 288 s/run
- 70 events/run including 7 blank trials (HRF-decay anchors)
- 4-second SOA (3 s image + 1 s ITI)
- ~770 events/session, ~693 non-blank trials
- 50 "special515" images repeated 3× each in ses-03 = 150 test trials
- 543 unique training images split across all sessions

### Stimulus design
- Each trial shows one image for ~3 s
- Images come from three sources:
  - `all_stimuli/special515/` (NSD shared1000 subset, 50 retained as test set)
  - `all_stimuli/unchosen_nsd_1000_images/`, `all_stimuli/MST_pairs/`, `all_stimuli/shared1000_notspecial/` (training)
- Events.tsv per run lists onset, duration, image_name

---

## Stage 1 — Functional + structural data on disk

### Files (sub-005)

| What | Path | Size |
|---|---|---|
| Raw T1 (already preprocessed) | `rt3t/data/sub-005_desc-preproc_T1w.nii.gz` | 19 MB |
| Skull-stripped T1 | `rt3t/data/sub-005_desc-preproc_T1w_brain.nii.gz` | 6 MB |
| FSL FAST PVEs | `rt3t/data/T1_brain_seg_pve_{0,1,2}.nii.gz` | 9 MB ea |
| BOLD reference | `rt3t/data/sub-005_ses-01_task-C_run-01_space-T1w_boldref.nii.gz` | 0.4 MB |
| Per-TR motion-corrected BOLD (rtmotion) | `motion_corrected_resampled/{ses}_run-{NN}_{TTTT}_mc_boldres.nii.gz` | ~10 KB/TR × 192 TRs/run × 11 runs |
| Per-TR motion params | `motion_corrected_resampled/{ses}_run-{NN}_motion.par` | ~6 KB/run |
| fMRIPrep BOLD (offline preproc) | `fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005/{ses}/func/sub-005_{ses}_task-C_run-{NN}_space-T1w_desc-preproc_bold.nii.gz` | ~40 MB/run |
| fMRIPrep T1→MNI xfm | `fmriprep_mindeye/.../anat/sub-005_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5` | ~70 MB |
| Events | `rt3t/data/events/sub-005_{ses}_task-C_run-{NN}_events.tsv` | 70 rows × 6 cols |
| Per-TR labels | `rt3t/data/events/sub-005_{ses}_task-C_run-{NN}_tr_labels.csv` | 192 rows × 3 cols |
| Stimulus images | `rt3t/data/all_stimuli/special515/special_NNNNN.jpg` | 50 images, 256×256 each |

### Two BOLD pipelines run in parallel
1. **rtmotion** (real-time-deployable): per-TR FSL MCFLIRT against run-01-vol-0
   reference, then `applywarp` to apply pre-computed cross-session FLIRT to align
   to T1w space. Produces 192 individual NIfTI volumes per run. Used for all RT
   tier cells.
2. **fmriprep** (offline): full fMRIPrep pipeline (motion correction, slice-time
   correction, distortion correction, T1w-space registration). Produces a single
   4D NIfTI per run. Used for the Offline tier.

These differ in: (a) preprocessing sophistication (fMRIPrep more), (b) timing
of registration (rtmotion uses streaming-compatible per-TR ops), (c) noise
characteristics. Round 3 showed BOLD source contributes ≤2pp on 2-AFC.

---

## Stage 2 — Brain mask hierarchy

Three nested masks for different purposes. **The relmask is the MindEye2
input dimension and the inference-time bottleneck.**

```
506,160 (full BOLD volume = 76 × 90 × 74)
→  19,174 (sub-005 finalmask: brain ∩ nsdgeneral, in T1w space)
→   2,792 (relmask: r > 0.2 across special515 reps in ses-01 training)
```

### Files

| Mask | Path | n voxels |
|---|---|---|
| Full BOLD volume | (any 4D NIfTI's spatial dims) | 506,160 |
| nsdgeneral ∩ finalmask | `rt3t/data/sub-005_final_mask.nii.gz` | 19,174 |
| Reliability mask within finalmask | `rt3t/data/sub-005_ses-01_task-C_relmask.npy` | 2,792 (boolean array) |

### How the relmask was constructed (paper §2.7.1)

Per the paper:
> "Reliability for a given voxel is defined as the average correlation of
> responses for repeated presentations of the same image. Specifically, we
> computed the Pearson's correlation of betas (from GLMsingle) across the
> first two instances of each repeated image in the training session and set
> a reliability threshold at r > 0.2 to generate a binary mask."

The relmask was computed from ses-01 GLMsingle βs (NOT ses-03), so the test
set in ses-03 doesn't leak into voxel selection.

---

## Stage 3 — GLM β estimation per trial

Two paths produce per-trial βs for downstream MindEye2 input. Both produce
βs in the 2792-voxel relmask frame.

### Path A: Canonical Princeton GLMsingle (paper "Offline" pipeline)

- Input: fMRIPrep BOLD across all 11 runs of ses-03
- Tool: GLMsingle (Prince, Charest, Kay et al. 2022) with TYPED_FITHRF_GLMDENOISE_RR mode
  - Stage 1: 20-HRF library lookup, per-voxel best fit
  - Stage 2: GLMdenoise PCA on noise pool, K cross-validated (chose `pcnum=0` for sub-005)
  - Stage 3: per-voxel SVD-based fracridge, fraction cross-validated
- Output: `glmsingle/glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz`
  - Contains `betasmd` shape (76, 90, 74, 693) — 693 non-blank trial βs in BOLD volume space
  - Plus `sub-005_ses-03_task-C_brain.nii.gz` — Princeton's own brain mask (183,408 voxels, ≠ our finalmask)
- Source: pulled from HuggingFace `rishab-iyer1/glmsingle`
- Local size: ~620 MB

### Path B: Real-time deployable LSS (paper "RT" pipelines)

- Input: rtmotion per-TR BOLD
- Tool: `nilearn.glm.first_level.FirstLevelModel`
  - hrf_model="glover" (canonical Glover HRF, see HRF sweep doc)
  - drift_model="cosine", drift_order=1, high_pass=0.01
  - noise_model="ar1" (per-voxel AR(1) prewhitening via Toeplitz-Cholesky)
  - signal_scaling=False
- Per-trial fit: Least-Squares-Separate ("current trial vs all others" contrast)
  - For each trial i, design has "probe" column (trial i's onset) and "reference" column (all other trials' onsets)
  - Output: per-voxel β for the "probe" column = single-trial β
- For "End-of-run" RT: fit on full-run BOLD (192 TRs)
- For "Slow" RT: fit on BOLD cropped to `onset_TR + 20 TRs ≈ 30 s` post-stim (Slow window)
- For "Fast" RT: fit on BOLD cropped to `onset_TR + 5 TRs ≈ 7.5 s` post-stim
- aCompCor noise components added as confounds (champion: K=7 PCs from CSF+WM via FAST PVEs > 0.5, eroded ×1, HP-filtered at 0.01 Hz before SVD)

The `delay` parameter (per Rishab's Discord clarification) is in **TRIALS**, not TRs:
  - delay=0 → Fast (~7.5 s post-stim of current trial)
  - delay=5 → Slow (= 7.5 + 5×4 s = 29.5 s)
  - delay=63 → End-of-run (decode at trial 63)

### Saved cells in `task_2_1_betas/prereg/`

Per cell saves three files: `{cell}_{ses}_betas.npy`, `{cell}_{ses}_trial_ids.npy`, `{cell}_{ses}_config.json`. Examples:
- `Canonical_GLMsingle_OfflineFull_ses-03_betas.npy` — canonical GLMsingle βs projected to relmask
- `RT_paper_EoR_K7_CSFWM_HP_e1_inclz_ses-03_betas.npy` — champion RT-deployable cell

---

## Stage 4 — Z-score normalization

Different per tier (paper main.tex):

| Tier | Z-policy | Where it lives | What it means |
|---|---|---|---|
| Offline (§2.5.1) | "z-scored voxelwise using the **training images from the entire session**" | session-wide, training-images-only stats applied to test trials | excludes 150 special515 test reps from stats |
| RT (§2.5.2) | "z-scored cumulatively as the session progresses" | causal, past-only | strict no-test-leakage, RT-deployable |

Tested on this dataset (Round 5 z-policy matrix, on canonical .npz):

| z policy | first-rep | rep-avg |
|---|---|---|
| session_all (leaky, includes test) | 62% | 76% |
| session_train (paper §2.5.1) | 60% | 76% |
| causal_cumz (paper §2.5.2 RT-style) | 56% | 78% |

The "leakage" from including test trials is < 2pp because special515 is only
22% of all trials.

---

## Stage 5 — Per-image pooling at evaluation time

50 special515 images × 3 reps = 150 test trials. Two evaluation modes:

### First-rep (default, paper §2.7 line 239)
> "Unless otherwise stated, evaluations use single-trial betas from the first
> presentation of the three repeats only."

Filter to first occurrence of each special515 image in chronological event order:
50 trials × 2792 voxels.

### Avg-3-rep (the "(avg. 3 reps.)" rows in Table 1)

Average β across the 3 reps per image: 50 averaged trials × 2792 voxels.

**Paper Table 1's 76% Offline 3T anchor is supposed to be first-rep per §2.7
but the canonical seedwise dump only has avg-3-rep numbers (90/88).** Open
question — see "Anomalies" below.

---

## Stage 6 — MindEye2 inference

### Architecture (per paper §2.6)

```
β (50, 2792) →
  ridge(subject_idx=0) →           # subject-specific projection
  backbone (residual MLP) →
    branch 1: clip_voxels (50, 256, 1664)    # 256 tokens × 1664-d, OpenCLIP ViT-bigG/14 token shape
    branch 2: diffusion_prior (used for reconstruction, not retrieval)
```

### Checkpoints

| Checkpoint | Path | Train data | Used for |
|---|---|---|---|
| **fold-10 (paper-faithful)** | `rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth` | ses-01, 543 non-special515 images, 150 epochs | Table 1 numbers |
| fold-0 (we'd been using until round 6) | `rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth` | same data, different fold seed | early-session work |
| sample=10 (multi-session) | (DGX-only) | sub-005 ALL sessions including ses-03 — has test-set leakage | DGX comparison |

Source: HuggingFace `macandro96/mindeye_offline_ckpts`. URL the user shared:
`https://huggingface.co/macandro96/mindeye_offline_ckpts/tree/main/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150`

### Inference

```python
model, ss, se = M.load_mindeye(ckpt_path, n_voxels=2792, device="mps")
# β: (50, 2792)  — per-image trials, post-z, post-pooling
b = torch.from_numpy(β).to(device).unsqueeze(1)  # (50, 1, 2792)
for i in range(50):
    voxel_ridge = model.ridge(b[i:i+1], 0)            # subject_idx=0 hardcoded
    backbone_out = model.backbone(voxel_ridge)         # (256, 1664) tuple
    clip_voxels = backbone_out[1]                      # the projector branch
    out.append(clip_voxels.float().cpu().numpy())
pred = np.concatenate(out, 0).reshape(-1, ss, se)     # (50, 256, 1664)
```

Note: hardcoded `subject_idx=0`. Paper checkpoint is single-subject so this is fine.

### Training-time post-model averaging

Per Train.py:1066-1078 (canonical Princeton):
```python
for rep in range(3):
    voxel_ridge = ddp(model).ridge(voxel[:,rep], 0)
    backbone0, clip_voxels0, _ = ddp(model).backbone(voxel_ridge)
    if rep==0:
        clip_voxels = clip_voxels0
    else:
        clip_voxels += clip_voxels0
clip_voxels /= 3
```

This averages CLIP_VOXELS *output* across the 3 reps — POST-model averaging,
distinct from our pre-model β-averaging. The two operations don't commute
through the nonlinear backbone in general. **The published seedwise_runs_dump
files contain post-model-averaged clip_voxels.**

---

## Stage 7 — Ground-truth CLIP embeddings

Per paper §2.6, MindEye2 outputs are aligned with OpenCLIP ViT-bigG/14's
penultimate-layer image token embeddings (256 × 1664).

### Image preprocessing (verified pixel-exact in round 7)

```python
pil = Image.open(special_NNNNN.jpg).convert("RGB")  # source: usually 424×424
arr = np.asarray(pil, dtype=np.float32) / 255.0
t = torch.from_numpy(arr).permute(2, 0, 1)
t224 = T.functional.resize(t, (224, 224),
                            antialias=False,           # ← MUST be False
                            interpolation=T.InterpolationMode.BILINEAR)
img_norm = T.functional.normalize(t224.unsqueeze(0).to(device),
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711])
```

The `antialias=False` matches paper exactly (verified by diffing against
saved 224×224 images in `sanity_check_individual_reps/` on HF — 0.0000 pixel
diff). With `antialias=True` (PyTorch default), pixel diff is 0.0175 mean,
which causes 1 trial to flip in retrieval scoring.

### CLIP encoder

```python
clip_m, _, _ = create_model_and_transforms("ViT-bigG-14",
                                            pretrained="laion2b_s39b_b160k",
                                            device=device)
clip_m.eval()
clip_m.visual.output_tokens = True
with torch.no_grad():
    _, gt_tokens = clip_m.visual(img_norm)  # (50, 256, 1664)
```

The pretrained weight key `laion2b_s39b_b160k` is OpenCLIP's standard ViT-bigG/14 snapshot.

---

## Stage 8 — Retrieval scoring

### Paper Table 1 metrics

Per paper §2.6:
> "For retrieval, the backbone embedding follows an analogous mapping to CLIP
> image space – but via an MLP "projector" rather than a diffusion prior.
> The resulting (predicted) fMRI-CLIP embedding can then be compared with
> the (ground-truth) CLIP image embeddings of candidate images; the top-k
> retrievals are defined as the k-nearest neighbors (based on cosine
> similarity) to the predicted embedding."

For a (50, 256, 1664) prediction × (50, 256, 1664) GT pair:

```python
sim = M.cosine_sim_tokens(pred, gt)  # → (50, 50)
# Image (forward) retrieval: argmax over candidate column
top1 = ((sim.argmax(axis=1) == np.arange(50))).mean()
# Brain (backward) retrieval: argmax over query row
brain_top1 = ((sim.argmax(axis=0) == np.arange(50))).mean()
```

### Closed-loop / neurofeedback metrics (extras we added)

- 2-AFC pairwise: for each (i, j) pair where i ≠ j, is `sim[i, i] > sim[i, j]`? Average. Chance = 50%.
- ROC-AUC merge/separate: distribution of diagonal vs off-diagonal sims.
- Cohen's d: effect size between diagonal and off-diagonal sims.
- Brier score, ECE: calibration metrics on softmax-of-similarity.
- Selective accuracy at confidence τ: accuracy on high-confidence subset.
- β-reliability: Pearson r between repeated presentations (single-trial signal floor).

---

## Stage 9 — How each Table 1 row gets produced

| Table 1 row | top-1 (Image↑) | brain (Brain↑) | What it actually is |
|---|---|---|---|
| Offline NSD (avg. 3 reps.) | 100% | 96% | NSD subj01, 7T, fold-10-equivalent ckpt, post-model-avg-3-reps |
| **Offline 3T (avg. 3 reps.)** | **90%** | **88%** | sub-005 ses-03, **canonical .npz** βs, fold-10 ckpt, post-model-avg-3-reps |
| Offline NSD (single-rep) | 78% | 82% | NSD subj01 first-rep |
| **Offline 3T (single-rep)** | **76%** | **64%** | sub-005 ses-03, canonical .npz βs, fold-10 ckpt, **first-rep only** |
| End-of-run RT | 66% | 62% | rtmotion, full-run LSS, fold-10, first-rep |
| Slow RT | 58% | 58% | rtmotion, delay=5 trials LSS, fold-10, first-rep |
| Fast RT | 36% | 40% | rtmotion, delay=0 trials LSS, fold-10, first-rep |

### What we have reproduced exactly

- Offline 3T avg-3-rep: 88% Mac / 88% DGX seed-0 (vs paper 90%) — within 2pp sampling variance ✓
- Slow RT: 58% paper-exact on fold-0 (DGX) ✓
- Fast RT: 36% paper-exact (Mac) on fold-0; +8pp overshoot on fold-10 due to RT-tier-specific calibration

### Where reproduction underperforms paper

- Offline 3T single-rep: 56% Mac / 62% DGX (vs paper 76%) — **−14 to −20pp**
  - DGX agent's `b1a19ea` z-policy forensic showed no z × pooling combination on canonical .npz reaches 76% first-rep
  - Best is 60% with session_train z + first-rep
  - The seedwise dump's `_all_clipvoxels.pt` files contain POST-model-avg-3-rep
    representations, not first-rep — so they can't be used to score first-rep directly
  - Open question: where does the 76% paper number come from?
- End-of-run RT: 56% Mac / 64% DGX (vs paper 66%) — DGX matches; Mac uses re-implemented LSS that misses 8pp

---

## Stage 10 — Anomalies and methodology gaps

### The "76% Offline 3T single-rep" anomaly

The clearest unresolved gap. Three candidate explanations:

1. **Paper used a different scoring path for Table 1 first-rep numbers**
   than the seedwise_runs_dump's training-time eval. The seedwise dump's
   `final_evals.csv` reports `fwd_acc=0.90` per seed (= avg-3-rep). If
   paper's first-rep numbers come from a separate post-hoc script that we
   don't have access to, this could be a documentation gap rather than a
   methodology bug.

2. **Per Rishab's Discord 2026-01-26**: some published numbers in their
   internal eval sheet were inadvertently generated using the rt_ft
   checkpoint family (`sub-005-ses-01_task-C-rt_ft_split=repeats3_delay=N_epochs=150`)
   rather than the offline_ft family (which we use). The paper will likely
   revise. If the 76% Offline 3T row was inadvertently from rt_ft, the
   "correct" number with offline_ft is ~62% — which our DGX score matches.

3. **Random seed effect during evaluation**: the seedwise dump shows 90% per
   seed deterministically for rep-avg, but first-rep variance across seeds
   wasn't reported in the dump. Paper's Table 1 may report a maximum or
   median across seeds rather than the mean we computed.

The DGX agent prepared a `RISHAB_LADDER_REPORT.md` and `task_2_1_for_rishab.md`
specifically asking which checkpoint produced Table 1's 76% number. No
definitive answer at the time of this writing.

### Why the seedwise_runs_dump βs aren't directly accessible

The MindEye2 training pipeline reads test data from a WebDataset tarball
(`wds/subj0X/new_test/0.tar` per Train.py:397). We don't have that tarball
locally; we only have:
- The published canonical GLMsingle .npz output (fed INTO the tarball construction)
- The published model output `_all_clipvoxels.pt` (post-model)

Our scoring uses canonical .npz βs through the fold-10 model and matches the
seedwise dump's clipvoxels at avg-3-rep within 1 trial (our 88% vs the
dump's reported 90%). This confirms that our βs and the βs used to train
the model are equivalent UP TO the rep-pooling axis.

For first-rep, we can't directly compare because the dump doesn't contain
first-rep model outputs separately. Either (a) the dump never ran first-rep
evals, OR (b) it ran them and saved them in a path we haven't identified.

---

## Stage 11 — Closed-loop deployment champion (RT-deployable)

After 8 rounds of ablation, the locked deployment recipe is:

```
Stage 0: scanner (TR=1.5, multiband EPI, 2mm iso)
Stage 1: rtmotion BOLD (per-TR FSL MCFLIRT cross-session registration)
Stage 2: brain mask hierarchy (finalmask 19174 → relmask 2792)
Stage 3: nilearn LSS GLM
  - hrf_model="glover" (canonical, see HRF sweep — flexibility hurts)
  - noise_model="ar1" (load-bearing, +8pp vs OLS)
  - drift_model="cosine", drift_order=1, high_pass=0.01
  - aCompCor noise pool added as confounds:
      pool = (CSF PVE > 0.5) ∪ (WM PVE > 0.5) ∩ flat_brain, eroded ×1
      HP-filter at 0.01 Hz before SVD
      top-7 components per run
Stage 4: causal cumulative z-score (paper §2.5.2 inclusive form)
Stage 5: first-rep filter at scoring time (paper §2.7)
Stage 6: MindEye2 fold-10 forward pass
Stage 7: cosine retrieval against OpenCLIP ViT-bigG/14 GT tokens
```

Numbers achieved on sub-005 ses-03 special515 (n=50 first-rep):
- top-1: 58%
- top-5: 88%
- brain retrieval: 64%
- 2-AFC pairwise: **97.2%** (beats canonical full GLMsingle's 96.2% on this metric)
- ROC-AUC merge/separate: 0.958
- Cohen's d: 2.42
- β-reliability: 0.241

Same recipe but with DeepMriPrep (`uv tool install --python 3.12 deepmriprep`)
in place of FAST PVEs gives 96.6% 2-AFC — fully native Apple Silicon path,
no FreeSurfer/FSL container needed.

---

## File reference (everything mentioned above)

### Local paths

| Domain | Path |
|---|---|
| Project root | `/Users/mhough/Workspace/data/rtmindeye_paper/` |
| Stimuli | `rt3t/data/all_stimuli/special515/` |
| Brain masks | `rt3t/data/sub-005_final_mask.nii.gz`, `rt3t/data/sub-005_ses-01_task-C_relmask.npy` |
| MindEye2 ckpts | `rt3t/data/model/` |
| Canonical GLMsingle | `glmsingle/glmsingle_sub-005_ses-03_task-C/` |
| RT-deployable βs | `task_2_1_betas/prereg/RT_paper_*.npy` |
| Cell scoring results | `task_2_1_betas/{retrieval_results_v2,unified_metrics,k_sweep_metrics,erode1_metrics,...}.json` |
| Drivers (this repo) | `hippy-feat/results/apple_silicon_2026-04-28/drivers/` |
| Documentation | `hippy-feat/results/apple_silicon_2026-04-28/{README,NEUROFEEDBACK_METRICS,VARIANT_G_TRIBEV2_DESIGN,PIPELINE}.md` |

### HuggingFace sources

- `rishab-iyer1/glmsingle` — canonical GLMsingle outputs for sub-005 (4 sessions + 2 combined) and sub-001 (3 sessions). 6.6 GB total.
- `macandro96/mindeye_offline_ckpts` — fold-10 + sample={1..10} checkpoints, plus seedwise_runs_dump model outputs.

### Repos / code

- `~/Workspace/hippy-feat/scripts/rt_paper_full_replica.py` — canonical RT replica (run_cell, fit_lss_nilearn)
- `~/Workspace/hippy-feat/scripts/prereg_variant_sweep.py` — JAX cell drivers, GLMdenoise PCA helpers
- `~/Workspace/rt_mindEye2/src/` — local MindEye2 fork (architecture + Train.py)
- `~/Workspace/rtcloud-projects-mindeye/mindeye.py` — canonical Princeton RT script (1019 LOC)
- `~/Workspace/dlbs/freesurfer-patch/recon-all` — patched recon-all for Mac container OOM workarounds

---

## Appendix: ablation summary across 9 rounds

For full per-round details see README.md updates. Headline:

| Round | What was tested | Result |
|---|---|---|
| 1 | EoR + GLMdenoise K=10 (relmask pool) | REJECTED — task leakage drops 6pp |
| 2 | CSF/WM K=10, HRF library, fracridge, Slow diag | All rejected as missing ingredient; pst=25 wins for Slow |
| 3 | fMRIPrep BOLD, Slow pst refinement, adaptive HRF-peak | BOLD source ruled out; pst=25 confirmed |
| 4 | Actual GLMsingle on rtmotion | 78% rep-avg matches Offline anchor; 50% first-rep stays open |
| 5 | Z-policy audit per paper §2.5.1/§2.5.2 | Z-policy moves numbers ≤2pp |
| 6 | Fold-10 ckpt download + rescore | 88% rep-avg ✓; first-rep stays at 56% |
| 7 | 5-seed canonical dump + GT preprocessing trace | Test set verified identical; preprocessing pixel-exact; 1-trial residual after fix |
| 8 | aCompCor refinements + HRF sweep + segmentation alternatives | K=7 + HP + erode×1 (FAST) wins at 97.2% 2-AFC; HRF flexibility hurts; DeepMriPrep matches FAST |
| 9 | Variant G + structured prior + fake-TRIBEv2 + recon-all start | Bayesian framework working; weak linear priors collapse; recon-all running for real TRIBEv2 |

---

*Last updated 2026-05-04 ~00:30 PT. Pipeline state reflects commit `19907f9`
on `m9h/hippy-feat:results/apple-silicon-2026-04-28`.*

---

# File-by-file inventory: what's actually in each file

This section unpacks what we have, where it lives, and what each file
contains. For HuggingFace files we list both the schema and what paper
number/aspect the file supports.

## Canonical Princeton GLMsingle output

**HF source**: `rishab-iyer1/glmsingle` (HuggingFace dataset repo, ~6.6 GB total)

**Top-level structure**:
```
glmsingle_sub-001_ses-01            (3 files)
glmsingle_sub-001_ses-02-03         (6 files)
glmsingle_sub-005_ses-01-02_task-C  (4 files)
glmsingle_sub-005_ses-01-03_task-C  (3 files)
glmsingle_sub-005_ses-01_task-C     (3 files)
glmsingle_sub-005_ses-02_task-C     (3 files)
glmsingle_sub-005_ses-03_task-C     (3 files)  ← what we use for the test session
glmsingle_sub-005_ses-06_task-C     (3 files)
glmsingle_sub-005_task-C            (8 files)  ← all-session combined
csv/                                (14 files) ← metadata tables
```

**Files in `glmsingle_sub-005_ses-03_task-C/` (used for the 76% Offline anchor)**:

| File | Size | Contents |
|---|---|---|
| `TYPED_FITHRF_GLMDENOISE_RR.npz` | ~620 MB | The full GLMsingle output (15 numpy arrays — see schema below) |
| `sub-005_ses-03_task-C_brain.nii.gz` | ~3 MB | Princeton's own brain mask, 183,408 voxels (≠ our finalmask) |
| `sub-005_ses-03_task-C_nsdgeneral.nii.gz` | ~3 MB | Subject-native nsdgeneral ROI mask, 20,484 voxels |

### `TYPED_FITHRF_GLMDENOISE_RR.npz` — the 693-trial β source

Schema (verified by direct `np.load`):

| Key | Shape | dtype | What it is |
|---|---|---|---|
| `betasmd` | (183408, 1, 1, 693) | float32 | Single-trial β estimates: 183,408 brain voxels × 693 non-blank trials |
| `HRFindex` | (183408, 1, 1) | int64 | Per-voxel best-fit HRF index (0..19) from the 20-HRF library |
| `HRFindexrun` | (183408, 1, 1, 11) | int64 | Same but per-run (cross-validated) |
| `glmbadness` | (183408, 1, 1, 11) | float32 | Per-run residual measure (lower = better fit) |
| `pcvoxels` | (183408, 1, 1) | bool | Voxels selected for the PC noise pool |
| `pcnum` | () | int64 | **Cross-validated K for GLMdenoise.** For sub-005 ses-03 this is **0** (zero noise components were picked) |
| `xvaltrend` | (11,) | float32 | Cross-validation trend showing CV-R² as K varies |
| `noisepool` | (183408, 1, 1) | bool | Voxels in the noise pool (low-task-R² voxels) |
| `pcregressors` | (11, 288, 11) | float32 | Noise PC regressors per run (288 TRs/run, 11 PCs); irrelevant since pcnum=0 |
| `R2` | (183408, 1, 1) | float32 | Per-voxel R² of the full model |
| `R2run` | (183408, 1, 1, 11) | float32 | Per-voxel R² per run |
| `rrbadness` | (183408, 1, 1, 20) | float32 | Per-voxel ridge regression badness across 20 frac levels |
| `FRACvalue` | (183408, 1, 1) | float32 | **Per-voxel cross-validated fracridge fraction.** Mean ≈0.254 |
| `scaleoffset` | (183408, 1, 1, 2) | float32 | Per-voxel scale + offset normalization (post-fit) |
| `meanvol` | (183408, 1, 1) | float32 | Mean BOLD volume |

**Key insight from the schema**: `pcnum=0` means GLMdenoise's CV picked NO noise
components for sub-005 ses-03. The FRACvalue per-voxel mean is 0.254 (substantial
fracridge shrinkage). HRFindex varies per voxel — Stage 1 HRF library is doing
real work; Stage 2 (GLMdenoise) is not; Stage 3 (fracridge) is doing
substantial per-voxel ridge.

The 693 βs in `betasmd` correspond to all non-blank trials in chronological
order across the 11 runs of ses-03. Trial labels are reconstructed from
events.tsv.

## Local reproduction βs (RT-deployable pipeline)

In `data/rtmindeye_paper/task_2_1_betas/prereg/` we have 35+ cell outputs.
Each cell saves three files:

```
{cell}_{ses}_betas.npy        — (n_trials, 2792) float32, post-cum-z
{cell}_{ses}_trial_ids.npy    — (n_trials,) numpy.ndarray of str image_names
{cell}_{ses}_config.json      — pipeline params: K, HP, erode, hrf_model, etc.
```

**Champion cell**: `RT_paper_EoR_K7_CSFWM_HP_e1_inclz_ses-03_*`
- (770, 2792) float32 raw βs, then inclusive causal cum-z applied
- aCompCor K=7 from CSF+WM via FAST PVEs > 0.5, eroded ×1, HP-filtered
- Glover HRF, AR(1), cosine drift, high_pass=0.01

**Reference for paper Offline anchor**: `Canonical_GLMsingle_OfflineFull_ses-03_*`
- (50, 2792) float32 — already filtered to 50 special515 first-rep, post-z
- This is the canonical .npz βs projected through our finalmask + relmask

## MindEye2 checkpoints

**HF source**: `macandro96/mindeye_offline_ckpts`

**Top-level structure** (~21 GB total):
```
data_scaling_exp/        — 4039 files (sample={1..10} multi-session ckpts)
realtime-dump/           — 1610 files (rt_ft fine-tune family)
seedwise_runs_dump/      — 4775 files (5-seed seedwise dumps for various ckpts)
sub-005_ses-01_task-C_bs24_MST_rishab_MSTsplit_{1,3,7,10}_avgrepeats_finalmask_epochs_{15,45,105,150}/   — 1 last.pth each
sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_{0,3,5,7,10}_avgrepeats_finalmask_epochs_*/   — 1 last.pth each
```

The directory naming conventions encode:
- `sub-005_ses-01_task-C` — subject and training session
- `bs24` — batch size 24
- `MST` — Most Similar Templates loss variant
- `rishab_repeats` — Rishab's repeat-handling code (3 reps averaged for GT)
- `3split_N` — fold N of a 3-way train/val split (or seed N)
- `MSTsplit_N` — alternative MST-based split, fold N
- `avgrepeats` — training βs are averaged across repeats
- `finalmask` — input is finalmask-projected (2792 voxels)
- `epochs_150` — trained for 150 epochs

**Paper-faithful checkpoint** (per Rishab Discord): `sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/last.pth` — fold 10 + 150 epochs.

### `last.pth` contents (verified by direct `torch.load`)

```python
state = torch.load("last.pth", map_location="cpu", weights_only=False)
# state is a dict with these keys:
#   'epoch'                — int, epoch when saved
#   'model_state_dict'     — 149 named tensors (the model weights)
#   'optimizer_state_dict' — Adam optimizer state at save time
#   'lr_scheduler'         — LR scheduler state
#   'train_losses'         — training loss curve
#   'test_losses'          — validation loss curve
#   'lrs'                  — learning rate schedule used
```

**`model_state_dict` first few entries** (149 total, ~7-8 GB total):
```
ridge.linears.0.weight                 (1024, 2792)   — projects 2792 voxels → 1024-d shared latent
ridge.linears.0.bias                   (1024,)
backbone.mixer_blocks1.0.0.weight      (1024,)         — LayerNorm
backbone.mixer_blocks1.0.0.bias        (1024,)
backbone.mixer_blocks1.0.1.0.weight    (1024, 1024)    — Mixer layer
... (residual MLP backbone)
projector branch                       (256, 1664)     — outputs CLIP-token-shaped embedding
diffusion_prior branch                 (CLIP-image)    — outputs unCLIP-image embedding
```

**Key dimensions**:
- Input: 2792 voxels (the relmask)
- Subject latent: 1024-d after ridge layer
- Output (projector branch): 256 × 1664 token grid (matches OpenCLIP ViT-bigG/14)
- Output (diffusion_prior branch): single CLIP embedding for unCLIP reconstruction

**Note**: when we load this model with our `mindeye_retrieval_eval.M.load_mindeye(ckpt_path, n_voxels=2792)`, we see `unexpected keys: 85` warnings — these are diffusion_prior submodules that the retrieval-only forward pass doesn't need. Loading still succeeds; we just don't use the diffusion_prior branch.

### `seedwise_runs_dump/offline/sub-005_..._3split_10_..._epochs_150/{0..4}/`

For each of 5 seeds (0..4), 8 files:

| File | Size | Schema | What it is |
|---|---|---|---|
| `final_evals.csv` | <1 KB | 9 metric rows | Training-time evaluation (avg-3-rep): pixcorr, ssim, alex2/5, clip, effnet, swav, **fwd_acc, bwd_acc**. Each seed reports `fwd_acc=0.9000`, `bwd_acc=0.8800`. |
| `..._all_clipvoxels.pt` | 85 MB | Tensor (50, 256, 1664) | **Predicted CLIP voxels** — model output for the 50 test images. Post-model-averaged across 3 reps per Train.py:1066-1078 |
| `..._all_images.pt` | 39 MB | Tensor (50, 3, 256, 256) | The 50 test images at 256×256 |
| `..._all_recons.pt` | ~80 MB | Tensor (50, 3, H, W) | unCLIP-reconstructed images per test trial |
| `..._all_predcaptions.pt` | ~5 KB | List of 50 strings | GIT-generated captions for the predicted images |
| `..._all_prior_out.pt` | ~80 MB | Tensor | Diffusion-prior intermediate output |
| `..._all_clipvoxels_fwdtrain.pt` | varies | Tensor | clip_voxels on training set (for diagnostic) |
| `..._MST_ID.npy` | 5 KB | (693,) int | Per-trial MST template identifier |

**Key reading**: the `final_evals.csv` reports 0.90 / 0.88 per seed — the AVG-3-REP numbers, matching Table 1 row "Offline 3T (avg. 3 reps.)". The published `_all_clipvoxels.pt` is post-model-averaged (3 reps → 1 vector per image), so when we score it against our GT we get 88% top-1 (within 1 trial of paper's 90%).

There's no "first-rep clipvoxels" file in the seedwise dump. **The seedwise dump only contains avg-3-rep model outputs, not first-rep model outputs.** This is consistent with the canonical Train.py validation loop (which always does post-model-avg-3-rep).

So **the paper's Table 1 first-rep numbers (76%, 64%) cannot be derived from the seedwise dump's _all_clipvoxels.pt files.** They must come from a separate post-hoc script that:
- Loads the canonical .npz βs OR a different β source
- Filters to first rep only at the β level
- Runs the model forward
- Scores retrieval

We've reproduced this protocol locally and get 56-62%, not 76%. The 76% number's exact provenance remains unverified.

### `data_scaling_exp/concat_glmsingle/sub-005_all_..._sample={1..10}_avgrepeats_finalmask_epochs_150_delay=0/`

Per fold (sample=10 we tested): same 8 file types as seedwise_runs_dump, plus:
- `sanity_check_individual_reps/special_NNNNN/` per test image:
  - `all_clipvoxels.pt` (3 reps × 256 × 1664) — pre-averaged per-rep predictions
  - `all_ground_truth.pt` (3, 3, 224, 224) — paper's preprocessed input image (3 reps, identical)
  - `all_recons.pt` — per-rep reconstructions
  - `all_retrieved.pt` — top-N retrieval candidates per rep

**This is where the paper's per-image inspection data lives.** We used the
`all_ground_truth.pt` files in round 7 to verify our image preprocessing
matches paper exactly (JPG → 224 bilinear NO antialias, 0.0000 pixel diff).

The `data_scaling_exp` family has these per-image artifacts because it
includes the cross-validation runs for the "data scaling" appendix figure
(performance vs amount of fine-tuning data). The `realtime-dump` and
`seedwise_runs_dump/offline` families don't have per-image sanity-check
folders.

### `realtime-dump/` (RT-FT family, per Rishab Discord 2026-01-26)

Two sub-families:
- `sub-005-ses-01_task-C-rt_ft_split=repeats3_delay={0,63}_epochs=150_delay={0,63}/` — fine-tuned on rt_ft preprocessed βs at delay=0 (Fast) or delay=63 (EoR)
- `sub-005_ses-01_task-C-offline_ft_split=repeats3_epochs=150_delay={0,63}/` — fine-tuned on offline-preprocessed βs

**Per Rishab's clarification**: some published Table 1 numbers were
"inadvertently generated" with `rt_ft` ckpts rather than `offline_ft`. Their
intended publication path is offline_ft for all rows. The rt_ft family
exists because they explored RT-faithful training (where model trains on
rtmotion-derived βs); paper text supports offline_ft as the canonical choice.

**For our reproduction**, we use the `offline_ft` family (specifically
`sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_..._epochs_150/last.pth`
which is in the top-level dir, not under realtime-dump). DGX agent's commit
`c43b63c` documents this distinction.

## What ISN'T published (and what we'd need)

We don't know exactly how the 3T sub-005 fine-tuning data is materialized
into model input. The MindEye2 NSD training pipeline (`Train.py`) reads
test data from a WebDataset tarball at `wds/subj0X/new_test/0.tar` (line
383-384), but this is for **NSD subjects 1-8**, NOT for sub-005. `Train.py`
contains no reference to "sub-005", "3T", "special515", or "finalmask" —
those only appear in the notebooks (`final_evaluations.ipynb`).

The rt-mindeye paper's 3T fine-tuning code path isn't in the .py files we
have access to in `~/Workspace/rt_mindEye2/src/`. It's plausibly:
- A separate fine-tuning script (not in this repo or our local clone)
- A notebook-based loader that constructs training inputs at runtime from
  the canonical .npz βs + events.tsv
- A different tarball convention specific to the 3T data prep

What we DO know:
- The test images = 50 special515 in ses-03 (verified by filename listing
  vs HF `sanity_check_individual_reps/special_NNNNN/` directories)
- The fold-10 ckpt's input dim is 2792 voxels (from `ridge.linears.0.weight`
  shape (1024, 2792))
- The ckpt expects βs in the same voxel ordering as our finalmask + relmask
  projection of canonical .npz βs (because our forward pass produces matching
  outputs to seedwise dump within 1 trial)

What we CAN'T directly verify:
- Whether the training-time βs are byte-identical to the canonical .npz
- Whether the paper's first-rep evaluation uses a different β source or
  different preprocessing than our reproduction infers

Things we know match (from indirect comparison):
- ✓ Avg-3-rep retrieval: 88% (us) vs 90% (paper) — within 1 trial sampling
- ✓ The 50 test images: identical (verified via filename listing)
- ✓ Test image preprocessing: pixel-exact match
- ✓ MindEye2 forward pass: produces clipvoxels matching seedwise dump within 1 trial

What we can't directly verify:
- Whether first-rep evaluation in the paper used the same z-score policy we use
- Whether the paper's first-rep numbers came from a separate eval script not in seedwise dump
- Whether the WebDataset tarball's βs differ from canonical .npz βs in any way (e.g., re-z-scored at packing time)

## Local file paths summary

```
/Users/mhough/Workspace/data/rtmindeye_paper/
├── rt3t/data/
│   ├── sub-005_desc-preproc_T1w.nii.gz         T1 (full head) for FreeSurfer
│   ├── sub-005_desc-preproc_T1w_brain.nii.gz   skull-stripped T1
│   ├── sub-005_final_mask.nii.gz               finalmask (19174 voxels)
│   ├── sub-005_ses-01_task-C_relmask.npy       relmask (2792 voxels within finalmask)
│   ├── T1_brain_seg_pve_{0,1,2}.nii.gz         FSL FAST tissue PVEs
│   ├── T1_synthseg.nii.gz                      mri_synthseg output (33-class)
│   ├── deepmriprep_out/p{0,1,2,3}*.nii.gz      DeepMriPrep tissue probabilities
│   ├── derivatives/brainmask/sub-005/sub-005_brain.nii.gz   pre-computed for recon-all
│   ├── all_stimuli/special515/special_NNNNN.jpg  50 test images
│   ├── events/sub-005_{ses}_task-C_run-{NN}_events.tsv    trial-level events
│   ├── events/sub-005_{ses}_task-C_run-{NN}_tr_labels.csv  TR-level decode labels
│   ├── getcanonicalhrflibrary.tsv              GLMsingle 20-HRF library (501, 20)
│   ├── avg_hrfs_s1_s2_full.npy                 per-voxel HRF indices from ses-01+ses-02 (76,90,74,1)
│   ├── model/                                  MindEye2 ckpts
│   │   ├── sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth   (fold-0)
│   │   └── sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/
│   │       └── last.pth                        (fold-10, paper-faithful)
│   └── freesurfer/sub-005/                     in-progress recon-all output
├── motion_corrected_resampled/                 rtmotion BOLD (192 vols/run × 11 runs)
├── fmriprep_mindeye/data_sub-005/              fMRIPrep-preprocessed BOLD + xfms
├── glmsingle/glmsingle_sub-005_ses-03_task-C/  canonical Princeton GLMsingle output
├── glmsingle/glmsingle_sub-005_ses-03_task-C_RTMOTION/  our GLMsingle on rtmotion (1.5 GB)
├── glmsingle/fold10_canonical_seed0/           seedwise dump pulled for round 7
├── task_2_1_betas/prereg/                      our cell βs + trial_ids + configs
├── task_2_1_betas/{retrieval_results_v2,unified_metrics,k_sweep_metrics,...}.json   results
└── rt_all_data/                                additional artifacts (not currently used)
```

## How each Table 1 number is materialized

| Table 1 row | Files involved | Expression |
|---|---|---|
| Offline 3T avg-3-rep 90% | canonical .npz `betasmd` + fold-10 `last.pth` + 50 special515 jpgs + OpenCLIP weights | `model.forward(z(reps_avg(betasmd[image]))) → cos_sim with OpenCLIP(image_resized_224)`, top-1 over 50 |
| Offline 3T 1st-rep 76%* | same files; first-rep filter | filter to first occurrence per image; same forward; top-1 |
| EoR RT 1st-rep 66% | rtmotion BOLD + fold-10 `last.pth` + 50 jpgs + OpenCLIP | nilearn LSS full-run + cum-z + first-rep filter; same forward+score |
| Slow RT 1st-rep 58% | same as EoR but `streaming_decode_TR = onset_TR + delay=5_trials` | per-trial LSS on cropped BOLD, same downstream |
| Fast RT 1st-rep 36% | same with `delay=0_trials` (~7.5s post-stim) | per-trial LSS on shorter window |

*The 76% Offline 1st-rep doesn't reproduce locally with the protocol we
infer from the paper. See "Anomalies" above.

---

# Where the doc itself lives

This document is part of `m9h/hippy-feat:results/apple-silicon-2026-04-28`
on GitHub. Adjacent docs:

| Doc | Topic |
|---|---|
| `README.md` | Round-by-round ablation history (60+ KB) |
| `NEUROFEEDBACK_METRICS.md` | Why 50-way top-1 is the wrong target for closed-loop; metric stack for Norman-lab-style work |
| `VARIANT_G_TRIBEV2_DESIGN.md` | Design for the structured-prior + foundation-model extension; recon-all blockers |
| `PIPELINE.md` | This document |
| `RISHAB_LADDER_REPORT.md` | Per-tier reproduction status, prepared for sharing with paper authors |
| `GLOSSARY.md` | LSS/LSA/persistent/streaming/cross-run terminology |
| `drivers/` | All driver scripts referenced above |

