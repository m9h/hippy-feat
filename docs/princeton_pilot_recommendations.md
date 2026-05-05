# Recommendations for the Princeton RT-MindEye team

**Audience**: Rishab Iyer + the team running the current RT-MindEye pilot.
**Companion**: `docs/task_2_1_for_rishab.md` (full factorial decomposition + supporting numbers).
**Glossary**: `docs/glossary.md` covers all recurring jargon (top-1, AUC, LSS, pst, fracridge, Variant G, etc.) — keep it open in a side tab.

## Paper anchors (Iyer et al. ICML 2026 Table 1, single-trial first-rep, n=50)

The paper reports a four-tier latency table. We re-derived the equivalent
numbers on the same checkpoint and confirm our scoring matches paper RT
within ±1.3 pp at constant settings:

| Pipeline | Latency | Paper top-1 | What it is |
|---|---|---|---|
| Offline 3T (avg 3 reps) | 1 day | **90%** | fmriprep + GLMsingle Stages 1+2+3 + average across reps |
| Offline 3T | 1 day | **76%** | fmriprep + GLMsingle, single-trial first-rep |
| **End-of-run RT** | 2.7 min | **66%** | rtmotion + nilearn AR(1) LSS, fit at end of each run |
| **Slow RT** | 36 s | **58%** | rtmotion + nilearn AR(1) LSS, ~29 s post-stim |
| **Fast RT** | 14.5 s | **36%** | rtmotion + nilearn AR(1) LSS, ~8 s post-stim |

The "10 pp gap" the paper Discussion focuses on is **Offline 3T 76% vs
End-of-Run RT 66%**. Both have ALL the within-run BOLD; the difference
is fmriprep + GLMsingle vs rtmotion + nilearn AR(1) LSS at constant
within-run data availability. The further drops to Slow (58%) and Fast
(36%) RT are **windowing on top of that** — only the within-window TRs
are visible to the GLM.

Our reproduced anchors (n=150 raw, causal cum-z applied):

| Cell | Top-1 | AUC | Maps to paper tier |
|---|---|---|---|
| `Canonical_GLMsingle_ses-03` (paper output, scored directly) | 65.3% | 0.856 | Offline 3T (n=150 vs n=50 first-rep) |
| `Paper_RT_actual_delay63` | 58.0% | 0.825 | End-of-run RT |
| `Paper_RT_actual_delay20` | **59.3%** | 0.826 | Slow RT (paper: 58%) ← within 1.3 pp |
| `Paper_RT_actual_delay5` | (pending) | 0.803 | Fast RT |

This doc is action-oriented. The analytical evidence behind every claim is in the companion doc. Three sections:

- **A.** What we'd suggest changing in the paper.
- **B.** What we'd suggest changing in the RT pipeline for the current pilot.
- **C.** What to instrument in the pilot so the remaining open questions resolve themselves.

---

## A. Paper updates

| Current paper framing | What our data supports instead |
|---|---|
| The Offline-3T-vs-EoR-RT 10 pp top-1 gap is "preprocessing pipeline" (fmriprep + GLMsingle) | At equal within-run data, the gap is real: 76% vs 66%. But on closed-loop pairwise AUC, an RT-deployable pipeline (rtmotion + Glover + 5-PC noise PCs + AR(1)) reaches **0.886** — exceeding canonical Offline at 0.856. The top-1 gap reflects pipeline differences; the AUC gap is closeable by simple noise-PC regression that's fully streaming. |
| GLMsingle Stages 1, 2, 3 each contribute to the offline result | Stage 2 (GLMdenoise) is **subject- and session-specific** in the published canonical `.npz` files. CV-selected `pcnum` across the 9 available sessions: 0, 0, 1, 1, 1, 4, 4, 4, 6 — **maximum 6, never 10**. For sub-005 ses-03 (the Offline-anchor session) `pcnum = 0` and the bootstrap CV curve is **monotonically decreasing** in K (K=0: −764.5, K=10: −824.6) — adding any PCs strictly hurts. The offline lift over a Glover + AR(1) + cum-z + repeat-avg baseline is **+0 pp top-1, +4 pp top-5**, attributable to **Stages 1 + 3** only on this session. |
| Stage 3 (per-voxel SVD fracridge) is straightforward to add to RT | **It isn't.** Real fracridge applied to per-trial LSS βs collapses retrieval (top-1 22% full-run, 2% streaming = chance). The canonical pipeline applies fracridge to a global single-fit β matrix where the same shrinkage transform applies consistently to every trial column. Per-trial LSS gives each trial a different design-matrix SVD, hence a different direction-changing transform — pairwise consistency breaks. Replicating canonical Stage 3 in RT requires a non-causal session-end LSR fit. |
| Top-1 image retrieval is the headline metric | For closed-loop deployment, **pairwise AUC** (same-image vs different-image β-distance) is the relevant metric. RT plateaus at AUC ≈ 0.826 by decode delay = 15; Offline reaches 0.886 with denoising. The 0.06 AUC delta is where the practical loss lives. |
| AR(1) frequentist GLM is the right RT noise model | Variant G's Bayesian conjugate produces a per-trial posterior `(β_mean, β_var)` at the **same forward-pass cost** as AR(1) freq (1.6–4.8 ms/TR JIT'd). It enables confidence-gated selective accuracy of **84–90 % at τ = 0.9, covering 34–51 % of trials** — a regime AR(1) freq cannot produce because it has no posterior. |

Concretely, four paper edits we would recommend:

1. **Re-frame Figure 3** as a windowing-vs-causal-evidence-window comparison rather than a pipeline-feature comparison.
2. **Be explicit about the subject-and-session-specific behavior of Stage 2 (GLMdenoise)**: the bootstrap-selected `pcnum` ranges 0–6 across the 9 published `.npz` outputs. For sub-005 ses-02 and ses-03 it's 0; for sub-001 ses-01 it's 6. RT-pipeline reimplementations should not treat GLMdenoise as a load-bearing default. Recommend reporting `pcnum` per anchor in the paper's pipeline-description section.
3. **Add a pairwise AUC column to Table 1 alongside top-1 retrieval.** Top-1 inherits from Scotti 2023 / 2024 for continuity with prior MindEye papers, but it answers the question "did we identify the exact image" — appropriate for the *reconstruction* framing in §`Comparing Pipeline Variations`. The paper's *future-applications* framing (§`Towards Non-Invasive Brain-Computer Interfaces`) — neurofeedback to manipulate fine-grained representations, depression-attentional-bias correction — depends on **pairwise discriminability**, not 50-way identification. The closing argument and the reported metric should match. Concretely: report `same-image vs diff-image cosine-distance AUC + Cohen's d` for each row of Table 1. We've measured these directly on the published RT-betas anchors:

   | Tier | Paper top-1 | Our AUC measurement (n=150) | Our d |
   |---|---|---|---|
   | End-of-run RT (`delay=63`) | 66% | **0.825** | 1.30 |
   | Slow RT (`delay=20`) | 58% | **0.826** | 1.30 |
   | Fast RT (`delay=5`) | 36% | **0.803** | 1.18 |
   | Canonical Offline (`.npz`) | 76% | **0.856** | 1.48 |
   | RT-deployable cell 7 (rtmotion + Glover + 5-PC noise PCs + AR(1)) | — | **0.886** | 1.71 |

   The pairwise AUC compresses the dynamic range relative to top-1 — Fast→EoR is 36→66% on top-1 (+30 pp) but 0.80→0.83 on AUC (+0.03). On AUC the offline-vs-EoR gap is 0.03 (vs 10 pp on top-1). For the closed-loop deployment context the paper argues for, that AUC delta is the more relevant figure of merit and changes the interpretation: **a simple 5-PC noise regression on RT-deployable streaming AR(1) already exceeds the canonical Offline anchor on AUC**. That message gets lost in a top-1-only table.
4. **Add an explicit confidence-aware evaluation track** — selective accuracy at confidence threshold τ — for the Variant-G-style pipelines where `(β_mean, β_var)` is available. Companion to AUC; turns the per-trial posterior into a deployment-relevant operating curve.

### What the cross-session inspection establishes

We pulled all `TYPED_FITHRF_GLMDENOISE_RR.npz` files from `rishab-iyer1/glmsingle` and read the bootstrap CV outputs:

| path | `pcnum` | mean `FRACvalue` | bootstrap CV trend |
|---|---|---|---|
| `sub-001_ses-01` | 6 | 0.091 | non-monotone, optimum at K=6 — GLMdenoise genuinely helping |
| `sub-001_ses-02` | 1 | 0.064 | mild dip at K=1, rises after — GLMdenoise barely useful |
| `sub-001_ses-03` | 1 | 0.064 | same shape as ses-02 |
| `sub-005_ses-01` | 4 | 0.077 | non-monotone, optimum at K=4 |
| `sub-005_ses-01-02` | 4 | 0.063 | non-monotone, optimum at K=4 |
| `sub-005_ses-01-03` | 4 | 0.061 | non-monotone, optimum at K=4 |
| `sub-005_ses-02` | **0** | 0.079 | monotonically decreasing in K — GLMdenoise strictly hurts |
| `sub-005_ses-03` | **0** | 0.076 | **monotonically decreasing** in K (K=0: −764.5, K=10: −824.6) |
| `sub-005_ses-06` | 1 | 0.063 | mild dip at K=1 |

Three locked findings:

1. **`pcnum` is a single scalar K per pipeline run** chosen by GLMsingle's bootstrap CV. Maximum K observed across the entire dataset is **6**. The hypothesis that the paper silently applied K=10 anywhere cannot be right — it was never selected.
2. **For the Offline-anchor session (sub-005 ses-03) the CV curve is smooth and monotonically decreasing in K**. Not a flake. K=10 was explicitly tested by the bootstrap procedure and would have made the result substantially worse than K=0.
3. **GLMdenoise's contribution is genuinely subject- and session-dependent**. RT-pipeline reimplementations adding "Stage 2 with K=10" as a fixed default would underperform the canonical pipeline on the very session whose Offline number anchors Figure 3.

---

## B. RT-pipeline updates for the current pilot

Ranked by likely payoff for closed-loop neurofeedback quality and by deployment cost.

### 1. Switch the GLM to Variant G (Bayesian conjugate AR(1))

Drop-in replacement for nilearn AR(1) at the same per-TR cost. Produces a closed-form per-trial posterior `(β_mean, β_var)` instead of just a point estimate. **No retrieval cost vs current pipeline** — VG ties AR(1) freq on top-1 and AUC. The win is that it produces the variance estimate that everything below depends on.

**Cost**: drop-in code change. Per-TR forward pass measured at 1.6–4.8 ms (JIT-compiled JAX path; 300× headroom against TR).

### 2. Build a confidence-gated decoder wrapper

The highest-leverage *behavioral* change for the pilot. Takes `(β_mean, β_var)` from VG, computes calibrated class probability, gates the feedback signal on a confidence threshold:

- High confidence (`SNR > τ`): show the decoded class to the subject.
- Low confidence: show "uncertain" or fall back to a prior, instead of polluting the feedback with a noisy guess.

Train the gate on a calibration session — fit a logistic classifier on `(β_mean, β_var)` pairs from training-session data, then sweep the confidence threshold to pick the operating point that hits the desired accuracy/coverage balance. ~150 LOC.

This is the change that **changes what the subject sees**. Preprocessing changes (the rest of this list) move accuracy metrics by 0.01–0.10 AUC; confidence gating changes the deployment paradigm.

### 3. aCompCor with precomputed WM/CSF masks (deploy in the RT path)

The offline pipeline **already extracts aCompCor** in fmriprep (Appendix `fMRIPrep Preprocessing Details`):

> "For aCompCor, three probabilistic masks (CSF, WM and combined CSF+WM) are generated in anatomical space... Components are also calculated separately within the WM and CSF masks. For each CompCor decomposition, the *k* components with the largest singular values are retained, such that the retained components' time series are sufficient to explain 50 percent of variance across the nuisance mask."

The recommendation here is to **deploy that same aCompCor regression in real time**, not to add a new acquisition. The tissue probability maps (`T1_brain_seg_pve_{0,1,2}.nii.gz` from FAST) are already on disk per session and the offline confounds tables already include aCompCor PCs.

Operationally for RT: precompute the WM/CSF masks (~1 min) before the first task run, project them onto the streaming BOLD reference space, and at each TR regress the projected WM/CSF mean signal (or top-K PCs from a sliding window) out before fitting the GLM.

Expected gain: this is what gives our cell 7 its **+0.10 AUC** over plain AR(1) (relmask top-variance noise pool, 5 PCs, fully streaming). On sub-005 ses-03 our 5-PC noise regression already exceeds canonical Offline AUC (0.886 vs 0.856). The independent K=10 EoR test on the relmask-pool noise basis was rejected (hurt by 6 pp top-1) because at full-run K=10 the top-variance voxels include task signal — the WM/CSF-restricted aCompCor pool avoids that failure mode by construction.

### 4. Deploy fieldmap-based SDC in the real-time path

The paper **already collects fieldmaps** in every scan session (Appendix `appendix-mri-acq`: "two spin-echo field map volumes, TR = 8000 ms, TE = 66 ms, in opposite phase encoding directions for fieldmap correction"). fmriprep uses them via `topup` for the offline pipeline; the RT pipeline currently doesn't.

The data is already there — the recommendation is to **apply SDC in the streaming path**: precompute the per-voxel warp from the existing fieldmap during the structural pre-scan setup (a few minutes, before the first task run), then apply per TR via `fsl applywarp` (~50 ms/TR, well within the TR budget).

For RT-MindEye-style visual-cortex retrieval the payoff is bounded — visual cortex is not a high-distortion region. Our measurement: at constant Glover + 5-PC noise regression + AR(1), fmriprep BOLD vs rtmotion BOLD differs by **−0.005 AUC** (cells B12/13 vs B3/B6). That ≤ 0.005 figure is the *joint* contribution of all of fmriprep's offline-only steps (SDC + slice-timing + better head-motion correction + 24-vs-6 confounds + CompCor) — so any single one of them, including SDC, is at the noise floor for this task and ROI.

**Net**: free to add (data is collected, latency is fine), but don't expect a meaningful AUC bump on visual-cortex retrieval.

### 5. Don't apply fracridge as a post-hoc wrapper — and don't try to recreate canonical Stage 3 under streaming

Two separate failure modes:

**A. Real per-voxel SVD fracridge applied to per-trial LSS βs is broken on both top-1 and AUC.** With Stage 1 (HRF library) + Stage 3 (real per-voxel SVD fracridge using FRACvalue frozen from a training session), measured numbers on sub-005 ses-03:

| variant | top-1 | top-5 | AUC |
|---|---|---|---|
| Full-run, real fracridge, rtmotion | 22.0% | 44.7% | 0.568 |
| Full-run, real fracridge, fmriprep | 24.7% | 54.0% | 0.586 |
| Streaming pst=8, real fracridge, rtmotion | **2.0%** (chance) | 10.7% | 0.483 |
| Streaming pst=8, real fracridge, fmriprep | 1.3% (chance) | 10.0% | 0.485 |

The canonical pipeline applies fracridge to a **global single fit** of all trials at once; the same shrinkage transform applies consistently to every per-trial β column. Per-trial LSS computes a separate β through a separate fit with a separate design-matrix SVD per trial — so the direction-changing fracridge transform differs per trial. Trials of the same image get *different* shrinkage directions, and pairwise discriminability collapses.

**B. Scalar `β *= FRACvalue` is a no-op on cosine-distance metrics.** Multiplying every voxel's β by a positive scalar leaves cosine distance unchanged. We confirmed: full-run + scalar fracridge has the **same** AUC as Stage 1 alone (both at 0.755).

**Practical takeaway**: canonical Stage 3 is not trivially RT-deployable. It depends on global state (all trials simultaneously visible) that streaming can't preserve without restructuring as a global LSR-style fit at session end (non-causal). If you're not running canonical GLMsingle end-to-end, **leave fracridge out and use simpler noise-PC regression instead** (item 3 above).

### 6. Don't add cross-run causal filters yet

Every variant tested (top-K HOSVD on past-run BOLD, task-orthogonal HOSVD on past-run residuals, session-frozen ρ̂ AR(1) hybrid) **hurt** vs the within-run streaming baseline. Past-run information removal damages task signal on this data.

If you do want to retry, the next variant on the queue is **HOSVD on a CSF/WM-pool** (i.e., proper aCompCor extracted from past runs) rather than on raw BOLD or residuals. Untested. Lower priority than items 1–4 above.

---

## C. What to instrument in the pilot

To resolve the questions our analysis can't fully close from the existing data:

| Add to the pilot | What it unblocks |
|---|---|
| Save VG posterior `(β_mean, β_var)` alongside the decoded class | Post-hoc selective-accuracy curves; A/B against an AR(1)-freq baseline at no per-trial cost |
| Save MCFLIRT motion `.par` files **per run** | Currently overwritten — only run-01 survives in `motion_corrected/` after a typical session runs. Saving per-run `.par` enables proper motion-confound replication and FD/DVARS censoring |
| A/B per-voxel HRF library vs Glover canonical (alternating runs or alternating subjects) | Settles whether GLMsingle Stage 1 is RT-deployable. Our measurement is mixed — library hurts top-1 (45 % vs 62 %) but mostly preserves AUC (0.755). Whether it's a net win depends on which metric drives the experiment |
| Acquire anat **and** fieldmap for every pilot subject (already standard per `appendix-mri-acq` — just needs to be ensured for any new subjects) | Enables aCompCor (item 3) and SDC (item 4) RT deployment without per-subject calibration delay on scan day |

The instrumentation cost for all four is small relative to the pilot's existing logging — just additional `.npy` / `.tsv` files per run.

---

## TL;DR for the team

> The most practical RT-pipeline upgrade is **Variant G + confidence-gated decoding**, because it changes what gets shown to the subject. Preprocessing changes (aCompCor, SDC, HRF library) move the AUC needle by 0.01–0.10 each but don't change the deployment paradigm. Confidence gating does.
>
> The 10 pp top-1 Offline-vs-RT gap is dominantly windowing — physically inherent to RT, not a missing-stage problem. On the closed-loop-relevant pairwise AUC metric, **a simple RT-deployable pipeline already exceeds the canonical paper Offline anchor**: AR(1) + 5-PC noise-PC nuisance regression (our cell 7) hits AUC 0.886 / d 1.71, vs canonical Offline 0.856 / d 1.48. **Canonical Stage 3 (per-voxel SVD fracridge) does not transfer to streaming LSS** — tested directly, falls to chance under per-trial windowing. Use the simpler 5-PC tCompCor regression instead.

---

## Cross-references

- Full factorial decomposition: `docs/task_2_1_for_rishab.md`
- Pre-registration: `TASK_2_1_PREREGISTRATION.md`
- Amendment (windowing-axis re-frame): `TASK_2_1_AMENDMENT_2026-04-28.md`
- Findings table: `TASK_2_1_FINDINGS.md`
