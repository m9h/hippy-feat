# Recommendations for the Princeton RT-MindEye team

**Audience**: Rishab Iyer + the team running the current RT-MindEye pilot.
**Companion**: `docs/task_2_1_for_rishab.md` (full factorial decomposition + supporting numbers).
**Glossary**: `docs/glossary.md` covers all recurring jargon (top-1, AUC, LSS, pst, fracridge, Variant G, etc.) — keep it open in a side tab.

This doc is action-oriented. The analytical evidence behind every claim is in the companion doc. Three sections:

- **A.** What we'd suggest changing in the paper.
- **B.** What we'd suggest changing in the RT pipeline for the current pilot.
- **C.** What to instrument in the pilot so the remaining open questions resolve themselves.

---

## A. Paper updates

| Current paper framing | What our data supports instead |
|---|---|
| The Offline-vs-RT 10 pp top-1 gap is "preprocessing pipeline" (fmriprep + GLMsingle) | The gap is dominantly **β-windowing** — RT's per-trial GLM sees only `onset + delay` TRs of BOLD, while Offline fits on the full session. Same data set, different β-estimation regime. The gap is structurally inherent to the RT setting, not a pipeline-feature gap. |
| GLMsingle Stages 1, 2, 3 each contribute to the offline result | Stage 2 (GLMdenoise) is **subject- and session-specific** in the published canonical `.npz` files. CV-selected `pcnum` across the 9 available sessions: 0, 0, 1, 1, 1, 4, 4, 4, 6 — **maximum 6, never 10**. For sub-005 ses-03 (the Offline-anchor session) `pcnum = 0` and the bootstrap CV curve is **monotonically decreasing** in K (K=0: −764.5, K=10: −824.6) — adding any PCs strictly hurts. The offline lift over a Glover + AR(1) + cum-z + repeat-avg baseline is **+0 pp top-1, +4 pp top-5**, attributable to **Stages 1 + 3** only on this session. |
| Top-1 image retrieval is the headline metric | For closed-loop deployment, **pairwise AUC** (same-image vs different-image β-distance) is the relevant metric. RT plateaus at AUC ≈ 0.826 by decode delay = 15; Offline reaches 0.886 with denoising. The 0.06 AUC delta is where the practical loss lives. |
| AR(1) frequentist GLM is the right RT noise model | Variant G's Bayesian conjugate produces a per-trial posterior `(β_mean, β_var)` at the **same forward-pass cost** as AR(1) freq (1.6–4.8 ms/TR JIT'd). It enables confidence-gated selective accuracy of **84–90 % at τ = 0.9, covering 34–51 % of trials** — a regime AR(1) freq cannot produce because it has no posterior. |

Concretely, three paper edits we would recommend:

1. **Re-frame Figure 3** as a windowing-vs-causal-evidence-window comparison rather than a pipeline-feature comparison.
2. **Be explicit about the subject-and-session-specific behavior of Stage 2 (GLMdenoise)**: the bootstrap-selected `pcnum` ranges 0–6 across the 9 published `.npz` outputs. For sub-005 ses-02 and ses-03 it's 0; for sub-001 ses-01 it's 6. RT-pipeline reimplementations should not treat GLMdenoise as a load-bearing default. Recommend reporting `pcnum` per anchor in the paper's pipeline-description section.
3. **Add an AUC / confidence-aware evaluation** alongside top-1 to make the closed-loop deployment relevance explicit.

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

### 3. aCompCor with precomputed WM/CSF masks

Pre-collected anat → tissue segmentation → 5-PC noise regression per TR. Expected **+~0.10 AUC** over plain AR(1). RT-deployable in two ways:

- **Slow but standard**: FreeSurfer aseg from a prior session (offline), reused on scan day.
- **Same-day**: FastSurfer ~60 s GPU pass on the scan-day anat — gets the WM/CSF mask before the first functional run.

The relmask FAST PVE files (`T1_brain_seg_pve_{0,1,2}.nii.gz`) are already on disk. Important: pull the PCs from segmented WM/CSF, **not** from a high-variance pool inside the relmask. The independent K=10 EoR test on a relmask-pool noise basis was rejected (hurt by 6 pp) because relmask voxels are task-driven by construction. This is consistent with the canonical pipeline's bootstrap CV preferring K=0 on sub-005 ses-02/03 — the noise pool selected by the canonical `.npz` (`pcvoxels` and `noisepool` fields) is more anatomically restricted than a top-variance relmask pool, which is why GLMdenoise can be useful on other sessions but is the wrong default for RT.

### 4. Fieldmap-based Susceptibility Distortion Correction (SDC) at scan start

SDC corrects EPI geometric distortion at air/tissue interfaces (orbitofrontal cortex, ventral temporal lobe, brainstem). fmriprep does it offline; RT pipelines typically don't.

Operationally: a ~2-minute fieldmap acquisition at scan start → compute the per-voxel warp during the structural pre-scan setup (a few minutes, before the first task run) → apply per TR via `fsl applywarp` (~50 ms/TR, well within the TR budget).

For RT-MindEye-style visual-cortex retrieval the payoff is bounded (~1–3 pp top-1) because visual cortex is not a high-distortion region. But it's free signal once the fieldmap is collected — and it's the only fmriprep stage strictly absent from the RT pipeline.

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
| Pre-collect anat **and** fieldmap before pilot day for every subject | Enables aCompCor (item 3) and SDC (item 4) without per-subject calibration delay on scan day |

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
