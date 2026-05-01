# Recommendations for the Princeton RT-MindEye team

**Audience**: Rishab Iyer + the team running the current RT-MindEye pilot.
**Companion**: `docs/task_2_1_for_rishab.md` (full factorial decomposition + supporting numbers).

This doc is action-oriented. The analytical evidence behind every claim is in the companion doc. Three sections:

- **A.** What we'd suggest changing in the paper.
- **B.** What we'd suggest changing in the RT pipeline for the current pilot.
- **C.** What to instrument in the pilot so the remaining open questions resolve themselves.

---

## A. Paper updates

| Current paper framing | What our data supports instead |
|---|---|
| The Offline-vs-RT 10 pp top-1 gap is "preprocessing pipeline" (fmriprep + GLMsingle) | The gap is dominantly **β-windowing** — RT's per-trial GLM sees only `onset + delay` TRs of BOLD, while Offline fits on the full session. Same data set, different β-estimation regime. The gap is structurally inherent to the RT setting, not a pipeline-feature gap. |
| GLMsingle Stages 1, 2, 3 each contribute to the offline result | On sub-005, the canonical `TYPED_FITHRF_GLMDENOISE_RR.npz` has **`pcnum = 0`** — Stage 2 (GLMdenoise) was inactive (CV picked zero PCA components). The offline lift over a Glover + AR(1) + cum-z + repeat-avg baseline is **+0 pp top-1, +4 pp top-5**, attributable to **Stages 1 + 3** only. |
| Top-1 image retrieval is the headline metric | For closed-loop deployment, **pairwise AUC** (same-image vs different-image β-distance) is the relevant metric. RT plateaus at AUC ≈ 0.826 by decode delay = 15; Offline reaches 0.886 with denoising. The 0.06 AUC delta is where the practical loss lives. |
| AR(1) frequentist GLM is the right RT noise model | Variant G's Bayesian conjugate produces a per-trial posterior `(β_mean, β_var)` at the **same forward-pass cost** as AR(1) freq (1.6–4.8 ms/TR JIT'd). It enables confidence-gated selective accuracy of **84–90 % at τ = 0.9, covering 34–51 % of trials** — a regime AR(1) freq cannot produce because it has no posterior. |

Concretely, three paper edits we would recommend:

1. **Re-frame Figure 3** as a windowing-vs-causal-evidence-window comparison rather than a pipeline-feature comparison.
2. **Add a footnote on `pcnum = 0`** for sub-005 — and rerun the same `.npz` inspection on other subjects to see whether GLMdenoise was suppressed pipeline-wide or just on this subject.
3. **Add an AUC / confidence-aware evaluation** alongside top-1 to make the closed-loop deployment relevance explicit.

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

The relmask FAST PVE files (`T1_brain_seg_pve_{0,1,2}.nii.gz`) are already on disk. Important: pull the PCs from segmented WM/CSF, **not** from a high-variance pool inside the relmask — relmask voxels are task-driven by construction. The paper's `pcnum = 0` outcome and our independent K = 10 EoR test (rejected, hurt by 6 pp) both warn about this failure mode.

### 4. Fieldmap-based Susceptibility Distortion Correction (SDC) at scan start

SDC corrects EPI geometric distortion at air/tissue interfaces (orbitofrontal cortex, ventral temporal lobe, brainstem). fmriprep does it offline; RT pipelines typically don't.

Operationally: a ~2-minute fieldmap acquisition at scan start → compute the per-voxel warp during the structural pre-scan setup (a few minutes, before the first task run) → apply per TR via `fsl applywarp` (~50 ms/TR, well within the TR budget).

For RT-MindEye-style visual-cortex retrieval the payoff is bounded (~1–3 pp top-1) because visual cortex is not a high-distortion region. But it's free signal once the fieldmap is collected — and it's the only fmriprep stage strictly absent from the RT pipeline.

### 5. Don't apply fracridge as a post-hoc wrapper

Per-voxel SVD-based fracridge (the canonical Stage 3) **only works when the decoder is fine-tuned on fracridge βs**. As a shim around an RT-pipeline OLS or AR(1) β it tanks retrieval (22–43 % top-1 measured) due to per-voxel pattern distortion that the frozen MindEye ridge layer wasn't trained on.

If you're not running canonical GLMsingle end-to-end, leave fracridge out.

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
> The 10 pp top-1 Offline-vs-RT gap is dominantly windowing — physically inherent to RT, not a missing-stage problem. The deployable target is **closing the AUC gap**, which can be done in stages: VG + confidence-gating (paradigm change), then aCompCor (~+0.10 AUC), then SDC (~+1–3 pp top-1, smaller AUC).

---

## Cross-references

- Full factorial decomposition: `docs/task_2_1_for_rishab.md`
- Pre-registration: `TASK_2_1_PREREGISTRATION.md`
- Amendment (windowing-axis re-frame): `TASK_2_1_AMENDMENT_2026-04-28.md`
- Findings table: `TASK_2_1_FINDINGS.md`
