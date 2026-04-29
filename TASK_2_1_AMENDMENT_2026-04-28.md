# Task 2.1 — Pre-registration Amendment

**Locked**: 2026-04-28 (supersedes 2026-04-27 lock in `TASK_2_1_PREREGISTRATION.md`)
**Author**: Morgan G. Hough
**Status**: Variant matrix re-locked around the **windowing axis**, the
actual mechanism behind the paper's 10pp Offline-vs-RT gap.

## Why amend

The 2026-04-27 lock organized the 12-cell variant sweep around motion source
× HRF × noise model × denoising. Today's empirical work (results branch
`results/apple-silicon-2026-04-28`, commits 0ece7a1..2f96057, plus DGX
agent commits 9511fe6/78f712a/e462681) showed:

1. **Cell 11 was using full-run BOLD**, hiding the entire RT-vs-Offline gap.
   Once cell 11 was switched to streaming pst=8, top-1 dropped from 74% to
   68% and the paper's 10pp gap became visible (8pp on this checkpoint).
2. **Of the 8pp Mac gap, ~6pp is windowing alone**; only ~2pp is the
   motion source. GLMsingle Stages 1-3 contributed 0pp on this checkpoint
   (cell 12 hits 76% paper-exact without them).
3. The actual scientific question — *what is behind the difference between
   offline (whole session) and RT (segment)* — is overwhelmingly
   "windowing", not "preprocessing pipeline". Within "windowing", the
   actionable subquestion (Ernest Lo's task) is **how much of the
   offline lift can be recovered causally by stacking data across past runs**.

The original 12 cells were organized along axes that don't carry the
weight. They need to be re-categorized along the windowing axis, with new
cells specifically for the cross-run question.

## Three regimes (the new primary axis)

| Regime | Window | Cross-run access | Real-time deployable |
|---|---|---|---|
| **A: Offline reference** | Full session | Non-causal (all 11 runs) | No |
| **B: Within-run streaming** | pst≤T_run, current run only | None | Yes |
| **C: Cross-run streaming** | pst≤T_run + accumulated past runs (causal) | Yes (runs 1..r-1) | **Yes — the goal** |

The paper's anchors:
- Paper Offline = Regime A on fmriprep BOLD with GLMsingle Stages 1-3 → 76%
- Paper RT = Regime B on rtmotion BOLD with nilearn AR1 + MCFLIRT confounds + cum-z + repeat-avg → 66%
- Δ = 10pp gap

The Task 2.1 deliverable: **show how much of that 10pp comes from each
mechanism, with priority on whether it can be recovered in Regime C**
(deployable real-time).

## Re-categorization of existing cells

| # | Cell name | Original framing | Actual regime | Status under amendment |
|---|---|---|---|---|
| 1 | `OLS_glover_rtm` | RT baseline | A (full-run) | **Reframe**: rename to `OLS_glover_rtm_FULLRUN`; needs Regime B/C variants |
| 2 | `AR1freq_glover_rtm` | RT GLM choice (H1) | A (full-run) | **Reframe**: rename `..._FULLRUN`; needs B/C variants |
| 3 | `AR1freq_glover_rtm_nilearn` | nilearn parity for #2 | A (full-run) | **Keep as a sanity-check cell**, not a Task 2.1 deliverable |
| 4 | `VariantG_glover_rtm` | Bayesian GLM (H2) | A (full-run) | **Reframe**: rename `..._FULLRUN`; needs B/C variants |
| 5 | `VariantG_glover_rtm_prior` | empirical-Bayes prior (H3) | A (full-run) | **Discard**: H3 not supported (Δ=-0.0009); cell adds no info beyond #4 |
| 6 | `AR1freq_glmsingleS1_rtm` | per-voxel HRF library | A (full-run) | **Defer**: cell is a known bug case (44% top-1); needs HRF-index alignment fix before re-run |
| 7 | `AR1freq + GLMdenoise + fracridge` | within-run denoising (H4) | A (full-run) | **Reframe**: rename `..._FULLRUN`; the right Regime C version uses across-run HOSVD/TT, not within-run PCA |
| 8 | `VariantG + GLMdenoise + fracridge` | (H4b) | A (full-run) | Same as #7 |
| 9 | `VariantG + aCompCor` | (H4c) | A (full-run) | Same as #7 |
| 10 | `RT_paper_replica_partial` | paper RT minus repeat-avg | A (full-run BOLD) | **Reframe**: rename `..._FULLRUN`; this was never RT-quality |
| 11 | `RT_paper_replica_full` | paper RT canonical | B at pst=8 only after fix | **Anchor B** — `RT_paper_replica_full_streaming_pst8` is the correct paper-RT replica; the full-run version drops |
| 12 | `Offline_paper_replica_full` | paper Offline canonical | A (correctly) | **Anchor A** — keep as canonical paper Offline |
| 13 | `EKF_streaming_glover_rtm` | streaming Kalman | B with state-reset bug | **Discard**: paradigm bug acknowledged in commit 78f712a |
| 14 | `HOSVD_denoise_AR1freq_glover_rtm` | across-run filter | B (per-run, not across-run) | **Discard**: implementation diverges from intent (Morgan: was meant to be Regime C cross-run filter; cell 14 is per-run only) |
| 15 | `Riemannian_prewhiten_AR1freq_glover_rtm` | geometric prewhitening | A, rank-deficient | **Discard**: documented negative result |
| 16 | `(prereg_online_ekf_cell)` | truly online streaming EKF | C-adjacent (one-pass session) | **Reframe**: parameterization had identifiability issues; closest to Regime C in spirit, needs paradigm rebuild |
| 17 | `HybridOnline_AR1freq_glover_rtm` | session-wide ρ + per-trial β | C for noise model only; β still uses full run | **Half-Regime-C**: only the AR(1) ρ is cross-run; β fit is still A. Needs full-Regime-C variant |
| 20 | `LogSig_AR1freq_glover_rtm` | sliding-window log-sig features | B (within-window) | **Reframe**: as currently implemented, is within-run; cross-run log-sig variant is potentially interesting but not implemented |

## Locked new cells (minimum set to answer the windowing question)

These are the cells the amendment locks. Each tests a specific
windowing-related claim. All run on rtmotion BOLD (RT-deployable) with
nilearn AR(1) + MCFLIRT confounds + cum-z + repeat-avg = the paper RT
pipeline minus the BOLD-windowing change.

| New cell | Window | Cross-run filter | Tests |
|---|---|---|---|
| `RT_paper_replica_FULLRUN` | full-run BOLD | none | (= the cell 11 bug rerun: how much does NO windowing buy at constant motion+GLM?) |
| `RT_paper_replica_streaming_pst8` (existing) | pst=8 | none | **Regime B anchor — paper RT** |
| `RT_paper_replica_streaming_pst{4,6,10}` | pst sweep | none | Ernest's "stimulus delay" axis (within-run) |
| `RT_streaming_pst8 + HOSVD_template_K=K0` | pst=8 | top-K HOSVD of {runs 1..r-1} as confound regressors | **Regime C — does cross-run filter close the gap?** |
| `RT_streaming_pst8 + HOSVD_template_K=K1` | pst=8 | top-K HOSVD truncation sweep | sensitivity of Regime C to rank K |
| `RT_streaming_pst8 + LogSig_TT_template` | pst=8 | depth-2 log-sig of cumulative cross-run BOLD as confound | streaming primitive variant of Regime C |
| `RT_streaming_pst8 + ARsession_HybridOnline` | pst=8 | session-wide ρ̂ frozen from past runs (cell 17 corrected) | how much of cross-run lift is just AR(1) parameter estimation |

## Locked hypotheses (revised H1–H5)

The 2026-04-27 H1-H5 framed gains in absolute β-reliability terms; they
remain *valid statements* but they're testing the wrong variable for
Task 2.1's deliverable. Revising:

- **H1' (windowing dominance)**: `Δ_window` (Regime A − Regime B at constant motion+GLM) accounts for ≥ 70% of the Offline-vs-RT gap measured against `Offline_paper_replica_full`. **Mac data so far: 6/8 = 75% ✓ provisional**.
- **H2' (motion residual)**: `Δ_motion` (fmriprep − rtmotion at constant window) accounts for ≤ 30% of the gap. **Mac data: 2/8 = 25% ✓ provisional**.
- **H3' (cross-run recovery)**: a Regime C cell with HOSVD top-K cross-run filter recovers ≥ 50% of `Δ_window` while remaining causal/RT-deployable. **Untested — the actual deliverable**.
- **H4' (within-run pst saturation)**: top-1 retrieval as a function of `post_stim_TRs` saturates by pst=10 within a single run. **Mac data: pst=8 → 68%, pst=10 → 68%, ✓ provisional**.
- **H5' (preprocessing residual)**: GLMsingle Stages 1-3 contribute < 2pp to Offline anchor at the canonical-Glover + nilearn AR(1) baseline. **Mac data: 76% with vs without GLMsingle Stages 1-3 = 0pp difference at top-1, +2pp at top-5; provisional ✓ for top-1, contradicts at top-5**.

## What gets discarded

For the Task 2.1 deliverable specifically, the following cells provide
no Task 2.1 information and are dropped from the active matrix:

- Cell 5 (`VariantG_glover_rtm_prior`): H3 not supported.
- Cell 13 (`EKF_streaming`): paradigm bug.
- Cell 14 (`HOSVD_denoise`): implementation ≠ intent.
- Cell 15 (`Riemannian_prewhiten`): rank-deficient negative result.

These betas remain on disk and in the JSON outputs as historical
artifacts; they're just not Task 2.1 evidence.

## Discord / DGX hand-off

Once the locked new cells run on Mac (small set, ~1 hour total), the DGX
agent should mirror them with the canonical `sample=10_..._epochs_150`
checkpoint. The expected reproduction:
- Mac and DGX `RT_paper_replica_streaming_pst8` should match within ±2pp.
- Both should match paper RT 66% within ±3pp on the canonical checkpoint
  (Mac currently at 68% on the non-canonical ses-01 checkpoint).
- The cross-run Regime C cells are **net-new science** — neither
  platform has prior data; the question is whether H3' holds.
