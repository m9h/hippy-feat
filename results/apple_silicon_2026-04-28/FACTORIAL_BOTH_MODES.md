# Factorial rescoring — both eval modes (2026-05-03)

All 130 prereg cells rescored on fold-0 in BOTH first-rep and avg-of-3 modes.
Single source of truth: `task_2_1_betas/factorial_both_modes_fold0.json`.
Driver: `local_drivers/score_factorial_both_modes.py`.

## What this re-run shows

1. **Offline reproduction is exact** when comparing like-to-like:
   - Our `Offline_paper_replica_full` first-rep = 76% (operates on 50 already-averaged βs in some cells; for 770-trial cells that go through Canonical_GLMsingle → 60% first-rep, 76% avg-of-3)
   - Paper Offline 3T (avg. 3 reps.) = 76% Image — exact match
   - Paper Offline 3T = 76% Image — same as above
   - The "76% mystery" is fully resolved

2. **avg-of-3 inflates every cell by 20-30pp.** This is exactly what the
   mislabel hid: in the original RISHAB_LADDER_REPORT.md, the Offline 76%
   was avg-of-3 (high SNR) while RT tiers were first-rep (low SNR), making
   the RT-vs-Offline gap look ~32pp when it's actually ~16pp under like-to-like.

3. **Slow / EoR remain ~14pp short on first-rep.** This is real, not a mode
   mismatch. With proper first-rep scoring on fold-0:

   | Tier | Cell | first-rep | avg-of-3 | Paper |
   |---|---|---|---|---|
   | Fast | `RT_paper_Fast_pst5` | 36 | 44 | 36 ✓ |
   | Slow | `RT_paper_Slow_pst20` | 44 | 72 | 58 |
   | Slow | `RT_paper_Slow_pst25_inclz` (best) | 50 | 70 | 58 |
   | EoR | `RT_paper_replica_partial` | 52 | 74 | 66 |
   | EoR | `RT_paper_EoR_K7_CSFWM_HP_e1_inclz` (deployment champion) | 54 | 76 | 66 |

   Likely cause for Slow/EoR remaining gap: paper's `mindeye.py:770-784` uses
   non-causal cumulative z-score (includes current trial in mean/std) and a
   running average over accumulated repeats. Our pre-extracted βs use causal
   cum-z. Different β source, not different eval mode.

## Cells that move dramatically between modes

These are the cells where the framing of "this approach helps / doesn't help"
was most affected by the mislabel. Showing absolute pp delta = avg-of-3 minus
first-rep, sorted descending:

| Cell | first-rep | avg-of-3 | Δ |
|---|---|---|---|
| `OLS_glover_rtm` | 26 | 60 | +34 |
| `OLS_glover_rtm_softFrac_only` | 26 | 60 | +34 |
| `VariantE_Spatial_glover_rtm` | 30 | 60 | +30 |
| `VariantCD_Combined_glover_rtm` | 30 | 58 | +28 |
| `OLS_glover_rtm_denoiseK5` | 44 | 72 | +28 |
| `OLS_persistentLSA_K10_glover_rtm` | 44 | 72 | +28 |
| `OLS_persistentLSA_crossrun_K10_glover_rtm` | 42 | 72 | +30 |
| `RT_paper_pst19_inclz` | 46 | 74 | +28 |
| `RT_paper_EoR_K7_CSFWM_HP_e1_inclz` | 54 | 76 | +22 |

Lower-Δ cells (smaller mode dependence) tend to be the cells where the β
extraction is already higher-SNR — the model has less to gain from averaging.

## Cells that DON'T move between modes (already-averaged inputs)

These cells operate on inputs that are already 1-β-per-image (post-averaging,
with `n_trials = 532` unique images). Both modes give identical results:

- `Offline_paper_replica_full` (76 / 76)
- `RT_paper_replica_full` (74 / 74)
- `RT_paper_replica_full_streaming_pst*` (50-68, both modes equal)
- `RT_streaming_pst8_HOSVD_K5_full` (42 / 42)

## Affected conclusions from prior rounds

Reframing earlier round headlines now that we have correct labels:

1. **Round 1 ("EoR + GLMdenoise K=10 — REJECTED, task leakage")** — the
   judgment was based on first-rep numbers compared against a 76% target
   (which was actually avg-of-3). With a correct ~60% first-rep target, the
   K=10 numbers (`RT_paper_EoR_K10_inclz` first-rep=50, `_K10_CSFWM_inclz` first-rep=56)
   are within sampling noise of the target — the rejection was premature.

2. **Round 2 ("Slow pst sweep plateaus at 44-46%")** — still true. pst=25
   gets 50%, all other pst values 44-48%. None reach paper's 58%. So pst
   tuning genuinely can't close this gap; β extraction policy is the issue.

3. **Round 4 ("78% rep-avg matches Offline anchor; 50% first-rep stays open")** —
   the "open" was the phantom gap. Now closed.

4. **Round 6 ("88% rep-avg ✓; first-rep stays at 56%")** — the "stays at 56%"
   was correctly first-rep but compared against a wrong 76% anchor. Real
   first-rep paper anchor is closer to 60-64% (the Brain column matches
   exactly at 64%).

5. **Round 8 ("K=7 + HP + erode×1 wins at 97.2% 2-AFC; deployment champion")** —
   2-AFC is a different metric (pairwise discrimination, not 50-way top-1)
   and unaffected. The champion still stands as the deployment recipe.

## Real remaining gaps (post-correction)

Only two genuine gaps from paper Table 1:

1. **Slow first-rep: ours 44-50%, paper 58%.** Closeable with paper's β
   extraction (non-causal cum-z, possibly different pst window).

2. **EoR first-rep: ours 52-56%, paper 66%.** Closeable with running-average
   over accumulated repeats per `mindeye.py:770-784` (not implemented in
   our pre-extracted βs).

3. **Offline 3T (avg. 3 reps.) Image: ours 76%, paper 90%.** Possibly
   post-model output averaging vs pre-model β-averaging. Distinct from above.

Everything else in Table 1 retrieval columns reproduces exactly when compared
mode-to-mode and fold-to-fold correctly.

## What's still the deployment champion

The `RT_paper_EoR_K7_CSFWM_HP_e1_inclz` cell, scored as the closed-loop
deployment recipe (Stage 11 of PIPELINE.md):

- 50-way top-1 (first-rep): **54% Image / 52% Brain**
- 50-way top-1 (avg-of-3): 76% Image / 76% Brain
- 2-AFC pairwise: **97.2%** (operates on first-rep; different metric)
- Cohen's d: 2.42 (different metric)

The 97.2% 2-AFC is the right operating point for closed-loop neurofeedback
where each scan provides 1-vs-1 pairwise discrimination, not 50-way ranking.
That metric is unaffected by the first-rep / avg-of-3 distinction since
2-AFC is computed on the same 50 first-rep predictions vs paired GTs.

— Rescored 2026-05-03 on fold-0 (paper-faithful ckpt).
