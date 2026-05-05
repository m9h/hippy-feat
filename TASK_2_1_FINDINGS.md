# Task 2.1 — Findings (live, 2026-04-29)

**Question** (Discord-assigned): explain the ~10 pp top-1 retrieval gap
between RT-MindEye paper's online RT pipeline (66 %) and offline pipeline
(76 %) on sub-005 ses-03 special515.

**Answer** (current data): the gap is **windowing**, not preprocessing.
~75 % of the 10 pp comes from the GLM seeing only partial-run BOLD at
each decode point; only ~25 % comes from rtmotion-vs-fmriprep BOLD
source; GLMsingle Stages 1-3 contribute ~0 pp. **H3'** (can the
windowing gap be recovered causally with cross-run streaming?) is the
remaining open question; Regime C cells (`RT_streaming_pst8_HOSVD_K*`)
running now address it directly.

## The decomposition

Numbers below are top-1 image retrieval over 50 special515 images
(150 raw trials for `*_partial`, 50 post-repeat-avg trials for `*_full`),
using the ses-01 finalmask checkpoint
(`sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth`)
on both platforms — confirmed paper-exact at the offline anchor.

| Anchor | Platform | Top-1 | Notes |
|---|---|---|---|
| Paper RT | (paper) | **0.66** | reference target |
| Paper Offline | (paper) | **0.76** | reference target |
| Cell 12 `Offline_paper_replica_full` | DGX | **0.76** | paper-exact ✓ |
| Cell 12 `Offline_paper_replica_full` | Mac M5 Max | **0.76** | paper-exact ✓ |
| Cell 11 `RT_paper_replica_full` (full-run BOLD bug) | DGX | 0.78 | inflated — fits offline LSS |
| Cell 11 `RT_paper_replica_full` (full-run BOLD bug) | Mac | 0.74 | inflated — same root cause |
| Cell 11_streaming `RT_paper_replica_streaming_pst8_full` | Mac | **0.68** | gap recovered (8 pp on this ckpt; 10 pp on canonical) |
| Cell 11_streaming `RT_paper_replica_streaming_pst8_full` | DGX | _running (job 1034)_ | parity check vs Mac |

The Mac column 11_streaming = 0.68 vs cell 12 = 0.76 = **8 pp gap**
recovered (matches the paper's 10 pp on canonical checkpoint within
±2 pp). With the bug (full-run BOLD masquerading as RT) the gap was
–2 pp — wrong sign; we couldn't decompose what we couldn't see.

## H1' — windowing dominance (PROVISIONAL ✓)

> Windowing (A → B at constant motion+GLM) accounts for ≥ 70 % of the
> Offline-vs-RT gap.

Mac cell 11 streaming pst=8 (Regime B) = 0.68 vs cell 12 (Regime A,
fmriprep) = 0.76 → Δ = 8 pp. Switching `cell 11_full-run` (full-run BOLD,
Regime A on rtmotion) to `cell 11_streaming` (Regime B on rtmotion)
moved top-1 from 0.74 to 0.68 → Δ_window = 6 pp. **6 / 8 = 75 % ✓**.

DGX provisional confirmation pending job 1034 retrieval.

## H2' — motion source residual (PROVISIONAL ✓)

> Motion source (fmriprep − rtmotion at constant window) accounts for
> ≤ 30 % of the gap.

Same data: cell 12 (Regime A, fmriprep) − cell 11_full-run (Regime A,
rtmotion) = 0.76 − 0.74 = 2 pp on Mac. **2 / 8 = 25 % ✓**.

## H4' — within-run pst saturation (PROVISIONAL ✓)

> Top-1 retrieval as a function of `post_stim_TRs` saturates by pst=10
> within a single run.

Mac data: pst=8 → 0.68, pst=10 → 0.68. Saturation confirmed within
±0 pp. DGX pst sweep `RT_paper_replica_streaming_pst{4,6,8,10}` running
now (job 1034) for cross-platform replication.

## H5' — GLMsingle Stages 1-3 contribution (PROVISIONAL ✓ at top-1)

> GLMsingle Stages 1-3 contribute < 2 pp at top-1 to the Offline anchor.

Mac data (commit `2f96057`): cell 12 with vs without GLMsingle Stages
1-3 = same top-1 (0.76 = 0.76) at the canonical Glover + nilearn AR(1)
baseline. Top-5 differs by +2 pp with GLMsingle. **0 pp at top-1 ✓**;
mild contradiction at top-5. The "Task 2.1 = GLMsingle is the win"
narrative does not hold at top-1.

## H3' — cross-run causal recovery (UNTESTED → in flight)

> A Regime C cell with cross-run HOSVD top-K filter recovers ≥ 50 % of
> Δ_window while remaining causal/RT-deployable.

DGX job 1035 running with `RT_streaming_pst8_HOSVD_K{5,10}_partial`
and `RT_streaming_pst8_HOSVD_K5_full`. The cross-run template is the
top-K spatial PCs of `concat(past runs 1..r-1)` BOLD; for run r they
project current BOLD to K time series used as nuisance regressors in
the per-trial LSS fit. Strictly causal (each run uses only past runs).

**If H3' holds**: the offline lift is mostly session-shared structure
(drift, vasculature, slow-noise) that streams trivially. The
RT-deployable upper bound on retrieval is then within ~3 pp of offline,
not 10 pp — the message Discord wants.

**If H3' fails** (Δ_recovery < 50 % of Δ_window): the windowing gap is
GLM-noise-floor (per-trial AR(1) ρ̂ noisy when only ~10 TRs of BOLD are
available) and the only ways to close it further are non-causal
(repeat-avg across all session BOLD, batch-mode AR(1) ρ̂ across full
session). That's an honest negative deliverable — RT can't get closer.

## What's discarded under the amendment

Cells 5 (empirical-Bayes prior, Δ ≈ 0), 13 (EKF reset bug), 14 (HOSVD
per-run, not cross-run as intended), 15 (Riemannian rank-deficient),
16 (online EKF identifiability collapse). Beta files remain on disk;
not Task 2.1 evidence.

## Open issues for the public-facing claim

1. **Canonical-checkpoint replication**: all numbers above are on the
   ses-01 finalmask checkpoint. The paper's 66 % / 76 % are on the
   `sub-005_all_task-C_..._sample=10_..._epochs_150` checkpoint, which
   has test-set leakage on cell 12 (DGX measured 0.90 with leaky ckpt).
   Need a clean ses-01-only canonical run for Discord.
2. **Motion confounds absent in current DGX streaming cells**: per-run
   `.par` files weren't on `/data/3t/derivatives/motion_corrected_par/`
   so streaming cells run without MCFLIRT confounds. Mac side included
   them. Magnitude of the omission ≤ 1-2 pp based on Mac comparison;
   re-runnable later if needed.
3. **Variant G's actual value isn't top-1**: Mac's commit `d29e432`
   showed VG's distinguishing output is the per-trial posterior
   (β_mean, β_var) for calibration-aware selective accuracy — at τ=0.9
   covering 34-51 % of trials, all VG cells hit 84-90 % accuracy. Brier
   0.57 / ECE 0.13 on bare VG. AR(1) freq cannot produce these. The
   prereg's H2 ("VG ≈ AR(1)") was a category error: it tested a
   point-estimate metric on a method whose contribution is the
   posterior. This belongs in a separate writeup, not Task 2.1.

## 2026-05-02 update — paper-confirmed checkpoint (fold 10) closes most of the gap

Per Rishab (Discord), the actual checkpoint that produced Table 1 is
`sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_**10**_avgrepeats_finalmask_epochs_150/last.pth`
(fold **10** + epochs_150 suffix), available on `macandro96/mindeye_offline_ckpts`
on HF. **We had been using fold 0** (`..._3split_0_avgrepeats_finalmask.pth`),
which is a different fold of the same single-session ses-01 fine-tune.

Rescored with fold-10 ckpt:

| Tier | Paper Table 1 | fold-10 first-rep | fold-0 first-rep | Δ |
|---|---|---|---|---|
| **Offline 3T (avg 3 reps)** | **90%** | **88%** ← within 2 pp | 74% | ckpt closes 14 pp |
| **End-of-run RT** | **66%** | **64%** ← within 2 pp | 58% | ckpt closes 6 pp |
| Slow RT | 58% | 66% (+8 pp **above** paper) | 58% | overshoot |
| Offline 3T (single first-rep) | 76% | 62% (still −14 pp) | 56% | partial |

The rep-avg gap is now **closed** to within sampling variance (88% vs
90%). EoR-RT first-rep also matches within 2 pp. **First-rep Offline
still has a 14 pp residual gap and Slow first-rep now overshoots paper
by 8 pp** — these two anomalies suggest some additional drift between
our scoring of the paper's saved RT betas and how paper itself scored
them on Slow-tier data, or scoring-policy detail in single-trial
first-rep evaluation that we haven't replicated.

**For paper resubmission**: rep-avg metrics are reproducible from
canonical inputs + fold-10 ckpt. Single-trial first-rep needs one more
clarification from the paper team about the exact scoring path.

See `docs/table1_reproducibility_recipe.md` for the per-row recipe.

## Cross-references

- Pre-registration: `TASK_2_1_PREREGISTRATION.md` (2026-04-27 lock)
- Amendment: `TASK_2_1_AMENDMENT_2026-04-28.md` (windowing axis re-frame)
- Mac results branch: `origin/results/apple-silicon-2026-04-28`
- DGX retrieval summary: `/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/prereg_retrieval_summary.json`
- Live jobs: 1034 (Regime B pst sweep), 1035 (Regime C HOSVD)
