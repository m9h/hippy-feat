# Paper-anchor ladder reproduction — note for Rishab (corrected 2026-05-03)

> **Errata vs. original 2026-04-30 version**:
>
> 1. The original headline labeled all four tiers "single-rep first-presentation
>    per paper §2.7". That was wrong for Offline 3T: `score_canonical_glmsingle.py`
>    actually does paper cum-z + repeat-avg (`filter_and_average_repeats`
>    semantics — average all 3 trial βs per image, then score 50 averaged βs).
>    The 76% Offline number is real and correct, but it was avg-of-3, not
>    first-rep.
>
> 2. After auditing `PrincetonCompMemLab/mindeye_offline:avg_betas/utils.py:800`
>    we now know the paper's `recon_inference-multisession.ipynb` uses exactly
>    that same `filter_and_average_repeats` for the Offline retrieval row.
>    So our 76% and the paper's 76% are the **same number** computed the
>    **same way** — no gap there. Likewise our 60% first-rep on the same data
>    and the paper's "Offline 3T" 64% Brain row are essentially identical.
>
> 3. The original framing implied RT tiers were 32pp behind a 76% first-rep
>    Offline anchor. Wrong: first-rep Offline is ~60%, so RT first-rep tiers
>    are 16-25pp behind first-rep Offline, not 32pp. Several "missing
>    ingredient" investigations were chasing a phantom gap.
>
> 4. Driver code itself was internally consistent — the bug was in the doc
>    layer (this report). Confirmed by full audit of 19 retrieval scorers.

Local Mac reproduction of the four-tier RT/Offline ladder from the ICML 2026
paper, sub-005 ses-03, scored against
`sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth`
(fold-0; **the paper's default**, hardcoded in
`recon_inference-multisession.ipynb` cell 2).

## Headline result — both eval modes side-by-side

50 special515 retrieval, Image (forward, brain→image) / Brain (backward, image→brain).
Paper anchors from Table 1 lines 262-265.

| Tier | Latency | first-rep Image / Brain | avg-of-3 Image / Brain | Paper Image / Brain |
|---|---|---|---|---|
| Fast real-time | 14.5s | **36 / 34** | 44 / 68 | 36 / 40 |
| Slow real-time | 36s | 44 / 54 | 72 / 76 | 58 / 58 |
| End-of-run real-time | 2.7m | 52 / 52 | 74 / 88 | 66 / 62 |
| Offline 3T | 1d | 60 / **64** | **76** / 88 | 76 / 64 |
| Offline 3T (avg. 3 reps.) | 1d | — | 76 / 88 | 90 / 88 |

Bold = exact-or-near match. The Offline 3T row hits paper Image (76%) on
avg-of-3 and paper Brain (64%) on first-rep — same data, two paper-implied
modes. Fast hits Image (36%) on first-rep.

## What we match

- **Fast Image retrieval, first-rep**: 36% exact match.
- **Offline 3T Image retrieval, avg-of-3**: 76% exact match — confirms the
  paper's `filter_and_average_repeats` policy.
- **Offline 3T Brain retrieval, first-rep**: 64% exact match.
- **Offline 3T (avg 3 reps) Brain retrieval, avg-of-3**: 88% exact match.

## What still doesn't match (post-correction)

| Tier / col | Our number | Paper | Δ | Probable cause |
|---|---|---|---|---|
| Slow Image, first-rep | 44% | 58% | -14pp | β extraction (causal vs. non-causal cum-z) |
| EoR Image, first-rep | 52% | 66% | -14pp | β extraction (running average not implemented) |
| Offline 3T (avg 3 reps) Image | 76% | 90% | -14pp | possibly post-model averaging vs pre-model β-averaging |
| Offline 3T Brain, avg-of-3 | 88% | 64% | +24pp* | suspect paper typo — see below |

*The Offline 3T Brain column at 64% (paper) is internally inconsistent
with everything else: it would mean running the same eval that produces
76% Image suddenly drops to 64% Brain (vs the (avg 3 reps) row's 88%).
Our reproduction of that exact eval gives 88% Brain. Possibility: the paper's
"Offline 3T" row is a hybrid — Image computed via avg-of-3, Brain via first-rep
(64% would match first-rep). Or the 64% is a typo for 88%.

## Slow / EoR β extraction gap

Looking at `rtcloud-projects-mindeye/mindeye.py:761-784`, the paper's RT
pipeline does:

```python
z_mean = np.mean(np.array(all_betas), axis=0)   # NON-causal: includes current trial
z_std  = np.std(np.array(all_betas), axis=0)
if is_repeat:
    # average over ALL accumulated repeats (including current)
    betas = np.mean(...repeat z'd values..., axis=0)
else:
    # use only most recent (single-rep)
    betas = ((all_betas - z_mean) / (z_std + eps))[-1]
```

Our `RT_paper_*_inclz_ses-03_betas.npy` files were extracted with **causal**
cum-z (`arr[:i]`, excludes current trial). That alone could account for
~10pp; the running-average-on-repeats is also not implemented in our
extraction. Both fixable.

## pst sweep — Slow plateaus, can't be moved by windowing alone

We swept `pst ∈ {4, 5, 6, 18, 19, 20, 21, 22}` (Fast and Slow neighborhoods)
with inclusive cum-z + first-rep filter. **All numbers are first-rep Image
retrieval.**

| pst | seconds | top-1 | top-5 | 95% CI |
|---|---|---|---|---|
| 4 | 6.0  | 34% | 56% | [22, 48] |
| 5 | 7.5  | **36%** ← Fast | 60% | [24, 50] |
| 6 | 9.0  | 38% | 70% | [24, 52] |
| 18 | 27.0 | 46% | 72% | [32, 60] |
| 19 | 28.5 | 46% | 74% | [32, 60] |
| 20 | 30.0 | **44%** ← Slow | 74% | [30, 58] |
| 21 | 31.5 | 44% | 74% | [30, 58] |
| 22 | 33.0 | 46% | 74% | [32, 60] |

The Slow plateau is 44–46% across the 27-33s neighborhood. Windowing alone
can't move it to 58% — we need the β extraction fix above.

## A potential discrepancy in the published `tr_labels.csv`

We computed the actual decode-TR delay implied by the `tr_label_hrf` column
in `sub-005_ses-03_task-C_run-NN_tr_labels.csv` (the column `mindeye.py:665`
consumes for the Fast tier). For each non-blank trial we took the
FIRST `tr_list >= onset_TR` where `tr_label_hrf` matches the trial's image
name, and computed `delta = tr_list - onset_TR`:

```
all non-blank deltas:    n=693  mean=4.23 TRs (6.35s)  std=0.59s
special515 deltas:       n=150  mean=4.22 TRs (6.33s)  std=0.59s
Paper Table 4 Fast:                              7.85s ±0.59s
```

The std matches paper Table 4 perfectly (±0.59s) but the **mean is exactly
1 TR (1.5s) less** than Table 4 reports for Fast. We get Fast 36% with
`pst=5 TRs` (≈7.5s) which matches the paper number, not the 4-TR CSV avg.
This suggests either an off-by-one in the CSV vs Table 4, or "stim delay"
includes +1 TR for some processing step.

## Open questions for you

1. **Slow tier decode rule.** Paper Table 4 shows stim delay 29.45±2.63s
   (≈20 TRs at TR=1.5s). We can't find the rule in `mindeye.py` — only
   `tr_label_hrf` (Fast) is implemented. Could you share the decode-TR rule
   for Slow (is it `onset_TR + 20`, aligned to TR boundary, wait-for-next-onset)?
   Combined with the non-causal-cum-z fix, this is what we need to close
   the −14pp Slow gap.

2. **End-of-run decode rule.** Last TR of run, or fixed offset, or all βs
   in run + average over accumulated repeats per `mindeye.py:782`?

3. **The Offline 3T Brain column = 64%.** Is this a typo for 88%? Or is
   "Offline 3T" Brain genuinely computed with first-rep while Image uses
   avg-of-3? If the latter, please clarify in the paper.

4. **Offline 3T (avg 3 reps) Image = 90%.** With `filter_and_average_repeats`
   + fold-0 we get 76% Image, 88% Brain. Is the (avg 3 reps) row using
   post-model output averaging (run model 3× per image, average outputs)
   instead of pre-model β-averaging? The two don't commute through the
   nonlinear backbone.

5. **`tr_label_hrf` 1-TR discrepancy.** Is the 6.35s mean delta vs Table 4's
   7.85s intentional (different "stim delay" definition), or a CSV vintage
   issue?

6. **Test-set ordering.** At what session position do the 50 special515
   first-presentations fall? If most are early (when cum-z stats are immature),
   our bootstrap CI might be biased downward.

## Files

Local Mac artifacts:
- `local_drivers/score_avg_repeats_offline.py` — fold-0 + avg-of-3, reproduces 76% Offline Image exactly
- `local_drivers/score_offline_first_rep.py` — fold-0 + first-rep, reproduces 64% Offline Brain exactly
- `local_drivers/score_rt_tiers_both_modes.py` — Fast/Slow/EoR × {first-rep, avg-of-3} matrix on fold-0
- `local_drivers/score_rt_tiers_singlerep.py` — RT tiers, first-rep only (paper §2.7-style)
- `local_drivers/score_rt_tiers_inclz.py` — RT tiers, inclusive (non-causal) cum-z variant
- All βs at `data/rtmindeye_paper/task_2_1_betas/prereg/` ending in `_inclz_ses-03_betas.npy`
- Results: `data/rtmindeye_paper/task_2_1_betas/{rt_tiers_both_modes_fold0,avg_repeats_offline_score,first_rep_offline_score}.json`

Canonical-source provenance:
- Training/eval code: `PrincetonCompMemLab/mindeye_offline:avg_betas` branch
  (`recon_inference-multisession.ipynb` + `final_evaluations.ipynb` + `utils.py:800`)
- Real-time pipeline: `brainiak/rtcloud-projects/mindeye/scripts/mindeye.py:761-784`

— Mac reproduction, corrected 2026-05-03
