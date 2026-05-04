# EoR vs Offline gap — preprocessing experiments re-judged (2026-05-03)

After the subset0/1/2 correction, the question shifted from "why is EoR 14pp
behind Offline first-rep?" to "what's the actual structural gap between the
EoR pipeline (LSS at end-of-run) and the Offline pipeline (GLMsingle on full
session)?"

**Headline answer**: when compared apples-to-apples (subset2 vs subset2, both
avg-of-3-reps), the gap is **zero or slightly inverted**. Several EoR
configurations *exceed* the canonical Offline GLMsingle pipeline.

## The gap structure

Paper anchors after correction:
- Offline 3T = `filter_and_average_repeats` from canonical GLMsingle = subset2 = **76%**
- EoR real-time = subset1 (avg-of-2 by running average) at end-of-run = **66%**
- Apparent gap: 10pp

Our reproduction breakdown:

| Cell | subset0 | subset1 | subset2 |
|---|---|---|---|
| Canonical Offline GLMsingle | 76 | n/a | 76 (=subset2 by definition) |
| `RT_paper_EoR_K7_CSFWM_HP_e1_inclz` (deployment champion) | 54 | 66 | **76** |
| `RT_paper_EoR_K7_CSFWM_HP_erode_inclz` | 56 | 62 | **78** |
| `RT_paper_EoR_K7_CSFWM_erode_inclz` | 56 | 62 | **78** |
| `RT_paper_EoR_K0_CSFWM_inclz` (no aCompCor) | 56 | 62 | **78** |
| `RT_paper_EndOfRun_pst_None_inclz` (no preprocessing knobs) | 56 | 62 | **78** |
| `RT_paper_EoR_fmriprep_inclz` (fmriprep BOLD instead of rtmotion) | 54 | 66 | **76** |

**On subset2, the EoR pipeline reaches or exceeds Offline.** Six different
EoR configurations land at 76-78% subset2.

The "10pp gap" reported in the paper is a like-vs-unlike comparison: paper
Offline = subset2, paper EoR = subset1. Comparing subset2 to subset2 dissolves
the gap. The paper's own data — if scored at subset2 across the board —
should show this. The latency tradeoff (1d for subset2 in Offline vs 2.7m for
subset2 in EoR) remains real and is the actual story.

## Preprocessing knob effects (subset2, fold-0)

### aCompCor K-sweep (CSFWM noise pool)

| K | subset0 | subset1 | subset2 | Δ from K=0 |
|---|---|---|---|---|
| 0 (no aCompCor) | 56 | 62 | 78 | — |
| 3 | 52 | 70 | 70 | -8 |
| 5 | 56 | 68 | 74 | -4 |
| 7 | 52 | 70 | 74 | -4 |
| 10 | 56 | 70 | 68 | -10 |
| 15 | 46 | 60 | 64 | -14 |
| 20 | 32 | 50 | 64 | -14 |

**aCompCor doesn't help subset2; K=0 is best by 4pp.** It does help subset1
slightly (K=3-7 plateau at 68-70 vs K=0's 62). aCompCor's value is in
recovering from short-window noise (subset0/subset1), not avg-of-3.

K>10 catastrophically hurts. Paper rejection of K=10 (round 1) was correct
but for a less dramatic reason than originally framed.

### HRF model (with K=7 + CSFWM + HP + erode×1)

| HRF | subset0 | subset1 | subset2 |
|---|---|---|---|
| Glover canonical | 54 | 66 | 76 |
| Glover + temporal derivative | 52 | 68 | 76 |
| Glover + temp + dispersion | 50 | 60 | 68 |
| SPM canonical | 52 | 62 | 68 |
| SPM + temp + disp | 52 | 56 | 64 |
| 20-HRF library (GLMsingle stage 1) | 40 | 50 | 64 |

**Glover wins on subset2. Adding temporal derivative is neutral-to-slight-help
on subset1 (66→68) and neutral on subset2.** Dispersion derivative hurts
substantially. SPM canonical is 8pp worse than Glover. The 20-HRF library
hurts despite being part of GLMsingle's pipeline — when ridge regularization
isn't applied jointly, HRF flexibility just adds noise.

### High-pass filter and erosion (with K=7 + CSFWM)

| Variant | subset0 | subset1 | subset2 |
|---|---|---|---|
| K7+CSFWM (baseline) | 52 | 70 | 74 |
| K7+CSFWM+HP | 54 | 64 | 72 |
| K7+CSFWM+e1 | 58 | 68 | 74 |
| K7+CSFWM+HP+e1 (deployment champion) | 54 | 66 | 76 |
| K7+CSFWM+HP+erode | 56 | 62 | **78** |
| K7+CSFWM+erode (no HP) | 56 | 62 | **78** |

**Erosion of the noise-pool mask is the strongest single-knob improvement** —
+4pp on subset2. The deployment champion (e1 = erode×1) sits at 76%; erode×N
(unspecified) reaches 78%, beating canonical GLMsingle.

HP-filter is marginal: it costs 6pp on subset1 alone but recovers 4pp when
combined with erode×1.

### Stein shrinkage (with K=7 + CSFWM + HP + e1)

| Stein λ | subset0 | subset1 | subset2 |
|---|---|---|---|
| 070 | 54 | 64 | 72 |
| 085 | 54 | 64 | 74 |
| 095 | 54 | 66 | 74 |
| 100 (no shrinkage) | 54 | 66 | 76 |

Shrinkage hurts. The β-as-emitted (no shrinkage) is best.

### Fracridge — catastrophic outside GLMsingle stack

Per-voxel fracridge applied as a standalone post-LSS step: subset2 = 2-18%
(near-chance) for most fractional values. Only `frac=1.0` (no shrinkage) at
subset2=60% is anywhere near baseline. Fracridge's value in GLMsingle comes
from its joint use with HRF library + GLMdenoise + cross-validation; bolted
onto LSS in isolation, it just destroys signal.

### BOLD source (rtmotion vs fmriprep)

| Source | subset0 | subset1 | subset2 |
|---|---|---|---|
| rtmotion (K7+CSFWM+HP+e1) | 54 | 66 | 76 |
| fmriprep (K7+CSFWM+HP+e1) | 54 | 66 | 76 |

**Identical.** BOLD source doesn't move the needle when other settings match.

### Segmentation source (FAST vs DeepMriPrep vs SynthSeg) — for the noise pool

| Segmentation | subset0 | subset1 | subset2 |
|---|---|---|---|
| FAST (default, K7 CSFWM HP e1) | 54 | 66 | 76 |
| DeepMriPrep (HP + e1 not applied — naming variant) | 50 | 64 | 74 |
| SynthSeg (HP + e1) | 44 | 60 | 62 |
| DeepMriPrep (HP only) | 54 | 66 | 70 |

FAST and DeepMriPrep tie at subset1 (66). FAST wins subset2 by 2pp.
SynthSeg lags by 8-14pp — its noise-pool definition is too liberal.

## Conclusions

1. **The "EoR vs Offline gap" was an artifact of subset mismatch.** When
   evaluating both at subset2 (avg-of-3-reps), the EoR LSS pipeline reaches
   the same retrieval ceiling (76-78%) as canonical Offline GLMsingle.

2. **Several EoR configurations outperform canonical GLMsingle on subset2**
   by 2pp: K0+CSFWM, K7+CSFWM+erode, K7+CSFWM+HP+erode. Caveat: 2pp on n=50
   is within sampling noise (one trial flip = 2pp).

3. **For subset1 (the paper-reported EoR latency)**, the deployment champion
   (K7+CSFWM+HP+e1) ties paper anchor 66% exactly. Glover+temporal-derivative
   ekes out 68%, and lower-K CSFWM (K=3, K=5, K=7, K=10) plateau at 68-70%.
   K=0 underperforms subset1 (62%) — this is where aCompCor pulls its weight.

4. **The single most surprising knob: noise-pool erosion.** +4pp on subset2,
   no other knob comes close.

5. **Glover canonical HRF is sufficient.** Adding flexibility (FLOBS, 20-HRF
   library, temporal derivative beyond first-order) hurts when ridge isn't
   coupled. Glover+temporal-derivative is the only flex variant that's
   neutral-to-positive.

6. **fracridge bolted onto LSS is catastrophic.** It only works inside
   GLMsingle's joint pipeline. This explains why the canonical GLMsingle
   stack was hard to outperform via piecemeal additions.

## What this means for the paper

Paper Table 1's reported 10pp Offline-EoR gap is a real *latency-vs-quality*
tradeoff but the *quality* component is much smaller than 10pp when you
match subsets. The paper could include a "subset2-equivalent" row for EoR
showing it reaches 76-78% with 2.7-minute latency, equal to canonical
Offline at 1-day latency. This would be a stronger claim about real-time
viability than the current framing.

## What this means for closed-loop deployment

The deployment champion (K7+CSFWM+HP+e1, 97.2% 2-AFC at subset0=first-rep)
is the right operating point for closed-loop, where each scan provides
1-vs-1 binary discrimination and you can't afford to wait for 3 reps.

For an "offline-quality" tier in a closed-loop paradigm, the same recipe
+ erosion (K7+CSFWM+HP+erode) reaches subset2=78%, beating GLMsingle by
2pp at 2.7-minute latency.

— Subset analysis 2026-05-03, fold-0, n=130 prereg cells.
