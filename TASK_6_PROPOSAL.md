# Task 6 proposal — pretrain and fine-tune on real-time preprocessed data

**Status:** proposal, not yet confirmed with Rishab Iyer. Task 6 is the one
unassigned entry on the MedARC RT-MindEye task list (Discord 2026-03-06).
This doc is the argument for taking it on, scoped through the Variant G
(Bayesian AR(1)) lens developed in `hippy-feat/`.

---

## What Task 6 is (per Rishab's 3/6 Discord post)

> *"pretrain and fine-tune on real-time preprocessed data"*

Short form: **remove the offline-training / real-time-inference distribution
mismatch by making training itself use RT-compatible preprocessing end-to-end.**

### The current MindEye 2 setup has a distribution mismatch baked in

```
pretrain on 7 NSD subjects  → fMRIPrep + GLMsingle betas  (offline, heavily denoised)
fine-tune on sub-005 ses-01 → fMRIPrep + GLMsingle betas  (offline)
test on sub-005 ses-03      → RT motion + nilearn Glover  (noisy, single-trial, real-time)
                                                            ↑
                                       Decoder sees this distribution for the first time
                                       at inference.
```

The ridge learned to map from "clean GLMsingle-denoised betas" → CLIP. At
inference it gets noisier RT-preprocessed betas. That mismatch is part of
what costs the **10 pp image-retrieval gap** between paper Offline 3T (76 %)
and End-of-run RT (66 %).

**Task 6 flips this**: same architecture, same data, but run *everyone's* BOLD
through RT-compatible preprocessing during training as well.

```
pretrain on 7 NSD subjects  → RT-style betas              (matched to deployment)
fine-tune on sub-005 ses-01 → RT-style betas
test on sub-005 ses-03      → RT-style betas              (same distribution)
```

### Why Task 6 is distinct from Task 2.1

Task 2.1 **attributes** the 10 pp gap to fMRIPrep-motion vs GLMsingle-HRF
contributions (what's responsible, in pp). Task 6 **eliminates** the gap by
retraining. They're complementary: 2.1 tells you which preprocessing component
caused the degradation; 6 replaces the need to preserve offline-style
preprocessing at all.

---

## Where Variant G fits

`hippy-feat/scripts/rt_glm_variants.py:VariantG_Bayesian` emits per-voxel
per-trial **posterior variance** alongside the usual beta point estimate —
using an AR(1)-prewhitened conjugate Gaussian GLM (≈4.6 ms/TR on the GB10
after the JIT fix). `scripts/mindeye_retrieval_eval.py`-style tests confirm
the variance output is strictly positive, decreases with more TRs, and
tracks plausibly with noise.

### The novel angle

MindEye 2 was trained on **GLMsingle point estimates**. The authors didn't
have per-trial variance to feed the decoder. Task 6 with Variant G output
gives the decoder a signal it has literally never seen before:
`(beta_mean, beta_var)` pairs per voxel per trial.

Three ways that signal can be consumed by the ridge:

1. **Variance-weighted ridge.** `y = β · x / sqrt(var(x))` — down-weight
   voxels whose beta is noisy on this trial. Standard heteroscedastic
   regression; well-understood math.
2. **Concatenated input.** `ridge([beta_mean, beta_var]) → latent`. Doubles
   input dim; more params per subject.
3. **Confidence-gated inference.** Set `beta[trial, voxel] = 0` where
   `|beta| / sqrt(var) < threshold`. Zero architectural change, just a
   mask applied at inference. This is what `confidence_mask` in
   `rt_glm_variants.py` exists for.

Any of these is a publishable ablation. (1) is the cleanest scientifically.

### Which variants are plausible for Task 6 more broadly

| Train on | Resulting decoder specialization |
|---|---|
| Variant A output | Standard RT-regime ridge; should close most of the 10 pp |
| Variant C output | Per-voxel-HRF-aware ridge |
| **Variant G output** | **Variance-aware ridge** — the novel contribution |

Variant G's the clearest research story; A and C are more like "matched
distribution" baselines that could be ablation points in the Task 6 paper.

---

## Minimum viable experiment (MVE)

The full Task 6 — retraining MindEye 2 from scratch on RT-preprocessed NSD
data — is **weeks of GPU time**. That's why nobody has claimed it.

A practical first experiment that tests the variance-aware hypothesis at low
cost:

### MVE-1: frozen-backbone, retrain-ridge-only

1. Start from the paper's `multisubject_sdxlturbo_excludingsubj01_40sess.pth`
   backbone (already on disk at `/data/3t/data/model/`).
2. **Freeze** the shared backbone + diffusion prior + unCLIP.
3. Generate Variant G output on sub-005 ses-01 all 11 runs (2,792 voxels,
   ~693 trials × 3 reps avg).
4. **Retrain only the sub-005 ridge** with one of:
   - (baseline) input = `beta_mean` → same as paper, our RT-regime data
   - (variance-weighted) input = `beta_mean / sqrt(beta_var + eps)` → the novel bit
   - (confidence-gated) input = `beta_mean * (|beta_mean| > 2 * sqrt(beta_var))`
5. Evaluate all three variants on ses-03 test trials (50 special515 × 3 reps).
6. Compare against paper's reported Offline 3T (76 %) and End-of-run RT (66 %)
   and our Task 2.1 condition A/B numbers.

### MVE-1 cost and timeline

- Compute: ~1 hour GPU for each ridge retrain (it's a single linear layer)
- Data: already on disk (fMRIPrep'd sub-005, paper checkpoint, paper test set)
- Code: 90 % reuses `task_2_1_factorial.py` and `mindeye_retrieval_eval.py`;
  one new script `scripts/task_6_mve_retrain_ridge.py`
- Result: publishable figure regardless of outcome
  - Positive result → "variance-aware decoding beats point-estimate decoding"
  - Null result → "per-trial variance isn't enough signal; need full
    heteroscedastic GP or full retrain"

### MVE-2 (if MVE-1 is positive): cross-subject pretrain with variance-weighted ridge

1. Run Variant G on NSD 7 subjects' BOLD (have to download raw NSD BOLD from
   AWS; ~200 GB/subject — big but feasible over a week on DGX Spark)
2. Pretrain from scratch with variance-weighted ridge heads per subject,
   shared backbone
3. Fine-tune on sub-005 Variant G output
4. Evaluate on ses-03

This is the "real" Task 6. ~2 weeks GPU time.

---

## Open questions for Rishab

1. Is Task 6 still unassigned? (3/6 list, no subsequent Discord post claimed it)
2. Does the "RT-style betas" pretraining need to run through the actual
   RT-Cloud pipeline, or is the jaxoccoli implementation (Variants A-G)
   close enough? (The latter is *faster* and *matches what'll run at
   inference*; the former is *more paranoid about framework-level bugs*.)
3. Is there appetite for the variance-aware angle specifically, or would
   the lab prefer a vanilla "match-distribution" Task 6 that uses Variant A
   output exclusively?
4. For NSD raw BOLD: is the lab OK with us pulling it to DGX Spark storage
   (non-trivial I/O + 200 GB/subject), or is there a shared staged copy?

---

## Relationship to other tasks

| Task | Who | Overlap with Task 6 |
|---|---|---|
| 2.1 fMRIPrep vs GLMsingle | Me | Measures the gap; Task 6 closes it |
| 3 Foundation-model time series → decoder input | Boris | Orthogonal; different input modality |
| 4 T-PHATE / MRAE | Alexa | Orthogonal dim-reduction layer |
| 5 Learned embedding for transfer | Akash | **Potential collaboration** — if Akash builds the learned adapter, Task 6 could retrain it on RT-preprocessed data |
| 7 Robust CLIP | Rushikesh | Changes target space; Task 6 could use either target |
| 8 MMDiT architecture | Cesar | Changes decoder; Task 6 could retrain MMDiT on RT data |
| 9 Hyperalignment | cindyhfls | **Direct overlap** — cross-session alignment is another way to close the offline→RT gap; different mechanism than retraining |

---

## Draft Discord message to Rishab

> Hey @Rishab Iyer — now that Task 2.1 is moving (github.com/m9h/hippy-feat
> has the factorial + retrieval eval pushed), I'm eyeing Task 6 next, but
> through a specific lens: retraining just the sub-005 ridge on Variant G
> (AR(1) Bayesian) output so the decoder gets per-trial posterior variance,
> which MindEye 2 training never saw. Frozen-backbone MVE is ~1 hr GPU, not
> a full 2-week pretrain. Writeup at
> github.com/m9h/hippy-feat/blob/main/TASK_6_PROPOSAL.md. OK if I take it?

---

## What's already built that feeds Task 6

- `scripts/rt_glm_variants.py:VariantG_Bayesian` — AR(1) conjugate GLM, 18 passing tests
- `scripts/rt_glm_variants.py:confidence_mask` — SNR-thresholded mask helper
- `scripts/mindeye_retrieval_eval.py` — retrieval evaluation harness
- `scripts/task_2_1_factorial.py` — per-trial beta production at the paper's 2792-voxel finalmask
- NGC PyTorch 26.03 arm64 SIF at `/data/derivatives/containers/`
- Paper checkpoint, fMRIPrep'd sub-005, rtcloud-projects clone, test stimuli — all on disk

Roughly, a new `scripts/task_6_mve_retrain_ridge.py` + matching sbatch would
reuse all of these and write a new ridge layer per-variant into
`/data/derivatives/rtmindeye_paper/task_6_ridges/`.
