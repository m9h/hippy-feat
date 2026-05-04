# Fast tier refiner — negative result

Tested whether a small per-voxel learned refiner can escape the Fast (pst=5,
14.5s latency) Image-retrieval ceiling at 36%.

## Setup

- Architecture: per-voxel scalar (gain·β + bias), 5584 trainable params, init gain=1, bias=0
- Frozen fold-0 ckpt downstream
- Loss: 1 - cosine_sim(pred_clip_voxels, CLIP-image-tokens) — same loss MindEye2 uses
- Training: 1138 pairs (ses-03 non-test + ses-01 + ses-02 trials excluding all special515)
- Validation: 126 held-out training pairs
- Test: ses-03 first-rep special515 (the 50 we always test)

## Result

| Metric | Baseline (no refiner) | Refiner (best-val ckpt) |
|---|---|---|
| Test Image % | 36 | 32 |
| Test Brain % | 34 | 34 |
| Best val Image % | — | 30.2 (epoch 0) |

**Best-val Image accuracy peaked at epoch 0 — the untrained refiner.** Training
drove val accuracy down monotonically; test accuracy degraded from 36% to 32%.

## Diagnosis

The supervision signal — cosine similarity to CLIP-image embeddings — is the
same loss fold-0 was trained against. There is no new information for the
refiner to extract that fold-0 hasn't already absorbed at avg-of-3 latency.
With 5584 free parameters and 1138 train pairs, the refiner has more than
enough capacity to overfit ses-01/02/03 training noise that doesn't generalize
to the ses-03 test set's first-rep distribution.

Additional structural issue: mixing ses-03 rtmotion-BOLD βs with ses-01/02
fmriprep-BOLD βs in training creates a feature-distribution mismatch with the
ses-03 rtmotion test set. The refiner partially adapts to the dominant
fmriprep distribution and degrades on rtmotion test inputs.

## What this rules out

- Per-voxel gain+bias correction at test time
- Direct cosine-to-CLIP supervision on training trials
- Pooling cross-session BOLD sources for refiner training

## What's still on the table for Fast

The negative result is consistent with the hypothesis from the Heunis taxonomy
analysis: the Fast SNR floor at ~36% is intrinsic to single-trial 7.5s BOLD
extraction, not something that per-voxel preprocessing or post-extraction
refinement can move. Remaining honest options:

1. **Acquisition changes** — multi-echo EPI, shorter TR via higher MB, longer
   stimulus duration. Not testable on existing data.
2. **Cross-latency distillation** — train a Fast→Slow distiller using streaming
   GLM Slow s1 (70%) as teacher. Different supervision signal than direct
   CLIP-cosine — gives the student something to learn that fold-0 doesn't already do.
3. **Time-resolved decoder** — pass the full 5-TR BOLD path (not the β) through
   a small temporal model. Requires retraining the decoder, not just refining
   inputs.
4. **Foundation-model backbone (TRIBEv2 / BrainLM)** — replace the MindEye2
   ridge+MLP head with a model pretrained on massive fMRI; may handle low-SNR
   single-trial inputs better.

Option 2 is the most directly comparable to this experiment and the next
experiment to run if continuing. The teacher signal (streaming GLM Slow s1
clip_voxels) is fundamentally different from the CLIP-cosine target and
contains information fold-0-on-Fast-input doesn't have.

— Fast refiner null result, 2026-05-04, fold-0, n=50 special515 ses-03.
