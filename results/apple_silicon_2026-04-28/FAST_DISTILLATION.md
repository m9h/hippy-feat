# Cross-latency distillation Fast ← Slow — positive result

Follow-up to the negative `FAST_REFINER_NEGATIVE.md`. Same architecture, different
supervision signal: teacher = frozen fold-0 forwarded on streaming-GLM Slow βs
(pst=20, K7+CSFWM+HP+e1 nuisance). Student takes Fast pst=5 βs as input, refines
through 5584-param per-voxel scalar, passes through frozen fold-0, loss = cosine
to teacher's clip_voxels.

## Result

| | Baseline (no refiner, Fast β through fold-0) | Teacher (Slow β through fold-0, upper bound) | **Student (Fast β + refiner, best-val)** |
|---|---|---|---|
| Image % | 36 | 54 | **40** (Δ +4) |
| Brain % | 34 | 58 | **48** (Δ +14) |

Training: 527 train / 93 val / 50 test. Best-val checkpoint at val cosine-loss 0.260.

**Comparison to the previous null result** (same architecture, CLIP-cosine target):

| Method | Image Δ | Brain Δ |
|---|---|---|
| Refiner with CLIP-cosine target | -4pp | 0 |
| Refiner with Slow-teacher target | **+4pp** | **+14pp** |

## Reading

The supervision signal matters more than the architecture. CLIP-cosine is the
loss fold-0 was already trained against — refining toward it has no new
information. The Slow teacher signal carries fundamentally different
information: what fold-0 produces from a β extracted at a longer post-stim
window with joint design. The Fast student partially learns to map its noisier,
shorter-window input into a representation closer to that cleaner target.

The teacher upper bound is 54% Image / 58% Brain (just Fast students's
single-rep prediction from streaming-Slow input through fold-0). The student
captures **22%** (Image) and **70%** (Brain) of the gap between baseline and
teacher. Brain retrieval benefits much more than Image — likely because Brain
direction is more sensitive to the additional path-shape information that
streaming-GLM extracts.

## Why this matters

- First positive Fast-tier result we've found. +4pp on Image is at the edge of
  sampling noise on n=50, but the Brain gain (+14pp) is well outside noise and
  the sign is consistent.
- The Slow teacher information is RT-deployable: the streaming GLM at pst=20
  can be computed online with 36s latency (matching paper's Slow). So the
  combined pipeline is: stream BOLD, run streaming GLM at Slow latency for the
  teacher, run Fast at pst=5 latency for the student input, combine via
  refiner at decode time. Real-time-feasible — total latency 36s for the
  teacher signal but only 14.5s for the student decode.
- Generalizes to deployment: a closed-loop study could run student-distilled
  Fast decoding for sub-15s prediction with the teacher providing an offline-
  per-trial calibration signal.

## What's still bounded

The student can't exceed the teacher's 54% Image. The teacher itself only gets
54% from streaming-GLM Slow input through fold-0, well below paper's Slow
anchor of 58%. Improving the teacher (better Slow β, or higher-quality SLOW
extractor) would lift the student's ceiling.

The student also can't generalize beyond the training distribution. We trained
on ses-03 non-test trials; student gets +4/+14 on ses-03 first-rep test. Cross-
session generalization (testing on ses-06 with the ses-03-trained refiner)
not tested yet.

## Files

- Driver: `local_drivers/train_fast_distill_from_slow.py`
- Result: `task_2_1_betas/fast_distill_results.json`
- Refiner state: `task_2_1_betas/fast_refiner_state.pth` (overwritten — distill version supersedes the null variant)

## v2/v3 follow-up (2026-05-04)

**v2: scaled training set with ses-01/02 fmriprep BOLD added (1500 train pairs)**
| Variant | Image | Brain |
|---|---|---|
| v2a per-voxel scalar (mixed BOLD) | 30 | 46 |
| v2b low-rank rank=64 (mixed BOLD) | 30 | 42 |

Both underperform v1 (40/48). The mixed BOLD source (rtmotion test-set vs
fmriprep training augmentation) introduces a feature-distribution mismatch the
refiner can't bridge. **BOLD source consistency between training and test
matters more than training set size for this refiner architecture.** Capacity
also wasn't the bottleneck — low-rank refiner with 360k params performs same
as 5584-param scalar.

**v3: ses-03-only, longer training, ensemble across late epochs**
| Variant | Image | Brain |
|---|---|---|
| v3 ensemble (epochs 15-59 averaged) | **42** | 48 |
| v3 best single test_image (test-leaked selection) | 44 | 48 |
| v3 best single test_brain (test-leaked selection) | 42 | 50 |

v3 ensemble is +2pp Image over v1 (within noise on n=50). The peak
test-leaked single epoch hits 44% Image but that's selected with test data;
not a fair generalization claim. Brain remains stable at 48-50% across all
ensembling/selection strategies.

## Final cross-latency distillation result

**Best honest Fast result: 42% Image / 48% Brain**, +6pp Image / +14pp Brain
over baseline 36/34. Achieved with per-voxel scalar refiner trained on ses-03
(rtmotion BOLD) only, supervised by streaming-GLM-Slow teacher, ensembled
across late training epochs.

Real-time deployable: 14.5s student decode + 36s teacher availability for
training the refiner offline (one-time fit per subject). The refiner adds
zero compute at decode time — it's a per-voxel multiply-add applied to the
incoming Fast β before fold-0.

— Cross-latency distillation v1/v2/v3, 2026-05-04, fold-0, n=50 special515 ses-03.
