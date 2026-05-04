# Log-signature feature experiments — overnight results

Three phases probing whether log-signature features (per-voxel depth-2 increment + Lévy area
over the post-stim BOLD window) can lift Fast/Slow single-rep retrieval above the β-only
baseline. All phases use the frozen fold-0 ckpt.

Baselines (rtmotion AR(1) LSS β + cum-z, single-rep, 50 special515):
- Fast (pst=5): ~36% Image (paper 36%)
- Slow (pst=20): ~44% Image (paper 58%)

## Phase A — zero-training: feature replacement & α-mixing

Tests whether Lévy-area or depth-1 increment carries retrieval signal alone, and
whether a manual mixing α can improve `β + α·feature`.

### Fast
- β alone: **36.0%** Image
- Lévy alone: 0.0% Image
- increment alone: 20.0% Image
- best β+α·Lévy: α=+0.1 → 38.0% (Δ +2.0pp)
- best β+α·increment: α=+0.1 → 38.0% (Δ +2.0pp)

### Slow
- β alone: **44.0%** Image
- Lévy alone: 0.0% Image
- increment alone: 0.0% Image
- best β+α·Lévy: α=-1.0 → 46.0% (Δ +2.0pp)
- best β+α·increment: α=-0.1 → 46.0% (Δ +2.0pp)

## Phase B — train per-voxel projector with 3 features (β, increment, Lévy)

### Fast
- baseline (β alone): 36.0%
- after training (final epoch): **24.0%** (Δ -12.0pp)
- best val Image during training: 36.0%
- n_train: 543, n_test: 50, elapsed: 3.9 min

### Slow
- baseline (β alone): 44.0%
- after training (final epoch): **32.0%** (Δ -12.0pp)
- best val Image during training: 46.0%
- n_train: 543, n_test: 50, elapsed: 0.9 min

## Phase C — train per-voxel projector with 9 features + early stopping

Features per voxel: β, increment, Lévy area, mean, std, max, min, range, slope.
Hidden dim 16. Early stop on val Image accuracy (patience 15, max 80 epochs).

### Fast
- baseline (β alone): 36.0%
- after training (best-val checkpoint): **36.0%** (Δ +0.0pp)
- best val Image during training: 40.7%
- n_train: 462, n_val: 81, n_test: 50, elapsed: 1.2 min

### Slow
- baseline (β alone): 44.0%
- after training (best-val checkpoint): **44.0%** (Δ +0.0pp)
- best val Image during training: 44.4%
- n_train: 462, n_val: 81, n_test: 50, elapsed: 0.5 min

## Verdict

- Phase B: Fast Δ=-12.0pp, Slow Δ=-12.0pp
- Phase C: Fast Δ=+0.0pp, Slow Δ=+0.0pp

If both phases B and C show Δ < +4pp on Fast and Slow, log-signature features (at the per-voxel scalar interface) do not add deployment-relevant signal beyond what the existing AR(1) LSS β captures. The path-shape information either isn't there in the post-stim BOLD window, or the (2792,)-scalar interface to fold-0 destroys it.