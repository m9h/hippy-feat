# Heavy-Tailed Self-Regularization Analysis of NeuralBench-EEG v1.0 Foundation Models

*WeightWatcher (HT-SR) per-layer α-spectrum analysis of all 6 EEG foundation models in [NeuralBench-EEG v1.0](https://github.com/facebookresearch/neuroai/tree/main/neuralbench-repo) — REVE, LaBraM, BENDR, BIOT, CBraMod, LUNA. Run 2026-05-25 on gx10-dgx-spark inside `pytorch_26.04.sif`.*

## TL;DR

| Model | Params | α-mean | %α<2 (LoRA yield) | %α in [2,6] (healthy) | Verdict |
|---|---:|---:|---:|---:|---|
| **LUNA**    | 40.5 M  | **3.93** | 10.4% | 79.1% | well-trained, heaviest tails |
| **LaBraM**  |  5.8 M  | 3.76 | **4.2%** | **93.8%** | cleanest by health fraction |
| **REVE**    | 69.2 M  | 3.61 | 5.6% | 87.6% | well-trained, large model |
| **CBraMod** |  4.9 M  | 3.32 | 26.7% | 68.0% | moderate fine-tuning yield |
| **BIOT**    |  3.2 M  | 2.48 | 44.0% | 52.0% | high fine-tuning yield |
| **BENDR**   | **157 M** | **2.10** | **68.0%** | 24.0% | severely under-trained |

**Three headline findings:**

1. **BENDR is dramatically under-trained for its size.** The *largest* model in the suite (157 M params) has 68% of its weight matrices with α<2 — borderline pretrained. Matches the literature: BENDR was trained on ~3 k hours, vs LUNA/REVE's ~60 k+ hours.
2. **LaBraM achieves the highest health fraction (94%) despite being the smallest** (5.8 M params). Pretraining quality dominates parameter count.
3. **Under-trained matrices are concentrated in attention** (verified on REVE: all 5 layers with α<2 are `to_qkv` or `to_out` matrices, no FFs). The empirical LoRA-QKVO recipe used by El Ouahidi et al. (REVE NeurIPS '25) is independently corroborated by HT-SR.

## Method

[WeightWatcher](https://weightwatcher.ai/) implements the **Heavy-Tailed Self-Regularization** theory of Martin & Mahoney (JMLR 2021, Nature Communications 2021, NeurIPS 2023 — SETOL extension). For each weight matrix `W`, it computes the empirical spectral density of `W^T W` and fits a power-law tail with exponent **α**. Per HT-SR:

- **α ∈ [2, 6]**: well-trained layer — the data implicitly regularised the weights into a healthy heavy-tailed distribution.
- **α < 2**: under-trained — the weights still look like their random initialisation. The layer will benefit from more training (or LoRA fine-tuning on a downstream task).
- **α > 6**: over-parameterised relative to the data — extra capacity that isn't being used.

Crucially, **WW requires no training/test data** — only the weights themselves. This makes it the right diagnostic for ranking pretrained foundation models without per-model evaluation infrastructure.

We ran `weightwatcher.WeightWatcher(model).analyze(min_evals=50)` on each FM. Per-matrix α values are saved to `/data/derivatives/eeg_sae/weightwatcher/{model}.csv`; cross-model summary at `all_models_summary.csv`.

## Cross-model details

### LUNA — highest mean α, heaviest tails

`PulpBio/LUNA` (`LUNA_large.safetensors`). 67 weight matrices analysed. α-mean **3.93**, median 3.64. 10.4% of matrices have α<2, 79.1% in [2,6], 10.4% above 6. **Heaviest-tailed weights of the suite** — consistent with rich feature decomposition. The 10% over-parameterised matrices mirror the same phenomenon REVE shows on block 5 (one specific attention-output matrix at α=9.6).

### LaBraM — cleanest by health fraction

`braindecode/labram-pretrained`. 48 weight matrices. α-mean 3.76, **94% in [2,6], only 4.2% under-trained**. The smallest model in the suite (5.8 M) is paradoxically the best-trained per HT-SR. ICLR 2024 spotlight; pretrained on ~2.5 k hours from ~20 datasets — fewer hours than REVE/LUNA but apparently dense enough to fully train its smaller weight budget.

### REVE — large and well-trained, per-block map available

`brain-bzh/reve-base`. 89 weight matrices across 12 transformer blocks + heads. α-mean 3.61, 87.6% in [2,6]. **Full per-block breakdown** (see "REVE per-block drill-down" below). Pretrained on 60 k hours / 25 k subjects (the largest EEG pretraining effort to date).

### CBraMod — moderate fine-tuning yield

`braindecode/cbramod-pretrained`. 75 matrices. α-mean 3.32, 68% healthy, 26.7% under-trained. A non-trivial fraction would benefit from fine-tuning, but the model is broadly competent. Used via the state-dict wrapper because the public default constructor doesn't match the released checkpoint's `d_model=576`.

### BIOT — small, half under-trained

`braindecode/biot-pretrained-six-datasets-18chs`. 25 matrices. α-mean 2.48, 44% under-trained, 52% healthy. The smallest model that's still actively used in benchmarks (3.2 M). Substantial LoRA-fine-tuning yield expected. Pretrained on six EEG datasets, modest scale.

### BENDR — large and under-trained

`braindecode/braindecode-bendr`. 25 matrices. α-mean **2.10** (below the healthy threshold), median 1.79, **68% of matrices under-trained**, only 24% healthy. Despite being the *largest* model in the suite (157 M params), it has the lowest training quality by every WeightWatcher metric. The transformer-based contextualizer atop a Conv1D encoder was pretrained on ~3 k hours of unlabeled EEG — far less than REVE/LUNA. **Almost any block would benefit from LoRA fine-tuning.** *Caveat:* required stripping `weight_norm` parametrisations before WW could read the weights — see Method caveats.

## REVE per-block drill-down

REVE's 12 blocks (`transformer.layers.0..11`), each containing 4 analysable matrices (Attention `to_qkv`, Attention `to_out`, FF `net.1`, FF `net.3`):

| block | α-mean | α-med | %α<2 | %healthy | notes |
|---:|---:|---:|---:|---:|---|
| 0 | 2.26 | 2.42 | 25% | 75% | one under-trained matrix |
| 1 | 2.45 | 2.50 | 25% | 75% | one under-trained matrix |
| 2 | 2.31 | 2.45 | 25% | 75% | one under-trained matrix |
| 3 | 2.72 | 2.76 | 25% | 75% | one under-trained matrix |
| 4 | 2.55 | 2.59 | 0% | **100%** | all healthy |
| 5 | 4.41 | 2.83 | 0% | 75% | one matrix α=9.6 (over-parameterised) |
| **6** | **3.63** | **3.48** | **0%** | **100%** | ★ our SAE extraction target |
| 7 | 2.95 | 3.03 | 0% | 100% |  |
| 8 | 3.50 | 3.46 | 0% | 100% |  |
| 9 | 3.57 | 3.38 | 0% | 100% |  |
| 10 | 3.09 | 3.33 | 25% | 75% | one under-trained matrix |
| 11 | 3.46 | 3.25 | 0% | 100% |  |

### The 5 under-trained matrices in REVE

All are attention matrices — no feed-forward (`net.*`) matrices have α<2:

```
transformer.layers.2.0.to_qkv   α=1.56
transformer.layers.1.0.to_qkv   α=1.58
transformer.layers.0.0.to_out   α=1.72
transformer.layers.3.0.to_qkv   α=1.81
transformer.layers.10.0.to_out  α=1.92
```

These are **the exact LoRA targets** the REVE paper recommends ("2-step linear-probe + LoRA on QKVO"). WeightWatcher recovers the same recipe from a *data-free* spectral diagnostic and pinpoints **which specific blocks** would benefit most (early blocks 0-3 + block 10).

## Cross-model interpretation

### Pretraining hours predicts α-mean

```
LUNA  (60k+ hours, multi-source)     α=3.93   ← best
REVE  (60k hours, 25k subjects)      α=3.61
LaBraM (~2.5k hours, 20 datasets)    α=3.76   ← outlier high; param-count efficient
CBraMod (varied)                     α=3.32
BIOT (~few k hours, 6 datasets)      α=2.48
BENDR (~3k hours, unlabeled)         α=2.10   ← worst
```

The ordering is roughly monotonic in pretraining-hours × pretraining-quality. LaBraM is the exception — it gets more from less data. We don't have a clean explanation for that, but it warrants investigation if the goal is to design new EEG foundation models efficiently.

### Param count does *not* predict training quality

```
BENDR    157 M   α=2.10   ← largest, worst-trained
REVE      69 M   α=3.61
LUNA      40 M   α=3.93   ← best alpha, mid-sized
LaBraM     6 M   α=3.76   ← smallest, second-best alpha
CBraMod    5 M   α=3.32
BIOT       3 M   α=2.48
```

Scaling without sufficient pretraining (BENDR) leaves a model under-trained at every layer. Conversely, smaller models densely trained reach high HT-SR quality.

## SAE-suitability vs LoRA-yield

These are two different practical questions, and the rankings disagree:

| Use case | Ranking (best → worst) |
|---|---|
| **SAE mechanistic interpretability** (need richly-decomposable representations) | LaBraM, LUNA, REVE, CBraMod, BIOT, BENDR |
| **LoRA fine-tuning yield** (want layers that aren't yet trained) | BENDR, BIOT, CBraMod, REVE, LUNA, LaBraM |

The "best SAE target" model has *low* %α<2; the "best LoRA target" model has *high* %α<2. They're inversely related.

### Recommended LoRA recipes

- **BENDR**: ~68% of matrices candidates. Probably train the whole contextualizer (`contextualizer.transformer_layers.*`) with rank-8 LoRA on Q/K/V projections of every layer. Highest yield by far.
- **BIOT**: 44% under-trained. Target the 4 PreNorm blocks in `encoder.transformer.layers.layers.*` — attention matrices specifically.
- **CBraMod**: 27% under-trained. Identify which specific matrices by reading `weightwatcher/cbramod.csv` and target those.
- **REVE, LaBraM, LUNA**: Sub-10% under-trained — linear-probe likely captures most of the available signal; LoRA gives diminishing returns.

## SAE-vs-WW diagnostic complementarity

A subtle finding from our SAE work: **WeightWatcher can call a layer healthy while an SAE on that layer's outputs shows rank collapse.**

Specifically: REVE block 11 has α=3.46 (healthy by WW) but our TopK SAE on its activations shows 81% dead features at d_dict=2048 with EV=1.000 — classic activation rank collapse.

The resolution: **WW analyses the weight matrices themselves; the SAE analyses the activations flowing through them.** A layer's *weights* can be well-distributed (good α) while the *data flowing through* the residual stream has collapsed onto a task-specific subspace by the final block. Both diagnostics are correct; they answer different questions.

Practical guideline: **WW tells you which layers' weights have been adequately trained. SAE tells you which layers preserve representational diversity at activation time. For SAE work, prefer the *earliest* layer that WW says is healthy** — in REVE's case, this is block 4-6. We chose block 6 empirically; WW independently validates it.

## Method caveats

1. **BENDR weight normalization.** BENDR's `contextualizer.relative_position.0.weight` and `final_layer.weight` use `nn.utils.weight_norm` parametrisations. WW errors with `AttributeError: 'ParametrizationList' object has no attribute 'data'`. Fix: call `nn.utils.parametrize.remove_parametrizations(..., leave_parametrized=True)` before WW. (Implemented in `scripts/analyze_eegfm_weightwatcher.py::_strip_parametrizations`.)

2. **CBraMod and LUNA architecture-config mismatches.** The braindecode default `CBraMod(n_outputs=1)` builds with `d_model=256` but the released checkpoint has `d_model=576`. NeuralBench's `NtLuna.build()` similarly takes a fixed channel/time-window assumption that doesn't match the released safetensors. Fix: skip the architecture entirely and use a *state-dict wrapper* (`_state_dict_wrapper` in the analyzer script) — wrap each 2D+ weight tensor as a synthetic `nn.Linear` and run WW on that. WW only cares about weight matrix shapes/values, not whether the original architecture is reconstructable.

3. **REVE gating.** `brain-bzh/reve-base` requires accepting a Responsible Use Agreement on HF before download. The analysis script reads `~/.cache/huggingface/token` via `HF_TOKEN` env var.

## Reproduction

Inside the `pytorch_26.04.sif` container:

```bash
pip install --quiet --user neuralbench[models] weightwatcher
python scripts/analyze_reve_weightwatcher.py --out-prefix .../weightwatcher/reve_base
python scripts/analyze_eegfm_weightwatcher.py  # all 6 models
```

Or via apptainer from `gx10-dgx-spark`:

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token) \
apptainer exec --no-init \
  --env HF_TOKEN --env HF_HOME=/data/derivatives/eeg_sae/hf_cache \
  --env MPLCONFIGDIR=/tmp/mpl --env PYTHONPATH=/home/mhough/dev/hippy-feat \
  -B /data:/data -B /home/mhough:/home/mhough \
  /data/derivatives/containers/pytorch_26.04.sif \
  python scripts/analyze_eegfm_weightwatcher.py
```

Artifacts land at `/data/derivatives/eeg_sae/weightwatcher/`:
- `{model}.csv` — per-matrix α values for each model
- `reve_base_by_block.csv` — REVE only, aggregated per transformer block
- `all_models_summary.csv` and `.json` — cross-model summary

## Citations

- Martin, C. H., & Mahoney, M. W. (2021). *Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning.* JMLR.
- Martin, C. H., Peng, S., & Mahoney, M. W. (2021). *Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data.* Nature Communications.
- El Ouahidi, Y. et al. (2025). *REVE: A Foundation Model for EEG — Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects.* NeurIPS.
- Jiang, W.-B. et al. (2024). *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI* (LaBraM). ICLR Spotlight.
- BrainCapture (2026). *Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders.* arXiv:2605.13930. (Used SleepFM, REVE, LaBraM but on private corpus.)
