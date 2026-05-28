# Notes & questions for Charles Martin

*From a WeightWatcher analysis of 6 EEG foundation models in NeuralBench-EEG v1.0, run 2026-05-25. Full report: [`WEIGHTWATCHER_EEGFM_ANALYSIS.md`](./WEIGHTWATCHER_EEGFM_ANALYSIS.md).*

## Context

I'm working on mechanistic interpretability of pretrained EEG transformers (sparse-autoencoder probes over frozen activations à la BrainCapture's *Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders*, arXiv:2605.13930). To choose which model to focus on and which layer to extract from, I ran WeightWatcher across all 6 EEG foundation models in Meta's recently-released NeuralBench-EEG v1.0 benchmark.

The results are clean and interesting, and three of them raise questions I think only you can answer.

## What was measured

`weightwatcher.WeightWatcher(model).analyze(min_evals=50)` on the published checkpoints. No fine-tuning, no eval data — just the released weights:

| Model | Source | Params | α-mean | α-med | %α<2 | %[2,6] | %α>6 |
|---|---|---:|---:|---:|---:|---:|---:|
| LUNA    | PulpBio/LUNA                            | 40.5 M  | 3.93 | 3.64 | 10.4% | 79.1% | 10.4% |
| LaBraM  | braindecode/labram-pretrained           |  5.8 M  | 3.76 | 3.63 |  4.2% | 93.8% | ~1% |
| REVE    | brain-bzh/reve-base                     | 69.2 M  | 3.61 | 3.20 |  5.6% | 87.6% | ~7% |
| CBraMod | braindecode/cbramod-pretrained          |  4.9 M  | 3.32 | 2.96 | 26.7% | 68.0% | 5.3% |
| BIOT    | braindecode/biot-pretrained-six-datasets|  3.2 M  | 2.48 | 2.13 | 44.0% | 52.0% | 4.0% |
| BENDR   | braindecode/braindecode-bendr           | **157 M** | **2.10** | 1.79 | **68.0%** | 24.0% | 8.0% |

REVE was also broken down per transformer block (12 blocks × 4 matrices = 48 of the 89 layers analysed). Per-block details on request.

## Three findings I think you'd find interesting

### 1. Under-trained matrices are uniformly attention; no FFs

REVE has 5 matrices with α<2 across its 12 transformer blocks. Every one of them is an attention `to_qkv` or `to_out`. Zero feed-forward (`net.1`, `net.3`) matrices are under-trained:

```
transformer.layers.2.0.to_qkv   α=1.56
transformer.layers.1.0.to_qkv   α=1.58
transformer.layers.0.0.to_out   α=1.72
transformer.layers.3.0.to_qkv   α=1.81
transformer.layers.10.0.to_out  α=1.92
```

The REVE paper (El Ouahidi et al., NeurIPS '25) prescribes "2-step linear-probe + LoRA on QKVO" as the empirically-found fine-tuning recipe. WeightWatcher independently recovers the same recipe from the weights alone, and adds quantitative guidance on *which blocks* yield the most (early blocks 0–3 + block 10).

**Q1.** Is the attention-vs-FF asymmetry a pattern you've seen in pretrained NLP transformers? Is there an HT-SR-theoretical reason that feed-forward layers reach the healthy regime first during pretraining, or is this an artifact of how attention gradients distribute?

### 2. The "block-11 paradox" — WW says healthy, SAE on activations says rank-collapsed

This is the finding I'd most like your take on.

- REVE block 11's weight matrices: α-mean **3.46**, all 4 matrices in the healthy [2,6] band.
- A TopK Sparse Autoencoder trained on block 11's *activations* (7.4 M REVE-output tokens from HBN RestingState, d_dict=2048, k=32): **81% dead features, EV = 1.000**. The activations span only ~370 directions, dramatically below the model's 512 hidden dim.

Same pattern, even stronger, on LaBraM's final block 11: α-mean 2.53 (lowest in LaBraM, 25% under-trained per HT-SR) — i.e. the final transformer block is the *worst* block in the entire model by weight-spectrum analysis.

Reading: a transformer's weights can be HT-SR-healthy while the activation distribution flowing *through* those weights has progressively collapsed onto a small task-specific subspace by the final block. The two diagnostics answer different questions: weight α is a static property of the trained matrices; activation rank is a dynamic property of the data distribution under those weights.

**Q2.** Is there theoretical support in HT-SR / SETOL for distinguishing "well-distributed weights" from "richly-populated activations"? Should WW alpha be paired with an activation-side diagnostic — perhaps the effective rank of layer outputs on the pretraining distribution — when choosing which layer to probe or fine-tune? I realize the formalism is data-free by design, but I'm curious where in your taxonomy this case falls.

**Q3.** Is the *final* transformer block being the worst block (LaBraM's case) a known phenomenon? My intuition is that the last block specializes to the pretraining-task projection and collapses the representation, but that doesn't fit cleanly with HT-SR — its weights should still look healthy if they were trained well. LaBraM's saying they're not.

### 3. Pretraining hours predicts α-mean, but with one clean outlier

```
                pretrain hours   params    α-mean   %α<2   %healthy
BENDR           ~3k              157 M     2.10     68%    24%
BIOT            few-k            3.2 M     2.48     44%    52%
CBraMod         varied            4.9 M    3.32     27%    68%
REVE            ~60k             69 M      3.61      6%    88%
LaBraM          ~2.5k            5.8 M     3.76      4%    94%   ← outlier
LUNA            ~60k+            40 M      3.93     10%    79%
```

The ordering is roughly monotonic in pretraining-hours × pretraining-quality — except LaBraM, which gets REVE-quality alpha with about an order of magnitude less pretraining data and a tenth the parameter count.

Architectural differences that *might* explain LaBraM's outlier behaviour:
- LaBraM uses VQ-tokenisation of EEG channel-patches → masked-token prediction (similar to BEiT). REVE/LUNA use masked autoencoding directly on continuous EEG.
- LaBraM is small enough that the available data may saturate its capacity. The bigger models may need more data to reach the same density of training.

**Q4.** Does HT-SR provide any way to *quantify* training efficiency — i.e. given the alpha distribution of a checkpoint, can you back out something like "this model has reached X% of its asymptotic capacity"? It seems like the SETOL extension might support this and I'd love your perspective on whether the LaBraM data point is a meaningful efficiency claim or noise within the power-law fit.

### Bonus observation: over-parameterised (α>6) matrices cluster on attention-output projections

REVE block 5 `to_out` matrix has α=9.627. LUNA has 10.4% of matrices with α>6. From the per-matrix CSVs, these outliers appear to be predominantly `to_out` / `proj` (attention output projections) rather than QKV or FF.

**Q5.** Are α>6 matrices effectively low-rank, in the sense that their SVD shows a sharp cliff well below nominal dimension? Could WW alpha > 6 be reframed as a *quantitative prescription* for safe low-rank factorisation (which would let practitioners shave a meaningful fraction of inference parameters without retraining)?

## Tooling notes — possible PR contributions

While running the analysis I hit three friction points that might be worth fixing upstream. Happy to PR any of these:

1. **`nn.utils.weight_norm` parametrisations**: BENDR errored with `AttributeError: 'ParametrizationList' object has no attribute 'data'`. Fix is `nn.utils.parametrize.remove_parametrizations(..., leave_parametrized=True)` before analysis. WW could detect parametrised weights and either materialise them automatically or emit a clearer error.

2. **`nn.UninitializedParameter` / `nn.LazyLinear`**: braindecode's CBraMod uses lazy modules; WW fails with "Attempted to use an uninitialized parameter" inside safetensors when reading metadata. A dummy forward initialises them. Could WW detect and offer a short fix message?

3. **Architecture-agnostic state-dict input**: For two models (CBraMod, LUNA), the public default constructor doesn't match the released checkpoint's `d_model`, breaking the normal `model = Cls(**kwargs); load_state_dict(...)` flow. I worked around it by wrapping every 2D+ tensor in the state_dict as a synthetic `nn.Linear` and feeding *that* to WW — architecture-agnostic, very robust. **Proposed helper**: `weightwatcher.from_safetensors("model.safetensors")` or `weightwatcher.from_state_dict(sd)`. Especially valuable for analysing arbitrary HF safetensors releases without chasing config metadata.

## Validation

**Q6.** Across models, my ranking by α-mean (LUNA 3.93 / LaBraM 3.76 / REVE 3.61) puts three models within 0.32 of each other. For pretrained foundation-model selection, what's the smallest α-mean gap you'd call "actionably different"?

**Q7.** EEG transformers are trained on much lower-dimensional and lower-rate inputs than NLP tokens. Do you have priors on whether HT-SR alpha distributions in EEG foundation models should look systematically different from NLP ones — for instance, a higher fraction of over-parameterised matrices because the data manifold is smaller?

## Artifacts

If useful for you to inspect:
- Per-matrix CSVs for each model, with `longname / alpha / log_norm / N / M`
- Per-block aggregations for REVE + LaBraM
- The 5 specific under-trained matrices in REVE
- The full TopK SAE result that produced the "block-11 paradox" data point

I can share any of these. The full pipeline is reproducible from the analyzer scripts at <https://github.com/m9h/hippy-feat> (paths `scripts/analyze_reve_weightwatcher.py` and `scripts/analyze_eegfm_weightwatcher.py`).

— Morgan
