"""TopK Sparse Autoencoder for mechanistic interpretability of foundation models.

Trained on frozen activations of an EEG/fMRI foundation model (REVE, LaBraM,
TRIBEv2, …), a TopK SAE learns an overcomplete dictionary in which each
example's latent code has exactly ``k`` nonzero entries. Compared to L1
SAEs, TopK avoids the shrinkage / dead-feature trade-off and admits a single
hyperparameter — ``k`` — that the *dictionary health audit* in this module
selects automatically.

References
----------
Gao L et al. (2024). Scaling and evaluating sparse autoencoders.
    arXiv:2406.04093  (OpenAI's TopK SAE).
BrainCapture (2026). Mechanistic Interpretability of EEG Foundation Models
    via Sparse Autoencoders. arXiv:2605.13930. This module is a clean-room
    JAX re-implementation; their code is PolyForm-Noncommercial and their
    corpus is private.

Factory: ``make_topk_sae(d_model, d_dict, k, key)`` → ``(params, forward_fn)``
matching the jaxoccoli factory idiom (NamedTuple + pure-JAX forward).

The ``k`` hyperparameter is held in the closure of the returned
``forward_fn`` (and not in the pytree) so it stays a Python int — that is
required by ``jax.lax.top_k`` and by JIT-time control flow in
``topk_sparsify``. Module-level helpers (``sae_forward``, ``recon_loss``,
``dictionary_health``) take ``k`` as an explicit argument.

Auxiliary functions:
    - ``sae_encode(params, x)``      — pre-TopK affine encoder output
    - ``topk_sparsify(z, k)``        — keep top-k by magnitude, zero rest
    - ``sae_decode(params, z)``      — affine decoder
    - ``sae_forward(params, x, k)``  — encode → top-k → decode
    - ``recon_loss(params, x, k)``   — mean-squared reconstruction error
    - ``dictionary_health(params, activations, k) → dict`` — BrainCapture-style audit
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Parameter container (k is intentionally NOT in here — see module docstring)
# ---------------------------------------------------------------------------

class TopKSAEParams(NamedTuple):
    """Learnable parameters of a TopK Sparse Autoencoder.

    Attributes
    ----------
    enc_weight : (d_dict, d_model) — encoder weight matrix.
    enc_bias   : (d_dict,)         — encoder bias.
    dec_weight : (d_model, d_dict) — decoder weight matrix.
    dec_bias   : (d_model,)        — decoder bias (subtracted from x before encoding).
    """

    enc_weight: jnp.ndarray
    enc_bias: jnp.ndarray
    dec_weight: jnp.ndarray
    dec_bias: jnp.ndarray


# ---------------------------------------------------------------------------
# Core ops
# ---------------------------------------------------------------------------

def topk_sparsify(z: jnp.ndarray, k: int) -> jnp.ndarray:
    """Keep the top-``k`` entries of ``z`` along the last axis, zero the rest.

    Selection is by signed value (not absolute) following Gao et al. 2024 —
    the encoder is expected to produce nonneg activations through a ReLU
    that has already been applied. For raw signed inputs, callers should
    ReLU first; ``sae_encode`` does this.

    Implementation note: a naive ``one_hot(top_idx, d_dict) * top_vals``
    materialises a ``(..., k, d_dict)`` tensor — for typical SAE sizes
    (batch=4096, k=32, d_dict=8192) that's ~4 GB per forward and twice
    again under autodiff, so it OOMs the GPU. We instead scatter into a
    ``zeros_like(z)`` of shape ``(..., d_dict)`` using ``.at[].set(...)``,
    which keeps memory at ``O(N * d_dict)`` instead of ``O(N * k * d_dict)``.

    Parameters
    ----------
    z : (..., d_dict)  pre-sparsification latents
    k : int (static)   number of nonzeros to retain per token

    Returns
    -------
    z_sparse : (..., d_dict)  with exactly ``k`` nonzeros per token
    """
    # k must be a Python int — jax.lax.top_k and the if-branch below need it static.
    if k <= 0:
        return jnp.zeros_like(z)
    d_dict = z.shape[-1]
    if k >= d_dict:
        return z

    # Flatten leading dims so the scatter is a simple 2-D op.
    lead_shape = z.shape[:-1]
    z_flat = z.reshape(-1, d_dict)
    top_vals, top_idx = jax.lax.top_k(z_flat, k)   # (N, k), (N, k)
    n = z_flat.shape[0]
    row_idx = jnp.arange(n)[:, None]               # (N, 1) — broadcasts over k
    z_sparse_flat = jnp.zeros_like(z_flat).at[row_idx, top_idx].set(top_vals)
    return z_sparse_flat.reshape(*lead_shape, d_dict)


def sae_encode(params: TopKSAEParams, x: jnp.ndarray) -> jnp.ndarray:
    """Encoder: ``ReLU( W_enc (x - b_dec) + b_enc )``.

    Subtracting the decoder bias before encoding is the standard SAE
    formulation (Bricken et al. 2023) — it makes ``b_dec`` learn the
    activation mean so the latents only have to explain residual structure.

    Returns
    -------
    z_pre : (..., d_dict)  pre-TopK latents (post-ReLU).
    """
    centered = x - params.dec_bias
    pre = centered @ params.enc_weight.T + params.enc_bias
    return jax.nn.relu(pre)


def sae_decode(params: TopKSAEParams, z: jnp.ndarray) -> jnp.ndarray:
    """Decoder: ``W_dec z + b_dec``.

    Returns
    -------
    x_hat : (..., d_model)  reconstruction.
    """
    return z @ params.dec_weight.T + params.dec_bias


def sae_forward(params: TopKSAEParams, x: jnp.ndarray, k: int
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Full forward: encode → top-k → decode.

    Returns
    -------
    recon   : (..., d_model)  reconstruction.
    latents : (..., d_dict)   sparse latent code (post-TopK).
    """
    z_pre = sae_encode(params, x)
    z = topk_sparsify(z_pre, k)
    recon = sae_decode(params, z)
    return recon, z


def recon_loss(params: TopKSAEParams, x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Mean-squared reconstruction error.

    With TopK sparsification no L1 / KL term is needed — sparsity is
    enforced structurally rather than by penalty.
    """
    recon, _ = sae_forward(params, x, k)
    return jnp.mean((x - recon) ** 2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_topk_sae(
    d_model: int,
    d_dict: int,
    k: int,
    *,
    key: jax.Array,
    tie_init: bool = True,
) -> tuple[TopKSAEParams, Callable[[TopKSAEParams, jnp.ndarray],
                                   tuple[jnp.ndarray, jnp.ndarray]]]:
    """Create a TopK SAE in the jaxoccoli factory style.

    Parameters
    ----------
    d_model  : input/output dimensionality (= the foundation model's hidden size).
    d_dict   : dictionary size. A typical choice is ``16 * d_model`` — the SAE
               literature places this in [4, 64] × ``d_model``. The dictionary
               health audit can be used to pick within that range.
    k        : top-k sparsity. Held constant per token; typical values are
               16–64 for d_dict in the thousands. **Must be a Python int**.
    key      : JAX PRNG key for weight init.
    tie_init : if True, initialize ``W_dec = W_enc.T``. The Bricken-style
               recipe — tied init, untied training — usually converges faster.

    Returns
    -------
    (params, forward_fn)
        ``forward_fn(params, x) -> (recon, latents)`` with ``k`` captured
        in the closure.

    Notes
    -----
    Both weight matrices are initialised with He scaling on the larger of
    (d_model, d_dict). Decoder columns are unit-normalised — the SAE
    convention that keeps "feature density" audit metrics scale-invariant.
    """
    if k > d_dict:
        raise ValueError(f"k={k} cannot exceed d_dict={d_dict}")
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    k = int(k)  # lock to Python int — top_k requires a static argument

    k_enc, k_dec, k_b = jax.random.split(key, 3)
    fan = max(d_model, d_dict)
    scale = jnp.sqrt(2.0 / fan)

    enc_weight = jax.random.normal(k_enc, (d_dict, d_model)) * scale
    if tie_init:
        dec_weight = enc_weight.T
    else:
        dec_weight = jax.random.normal(k_dec, (d_model, d_dict)) * scale

    # Unit-normalise decoder columns (= dictionary atoms).
    dec_norms = jnp.linalg.norm(dec_weight, axis=0, keepdims=True) + 1e-8
    dec_weight = dec_weight / dec_norms
    if tie_init:
        enc_weight = dec_weight.T

    enc_bias = jnp.zeros((d_dict,))
    dec_bias = jax.random.normal(k_b, (d_model,)) * 1e-3

    params = TopKSAEParams(
        enc_weight=enc_weight,
        enc_bias=enc_bias,
        dec_weight=dec_weight,
        dec_bias=dec_bias,
    )

    def forward_fn(params: TopKSAEParams, x: jnp.ndarray
                   ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return sae_forward(params, x, k)

    return params, forward_fn


# ---------------------------------------------------------------------------
# Dictionary health audit
# ---------------------------------------------------------------------------

def dictionary_health(
    params: TopKSAEParams,
    activations: jnp.ndarray,
    k: int,
    *,
    dead_threshold: int = 0,
) -> dict:
    """Audit a trained SAE on a held-out activation batch.

    The BrainCapture paper's "intrinsic dictionary health" metric set —
    used to pick ``k`` and ``d_dict`` so they transfer across encoders
    without per-model tuning.

    Parameters
    ----------
    activations    : (N, d_model)  held-out activation batch.
    k              : top-k sparsity used at forward time.
    dead_threshold : a feature is "dead" if it fires on this many or
                     fewer of the ``N`` tokens. Default 0 = literally never.

    Returns
    -------
    dict with:
        dead_fraction  : fraction of features that fire ≤ ``dead_threshold``
                         times across the batch. Lower is healthier; > 20%
                         is a red flag.
        mean_density   : average fraction of tokens for which each feature
                         is active. Should be close to ``k / d_dict``.
        max_density    : the most-firing feature's density. > 50% means a
                         "wrecking-ball" feature is dominating reconstruction.
        l0_actual      : empirical average L0 (nonzeros per token). Should
                         equal ``k`` exactly for TopK; deviation indicates
                         numerical issues.
        recon_mse      : reconstruction MSE on the held-out batch.
        explained_var  : 1 - Var(x - x_hat) / Var(x). Compatible with the
                         BrainCapture audit's "variance explained" tile.
    """
    activations = jnp.asarray(activations)
    recon, latents = sae_forward(params, activations, k)

    fired = latents > 0                                             # (N, d_dict)
    fire_counts = fired.sum(axis=0)                                 # (d_dict,)
    densities = fire_counts / activations.shape[0]                  # (d_dict,)
    dead_mask = fire_counts <= dead_threshold
    l0 = fired.sum(axis=-1).mean()

    err = activations - recon
    recon_mse = jnp.mean(err ** 2)
    var_x = jnp.var(activations)
    explained_var = 1.0 - jnp.var(err) / (var_x + 1e-12)

    return {
        "dead_fraction": float(jnp.mean(dead_mask)),
        "mean_density": float(jnp.mean(densities)),
        "max_density": float(jnp.max(densities)),
        "l0_actual": float(l0),
        "recon_mse": float(recon_mse),
        "explained_var": float(explained_var),
    }


# ---------------------------------------------------------------------------
# aux_k dead-feature resurrection (Gao et al. 2024 §3.3)
# ---------------------------------------------------------------------------
# Pure TopK collapses onto a small subdictionary that's "good enough" — the
# remaining ~80-95% of atoms never fire and never learn. aux_k breaks the
# stalemate by giving the dead atoms an explicit gradient: pick the top
# k_aux of the DEAD features (by their pre-TopK encoder output), use only
# them to reconstruct the residual (x - x_hat_main), and add a small
# fraction of that aux MSE to the loss. The dead features now learn
# whatever residual structure the main subdictionary missed.
#
# Dead-feature tracking uses an int "last_fired_step" array of length
# d_dict; a feature is dead if (current_step - last_fired_step) exceeds
# ``n_steps_to_kill``. The mask is recomputed every step inside JIT.


def aux_k_recon_loss(
    params: TopKSAEParams,
    x: jnp.ndarray,
    k: int,
    aux_k: int,
    dead_mask: jnp.ndarray,
    aux_coef: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Main + aux_k reconstruction loss (Gao 2024 §3.3).

    Important: dead-feature **selection** is by PRE-ReLU activation, not
    post-ReLU. Post-ReLU values for dead features are 0 by definition, so
    selecting from a masked post-ReLU vector picks features randomly with
    zero magnitude — no useful gradient flows. Selecting on pre-ReLU lets
    even negative-pre-activation features compete on their raw projection
    score; the encoder weights for dead features then drift toward
    producing positive pre-activations, which is how features "wake up".

    Parameters
    ----------
    params, x, k : as in ``recon_loss``.
    aux_k        : how many DEAD features to wake up per token.
    dead_mask    : (d_dict,) bool — True where the feature is currently dead.
    aux_coef     : weight of the auxiliary residual MSE in the total loss.

    Returns
    -------
    total_loss        : scalar — main_mse + aux_coef * aux_mse
    fired_this_step   : (d_dict,) bool — any token activated each feature?
    main_recon_mse    : scalar (for logging)
    """
    # Encoder pre-activation (no ReLU yet) — used for BOTH paths.
    centered = x - params.dec_bias
    pre = centered @ params.enc_weight.T + params.enc_bias              # (N, d_dict)

    # Main path: standard ReLU + TopK on the post-ReLU vector.
    z_main_post_relu = jax.nn.relu(pre)
    z_main = topk_sparsify(z_main_post_relu, k)
    x_hat_main = sae_decode(params, z_main)
    main_mse = jnp.mean((x - x_hat_main) ** 2)

    # Aux path: select TOP aux_k of DEAD features by PRE-ReLU score.
    # Alive features get -inf so they can't be selected. After selection,
    # apply ReLU — features with negative pre stay at 0 (no reconstruction
    # contribution) but the gradient pathway through their encoder weight
    # is alive via the residual MSE, which is what wakes them up.
    neg_inf = jnp.float32(-jnp.inf)
    pre_dead_only = jnp.where(dead_mask, pre, neg_inf)
    z_aux_selected = topk_sparsify(pre_dead_only, aux_k)
    z_aux = jax.nn.relu(z_aux_selected)
    residual = x - x_hat_main
    residual_hat = z_aux @ params.dec_weight.T                          # no dec_bias
    aux_mse = jnp.mean((residual - residual_hat) ** 2)

    total = main_mse + aux_coef * aux_mse
    fired = (z_main > 0).any(axis=tuple(range(z_main.ndim - 1)))
    return total, fired, main_mse


class SAETrainState(NamedTuple):
    """State carried across aux_k SAE training steps.

    Attributes
    ----------
    last_fired_step : (d_dict,) int32 — step number each feature last fired.
    current_step    : () int32        — global step counter.
    """
    last_fired_step: jnp.ndarray
    current_step: jnp.ndarray


def init_sae_train_state(d_dict: int) -> SAETrainState:
    """Initial state — all features marked as having fired at step 0 so
    they get a grace window before being considered dead."""
    return SAETrainState(
        last_fired_step=jnp.zeros((d_dict,), dtype=jnp.int32),
        current_step=jnp.int32(0),
    )


def make_sae_train_step_aux_k(
    optimizer,
    k: int,
    *,
    aux_k: int | None = None,
    aux_coef: float = 1.0 / 32.0,
    n_steps_to_kill: int = 200,
) -> Callable:
    """Return a JIT-able training step closure with aux_k resurrection.

    Parameters
    ----------
    optimizer       : optax optimizer.
    k               : main TopK sparsity.
    aux_k           : aux TopK over dead features (default = k).
    aux_coef        : weight of the residual-MSE aux term (Gao default 1/32).
    n_steps_to_kill : a feature is dead if it hasn't fired in this many steps.

    Returns
    -------
    step(params, opt_state, train_state, x)
      → (new_params, new_opt_state, new_train_state, total_loss, main_mse)
    """
    import optax

    k = int(k)
    aux_k = int(aux_k if aux_k is not None else k)
    n_steps_to_kill = int(n_steps_to_kill)

    def step(params: TopKSAEParams, opt_state, train_state: SAETrainState,
             x: jnp.ndarray):
        dead_mask = (train_state.current_step - train_state.last_fired_step
                     > n_steps_to_kill)

        def loss_fn(p):
            loss, fired, main = aux_k_recon_loss(
                p, x, k, aux_k, dead_mask, aux_coef,
            )
            return loss, (fired, main)

        (loss, (fired, main_mse)), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_step = train_state.current_step + 1
        new_last_fired = jnp.where(
            fired, new_step, train_state.last_fired_step,
        )
        new_state = SAETrainState(
            last_fired_step=new_last_fired,
            current_step=new_step,
        )
        return new_params, opt_state, new_state, loss, main_mse

    return step


# ---------------------------------------------------------------------------
# Training step helper (optax-style)
# ---------------------------------------------------------------------------

def make_sae_train_step(optimizer, k: int) -> Callable:
    """Return a JIT-able training step closure compatible with optax.

    Parameters
    ----------
    optimizer : optax optimizer (e.g. ``optax.adam(1e-3)``).
    k         : top-k sparsity (captured in the closure).

    Returns
    -------
    step(params, opt_state, x) -> (new_params, new_opt_state, loss)
    """
    import optax  # local import keeps sae.py importable without optax

    k = int(k)

    def step(params: TopKSAEParams, opt_state, x: jnp.ndarray):
        loss, grads = jax.value_and_grad(recon_loss)(params, x, k)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    return step


def init_sae_optimizer(params: TopKSAEParams, optimizer):
    """Initialise an optax opt_state for the SAE params.

    Companion to ``make_sae_train_step``.
    """
    return optimizer.init(params)
