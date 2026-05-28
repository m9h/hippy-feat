#!/usr/bin/env python
"""Train a TopK SAE on a pre-extracted activation .npz.

Loads ``scripts/extract_eeg_fm_acts.py`` output, fits ``jaxoccoli.sae`` to
the activations, saves checkpoint + dictionary-health audit JSON.

Single-GPU JAX inside the NGC JAX SIF. The activation tensor is held in
host memory and shuffled per-epoch into device-batched chunks — no PyTorch
DataLoader needed.

Outputs
-------
    <out_prefix>.npz   — final SAE params + (optional) sample of reconstructions
    <out_prefix>.json  — config + final dictionary health + train loss history
    <out_prefix>_health.csv — periodic audit metrics across training
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _shim_jaxoccoli_package():
    """Pre-register ``jaxoccoli`` as a bare namespace package so submodule
    imports don't trigger ``jaxoccoli/__init__.py`` (which eagerly imports
    every nibabel/scipy/etc.-dependent module). The JAX SIF has jax/optax
    but not nibabel; we only need ``jaxoccoli.sae`` here.
    """
    import sys
    import os
    import types
    if "jaxoccoli" in sys.modules:
        return
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(project_root, "jaxoccoli")
    shim = types.ModuleType("jaxoccoli")
    shim.__path__ = [pkg_dir]
    shim.__file__ = os.path.join(pkg_dir, "__init__.py")
    sys.modules["jaxoccoli"] = shim


def train(args):
    # JAX imports are local so this script can be argparse-checked outside the SIF.
    import jax
    import jax.numpy as jnp
    import optax

    _shim_jaxoccoli_package()
    from jaxoccoli.sae import (
        make_topk_sae, make_sae_train_step, init_sae_optimizer,
        recon_loss, dictionary_health, TopKSAEParams,
        make_sae_train_step_aux_k, init_sae_train_state,
    )

    print(f"[jax] devices: {jax.devices()}", flush=True)

    # ---- load activations ------------------------------------------------
    print(f"[load] {args.activations}", flush=True)
    blob = np.load(args.activations, allow_pickle=True)
    acts_full = blob["activations"]
    N_full = acts_full.shape[0]
    d_model = acts_full.shape[1]
    if acts_full.ndim != 2:
        raise ValueError(f"expected (N, d_model) activations, got {acts_full.shape}")
    sidecar = Path(args.activations).with_suffix(".json")
    if sidecar.exists():
        ext_cfg = json.loads(sidecar.read_text())
    else:
        ext_cfg = {}

    # Hold-out split for the audit (no leakage into the training MSE).
    # If --max-tokens caps N, downsample first; otherwise the .astype(float32)
    # copy of a 50+ GB activations array will OOM the host alongside
    # x_train / x_holdout copies.
    rng = np.random.default_rng(args.seed)
    if args.max_tokens is not None and args.max_tokens < N_full:
        keep = rng.choice(N_full, size=args.max_tokens, replace=False)
        keep.sort()
        acts = np.ascontiguousarray(acts_full[keep]).astype(np.float32, copy=False)
        N = args.max_tokens
        print(f"  [downsample] {N_full} → {N} tokens (cap=--max-tokens)", flush=True)
    else:
        acts = acts_full.astype(np.float32, copy=False)
        N = N_full
    print(f"  acts: N={N}  d_model={d_model}  d_dict={args.d_dict}  k={args.k}",
          flush=True)

    perm = rng.permutation(N)
    n_holdout = min(args.n_holdout, N // 5)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    x_train = jnp.asarray(acts[train_idx])
    x_holdout = jnp.asarray(acts[holdout_idx])
    # Free the host-side numpy buffer once both halves are on device. JAX
    # holds its own copy; otherwise we keep ~N*d_model*4 bytes resident
    # in RAM for the whole training loop (~50 GB on R10).
    del acts, acts_full, blob
    print(f"  train={len(train_idx)}  holdout={len(holdout_idx)}", flush=True)

    # ---- init SAE --------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    params, _ = make_topk_sae(
        d_model=d_model, d_dict=args.d_dict, k=args.k, key=key,
    )

    optimizer = optax.adam(args.lr)
    opt_state = init_sae_optimizer(params, optimizer)

    use_aux = args.aux_k > 0
    if use_aux:
        train_state = init_sae_train_state(args.d_dict)
        step_fn = jax.jit(make_sae_train_step_aux_k(
            optimizer, args.k,
            aux_k=args.aux_k,
            aux_coef=args.aux_coef,
            n_steps_to_kill=args.aux_steps_to_kill,
        ))
        print(f"[aux_k] enabled: aux_k={args.aux_k} coef={args.aux_coef} "
              f"kill_after={args.aux_steps_to_kill}", flush=True)
    else:
        step_fn = jax.jit(make_sae_train_step(optimizer, args.k))

    # ---- train loop ------------------------------------------------------
    n_train = x_train.shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    history = []                  # per-eval audit snapshots
    losses = []                   # per-step training losses
    t0 = time.time()
    for epoch in range(args.epochs):
        order = jax.random.permutation(jax.random.fold_in(key, epoch), n_train)
        for s in range(steps_per_epoch):
            bs = args.batch_size
            idx = order[s * bs:(s + 1) * bs]
            xb = x_train[idx]
            if use_aux:
                params, opt_state, train_state, loss, _ = step_fn(
                    params, opt_state, train_state, xb,
                )
            else:
                params, opt_state, loss = step_fn(params, opt_state, xb)
            losses.append(float(loss))

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            h = dictionary_health(params, x_holdout, args.k)
            h["epoch"] = epoch + 1
            h["wallclock"] = time.time() - t0
            h["train_loss_recent"] = float(np.mean(losses[-steps_per_epoch:]))
            history.append(h)
            print(
                f"  ep{epoch+1:03d}  loss={h['train_loss_recent']:.4e}  "
                f"recon_mse={h['recon_mse']:.4e}  "
                f"EV={h['explained_var']:.3f}  dead={h['dead_fraction']:.2%}  "
                f"l0={h['l0_actual']:.1f}", flush=True)

    final_health = history[-1] if history else dictionary_health(params, x_holdout, args.k)
    print(f"[done] {time.time() - t0:.1f}s  final EV={final_health['explained_var']:.3f}",
          flush=True)

    # ---- save ------------------------------------------------------------
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        enc_weight=np.asarray(params.enc_weight),
        enc_bias=np.asarray(params.enc_bias),
        dec_weight=np.asarray(params.dec_weight),
        dec_bias=np.asarray(params.dec_bias),
        k=np.int32(args.k),
        d_dict=np.int32(args.d_dict),
        d_model=np.int32(d_model),
    )

    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump({
            "extraction": ext_cfg,
            "config": {
                "d_model": d_model, "d_dict": args.d_dict, "k": args.k,
                "lr": args.lr, "batch_size": args.batch_size,
                "epochs": args.epochs, "n_train": int(n_train),
                "n_holdout": int(n_holdout),
            },
            "final_health": final_health,
            "history": history,
        }, f, indent=2)

    # CSV of audit history for quick plotting
    csv_path = Path(str(out_prefix) + "_health.csv")
    if history:
        keys = list(history[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for h in history:
                f.write(",".join(f"{h[k]:.6g}" for k in keys) + "\n")

    print(f"[done] wrote {out_prefix.with_suffix('.npz')}", flush=True)
    print(f"[done] wrote {out_prefix.with_suffix('.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True,
                    help="Input .npz from scripts/extract_eeg_fm_acts.py")
    ap.add_argument("--out-prefix", required=True,
                    help="Output path prefix (no extension)")
    ap.add_argument("--d-dict", type=int, default=None,
                    help="Dictionary size (default 16 * d_model, read from .npz)")
    ap.add_argument("--k", type=int, default=32,
                    help="TopK sparsity")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=2,
                    help="Run the dictionary health audit every N epochs")
    ap.add_argument("--n-holdout", type=int, default=20_000,
                    help="Cap on audit hold-out size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux-k", type=int, default=0,
                    help="aux_k dead-feature resurrection. 0 = disabled; "
                         "typical: same as --k (e.g. 32). Gao 2024 §3.3.")
    ap.add_argument("--aux-coef", type=float, default=1.0 / 32.0,
                    help="weight of the aux residual-MSE term")
    ap.add_argument("--aux-steps-to-kill", type=int, default=200,
                    help="feature is considered dead after this many steps "
                         "without firing")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Cap on training tokens (random subsample). "
                         "Use for memory-bounded sweeps on large releases. "
                         "Default: use all tokens.")
    args = ap.parse_args()

    # Late-bind d_dict from d_model if not supplied
    if args.d_dict is None:
        peek = np.load(args.activations, allow_pickle=True)
        d_model = int(peek["activations"].shape[1])
        args.d_dict = 16 * d_model
        print(f"[cfg] d_dict not given; defaulting to 16 * d_model = {args.d_dict}",
              flush=True)

    train(args)


if __name__ == "__main__":
    sys.exit(main())
