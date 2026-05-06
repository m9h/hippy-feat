#!/usr/bin/env python3
"""Reproduce Iyer et al. ICML 2026 Fig 3 + Fig A.7 from a Table-1 metrics JSON.

Consumes the JSON written by ``score_full_metrics.py`` and applies the
paper's per-metric min-max normalization:
  0 = random-COCO baseline      (Tab A.3 published values)
  1 = offline-NSD-3rep anchor   (Tab 1 row "Offline NSD (avg. 3 reps.)")

For ↓ metrics (efficientnet, swav) the formula is sign-flipped so that
higher normalized values are always better.

Outputs two PNGs to the same directory as the input JSON:
  - fig3_3group_bars.png      (low / high / retrieval × pipeline)
  - figA7_per_metric_bars.png (10 bars × pipeline)

Usage:
    python plot_table1_normalized_bars.py \\
        --json /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/full_metrics_fold10_table1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper anchors (hand-copied from main.tex, Table 1 + Tab A.3 row)
# ---------------------------------------------------------------------------
COCO_BASELINE = {  # Tab A.3 "Random Baseline" row
    "pixcorr":              0.014,
    "ssim":                 0.277,
    "alexnet2":             0.5025,
    "alexnet5":             0.5099,
    "inception":            0.5037,
    "clip":                 0.5038,
    "efficientnet":         0.982,
    "swav":                 0.655,
    "image_retrieval_top1": 0.016,
    "brain_retrieval_top1": 0.014,
}
NSD_ANCHOR = {  # Tab 1 "Offline NSD (avg. 3 reps.)" row
    "pixcorr":              0.231,
    "ssim":                 0.349,
    "alexnet2":             0.884,
    "alexnet5":             0.956,
    "inception":            0.859,
    "clip":                 0.788,
    "efficientnet":         0.803,
    "swav":                 0.423,
    "image_retrieval_top1": 1.00,
    "brain_retrieval_top1": 0.96,
}
DIRECTION = {  # +1 = ↑ (higher better), -1 = ↓ (lower better)
    "pixcorr":              +1, "ssim":             +1,
    "alexnet2":             +1, "alexnet5":         +1,
    "inception":            +1, "clip":             +1,
    "efficientnet":         -1, "swav":             -1,
    "image_retrieval_top1": +1, "brain_retrieval_top1": +1,
}
GROUPS = {
    "Low-level":  ["pixcorr", "ssim", "alexnet2", "alexnet5"],
    "High-level": ["inception", "clip", "efficientnet", "swav"],
    "Retrieval":  ["image_retrieval_top1", "brain_retrieval_top1"],
}
ORDERED_METRICS = (GROUPS["Low-level"] + GROUPS["High-level"]
                   + GROUPS["Retrieval"])
PRETTY = {
    "pixcorr": "PixCorr", "ssim": "SSIM",
    "alexnet2": "AlexNet(2)", "alexnet5": "AlexNet(5)",
    "inception": "Inception", "clip": "CLIP",
    "efficientnet": "EffNet", "swav": "SwAV",
    "image_retrieval_top1": "Image ret.",
    "brain_retrieval_top1": "Brain ret.",
}

# Cell name → display label for Table-1-style ordering.
# `delay` is in TRIALS, not TRs (see docs/table1_reproducibility_recipe.md
# Pitfall #9): delay=0 → Fast (default, ~7.5 s post-stim), delay=5 → Slow,
# delay=63 → End-of-Run.
CELL_LABELS = {
    "Probe_canonical_sessztrain_repavg":   "Offline 3T (avg 3 reps)",
    "Probe_canonical_sessztrain_firstrep": "Offline 3T",
    "Paper_RT_actual_delay63":             "End-of-run RT",
    "Paper_RT_actual_delay5":              "Slow RT",
    "Paper_RT_actual_delay0":              "Fast RT",
}
CELL_ORDER = [  # match Table 1 row ordering (best → worst)
    "Probe_canonical_sessztrain_repavg",
    "Probe_canonical_sessztrain_firstrep",
    "Paper_RT_actual_delay63",
    "Paper_RT_actual_delay5",
    "Paper_RT_actual_delay0",
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize(metric: str, score: float) -> float:
    """Min-max normalize one metric value: COCO → 0, NSD → 1."""
    coco, nsd = COCO_BASELINE[metric], NSD_ANCHOR[metric]
    if DIRECTION[metric] > 0:
        denom = nsd - coco
        return (score - coco) / denom if denom else 0.0
    denom = coco - nsd
    return (coco - score) / denom if denom else 0.0


def metric_value(entry: dict, metric: str) -> tuple[float, float]:
    """Extract (mean, std) for a metric. Retrieval has no std so std=0."""
    if metric in ("image_retrieval_top1", "brain_retrieval_top1"):
        return float(entry[metric]), 0.0
    return float(entry[f"{metric}_mean"]), float(entry[f"{metric}_std"])


def normalize_with_seeds(entry: dict, metric: str) -> tuple[float, float]:
    """Per-seed normalize → mean ± SE (matches paper's "5-seed errorbar")."""
    if metric in ("image_retrieval_top1", "brain_retrieval_top1"):
        return normalize(metric, float(entry[metric])), 0.0
    per_seed = entry[f"{metric}_per_seed"]
    norms = np.asarray([normalize(metric, v) for v in per_seed])
    return float(norms.mean()), float(norms.std() / np.sqrt(len(norms)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def fig3_3group(entries: dict, out_path: Path) -> None:
    """3-group bar chart: Low / High / Retrieval × pipeline."""
    cells = [c for c in CELL_ORDER if c in entries]
    group_names = list(GROUPS.keys())
    n_groups, n_cells = len(group_names), len(cells)
    width = 0.8 / n_cells
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, cell in enumerate(cells):
        means, ses = [], []
        for grp in group_names:
            per_metric = [normalize_with_seeds(entries[cell], m)
                          for m in GROUPS[grp]]
            ms = np.asarray([p[0] for p in per_metric])
            es = np.asarray([p[1] for p in per_metric])
            means.append(ms.mean())
            # group-level SE: combine per-metric SEs in quadrature, /N_metrics
            ses.append(np.sqrt((es ** 2).sum()) / len(per_metric))
        offset = (i - (n_cells - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=ses,
               label=CELL_LABELS.get(cell, cell),
               color=cmap(0.15 + 0.7 * i / max(n_cells - 1, 1)),
               capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Normalized score (0 = random COCO, 1 = offline NSD)")
    ax.set_title("Aggregated evaluation metrics — 3T offline + RT pipelines")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axhline(1, color="0.4", lw=0.5, ls="--")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}", flush=True)


def figA7_per_metric(entries: dict, out_path: Path) -> None:
    """Per-metric bar chart (10 bars × pipeline) — Fig A.7 layout."""
    cells = [c for c in CELL_ORDER if c in entries]
    n_metrics, n_cells = len(ORDERED_METRICS), len(cells)
    width = 0.8 / n_cells
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, cell in enumerate(cells):
        means, ses = [], []
        for m in ORDERED_METRICS:
            mu, se = normalize_with_seeds(entries[cell], m)
            means.append(mu)
            ses.append(se)
        offset = (i - (n_cells - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=ses,
               label=CELL_LABELS.get(cell, cell),
               color=cmap(0.15 + 0.7 * i / max(n_cells - 1, 1)),
               capsize=2)

    # Group dividers
    boundaries = [
        len(GROUPS["Low-level"]) - 0.5,
        len(GROUPS["Low-level"]) + len(GROUPS["High-level"]) - 0.5,
    ]
    for b in boundaries:
        ax.axvline(b, color="0.6", ls="--", lw=0.7)
    ax.text(1.5, ax.get_ylim()[1] * 0.96, "Low-level",
            ha="center", fontsize=9, color="0.3")
    ax.text(5.5, ax.get_ylim()[1] * 0.96, "High-level",
            ha="center", fontsize=9, color="0.3")
    ax.text(8.5, ax.get_ylim()[1] * 0.96, "Retrieval",
            ha="center", fontsize=9, color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in ORDERED_METRICS], rotation=20)
    ax.set_ylabel("Normalized score (0 = random COCO, 1 = offline NSD)")
    ax.set_title("Per-metric evaluation — 3T offline + RT pipelines")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axhline(1, color="0.4", lw=0.5, ls="--")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="full_metrics_*.json from "
                    "score_full_metrics.py")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: same as --json)")
    args = ap.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out_dir) if args.out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(json_path.read_text())
    entries = {e["cell"]: e for e in raw if "error" not in e}
    if not entries:
        raise SystemExit(f"no scorable entries in {json_path}")
    print(f"loaded {len(entries)} cells: {list(entries)}", flush=True)

    fig3_3group(entries, out_dir / "fig3_3group_bars.png")
    figA7_per_metric(entries, out_dir / "figA7_per_metric_bars.png")


if __name__ == "__main__":
    main()
