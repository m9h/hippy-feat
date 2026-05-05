#!/usr/bin/env python3
"""Read logsig_phase_{A,B,C}.json, write a consolidated summary doc."""
from __future__ import annotations
import json
from pathlib import Path

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
DOCS = Path("/Users/mhough/Workspace/hippy-feat/results/apple_silicon_2026-04-28")

def safe_load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        return {"_error": str(e)}

A = safe_load(LOCAL / "task_2_1_betas/logsig_phase_A.json")
B = safe_load(LOCAL / "task_2_1_betas/logsig_phase_B.json")
C = safe_load(LOCAL / "task_2_1_betas/logsig_phase_C.json")

lines = []
lines.append("# Log-signature feature experiments — overnight results")
lines.append("")
lines.append("Three phases probing whether log-signature features (per-voxel depth-2 increment + Lévy area")
lines.append("over the post-stim BOLD window) can lift Fast/Slow single-rep retrieval above the β-only")
lines.append("baseline. All phases use the frozen fold-0 ckpt.")
lines.append("")
lines.append("Baselines (rtmotion AR(1) LSS β + cum-z, single-rep, 50 special515):")
lines.append("- Fast (pst=5): ~36% Image (paper 36%)")
lines.append("- Slow (pst=20): ~44% Image (paper 58%)")
lines.append("")

# Phase A
lines.append("## Phase A — zero-training: feature replacement & α-mixing")
lines.append("")
lines.append("Tests whether Lévy-area or depth-1 increment carries retrieval signal alone, and")
lines.append("whether a manual mixing α can improve `β + α·feature`.")
lines.append("")
if "_error" in A:
    lines.append(f"**ERROR**: {A['_error']}")
else:
    for tier in ("Fast", "Slow"):
        if tier not in A:
            lines.append(f"### {tier}: missing"); continue
        r = A[tier]
        base = r["beta_alone"]["image"]
        levy_alone = r["levy_alone"]["image"]
        inc_alone = r["increment_alone"]["image"]
        best_levy_mix = max(r["beta_plus_alpha_levy"].items(), key=lambda kv: kv[1]["image"])
        best_inc_mix = max(r["beta_plus_alpha_increment"].items(), key=lambda kv: kv[1]["image"])
        lines.append(f"### {tier}")
        lines.append(f"- β alone: **{base*100:.1f}%** Image")
        lines.append(f"- Lévy alone: {levy_alone*100:.1f}% Image")
        lines.append(f"- increment alone: {inc_alone*100:.1f}% Image")
        lines.append(f"- best β+α·Lévy: α={best_levy_mix[0]} → {best_levy_mix[1]['image']*100:.1f}% (Δ {(best_levy_mix[1]['image']-base)*100:+.1f}pp)")
        lines.append(f"- best β+α·increment: α={best_inc_mix[0]} → {best_inc_mix[1]['image']*100:.1f}% (Δ {(best_inc_mix[1]['image']-base)*100:+.1f}pp)")
        lines.append("")

# Phase B
lines.append("## Phase B — train per-voxel projector with 3 features (β, increment, Lévy)")
lines.append("")
if "_error" in B:
    lines.append(f"**ERROR**: {B['_error']}")
else:
    for tier in ("Fast", "Slow"):
        if tier not in B:
            lines.append(f"### {tier}: missing"); continue
        r = B[tier]
        base = r["baseline_image"]; final = r["final_image"]; best = r["best_val_image"]
        lines.append(f"### {tier}")
        lines.append(f"- baseline (β alone): {base*100:.1f}%")
        lines.append(f"- after training (final epoch): **{final*100:.1f}%** (Δ {(final-base)*100:+.1f}pp)")
        lines.append(f"- best val Image during training: {best*100:.1f}%")
        lines.append(f"- n_train: {r['n_train']}, n_test: {r['n_test']}, elapsed: {r['elapsed_min']:.1f} min")
        lines.append("")

# Phase C
lines.append("## Phase C — train per-voxel projector with 9 features + early stopping")
lines.append("")
lines.append("Features per voxel: β, increment, Lévy area, mean, std, max, min, range, slope.")
lines.append("Hidden dim 16. Early stop on val Image accuracy (patience 15, max 80 epochs).")
lines.append("")
if "_error" in C:
    lines.append(f"**ERROR**: {C['_error']}")
else:
    for tier in ("Fast", "Slow"):
        if tier not in C:
            lines.append(f"### {tier}: missing"); continue
        r = C[tier]
        base = r["baseline_image"]; final = r["final_image"]; best = r["best_val_image"]
        lines.append(f"### {tier}")
        lines.append(f"- baseline (β alone): {base*100:.1f}%")
        lines.append(f"- after training (best-val checkpoint): **{final*100:.1f}%** (Δ {(final-base)*100:+.1f}pp)")
        lines.append(f"- best val Image during training: {best*100:.1f}%")
        lines.append(f"- n_train: {r['n_train']}, n_val: {r['n_val']}, n_test: {r['n_test']}, elapsed: {r['elapsed_min']:.1f} min")
        lines.append("")

# Verdict
lines.append("## Verdict")
lines.append("")
def best_delta(j):
    out = {}
    if "_error" in j: return out
    for tier in ("Fast", "Slow"):
        if tier in j:
            out[tier] = (j[tier].get("final_image", j[tier].get("beta_alone", {}).get("image", 0))
                         - j[tier].get("baseline_image", j[tier].get("beta_alone", {}).get("image", 0))) * 100
    return out

for label, data in [("Phase B", B), ("Phase C", C)]:
    if "_error" in data: continue
    deltas = []
    for tier in ("Fast", "Slow"):
        if tier in data:
            deltas.append(f"{tier} Δ={(data[tier]['final_image']-data[tier]['baseline_image'])*100:+.1f}pp")
    lines.append(f"- {label}: {', '.join(deltas)}")
lines.append("")
lines.append("If both phases B and C show Δ < +4pp on Fast and Slow, log-signature features (at the per-voxel scalar interface) do not add deployment-relevant signal beyond what the existing AR(1) LSS β captures. The path-shape information either isn't there in the post-stim BOLD window, or the (2792,)-scalar interface to fold-0 destroys it.")

out = DOCS / "LOGSIG_RESULTS.md"
out.write_text("\n".join(lines))
print(f"wrote {out}")
