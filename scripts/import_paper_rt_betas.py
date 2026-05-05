#!/usr/bin/env python3
"""Import the paper's actual saved RT-pipeline betas as scorable cells.

Paper saves RT betas at multiple decode delays in
/data/derivatives/rtmindeye_paper/rt3t/data/real_time_betas/
as `all_betas_ses-03_all_runs_delay{N}.npy` (693 non-blank trials × 2792
voxels, raw per-trial — pre cumulative z-score).

Drops them into the prereg betas directory with synthetic cell names
`Paper_RT_actual_delay{N}` so score_AUC_factorial_dgx.py and the retrieval
pass score them under their normal causal-cumulative-z-score policy.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_RT = Path("/data/derivatives/rtmindeye_paper/rt3t/data/real_time_betas")
EVENTS = Path("/data/derivatives/rtmindeye_paper/rt3t/data/events")
PREREG = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")
SESSION = "ses-03"
DELAYS = [0, 1, 3, 5, 10, 15, 20, 63]


def reconstruct_trial_ids(session: str) -> np.ndarray:
    out: list[str] = []
    for run in range(1, 12):
        e = pd.read_csv(
            EVENTS / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv",
            sep="\t",
        )
        e = e[e["image_name"] != "blank.jpg"]
        out += e["image_name"].astype(str).tolist()
    return np.asarray(out)


def main():
    PREREG.mkdir(parents=True, exist_ok=True)
    trial_ids = reconstruct_trial_ids(SESSION)
    print(f"reconstructed {len(trial_ids)} non-blank trial IDs from events")

    for d in DELAYS:
        src = PAPER_RT / f"all_betas_{SESSION}_all_runs_delay{d}.npy"
        if not src.exists():
            print(f"  delay={d:2d}: SOURCE MISSING ({src.name})")
            continue
        betas = np.load(src).astype(np.float32)
        if betas.shape[0] != len(trial_ids):
            print(f"  delay={d:2d}: SHAPE MISMATCH {betas.shape[0]} vs "
                  f"{len(trial_ids)} trial_ids — skip")
            continue
        cell = f"Paper_RT_actual_delay{d}"
        np.save(PREREG / f"{cell}_{SESSION}_betas.npy", betas)
        np.save(PREREG / f"{cell}_{SESSION}_trial_ids.npy", trial_ids)
        n_special = int(sum(1 for t in trial_ids
                             if t.startswith("all_stimuli/special515/")))
        print(f"  delay={d:2d}: saved {cell}  betas={betas.shape}  "
              f"n_special515={n_special}")


if __name__ == "__main__":
    main()
