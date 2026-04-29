#!/usr/bin/env python3
"""Run scripts/prereg_benchmark.py locally on our 12 cells of betas.

Decoder-free β-reliability + H1-H5 paired bootstrap. No checkpoint, no GPU
needed. Mac-vs-DGX comparison is at the β-reliability level — independent of
the cell-11 checkpoint inflation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_benchmark as B
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
B.PREREG_ROOT = LOCAL / "task_2_1_betas" / "prereg"

# Run main()
if __name__ == "__main__":
    sys.argv = ["run_prereg_benchmark_local"]
    B.main()
