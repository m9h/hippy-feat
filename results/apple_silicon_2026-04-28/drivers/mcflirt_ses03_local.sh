#!/bin/bash
# Local-Mac dispatcher that mirrors scripts/mcflirt_ses03.sbatch.
# Submits the 11 per-run MCFLIRT jobs in parallel via fsl_sub (shell method).

set -euo pipefail

export FSLDIR=/Users/mhough/fsl
export PATH=$FSLDIR/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ

SESSION=ses-03
PAPER_ROOT=/Users/mhough/Workspace/data/rtmindeye_paper
RAW_BIDS=$PAPER_ROOT/rt3t/data/raw_bids/sub-005/$SESSION/func
BOLDREF=$PAPER_ROOT/rt3t/data/sub-005_ses-01_task-C_run-01_space-T1w_boldref.nii.gz
OUT_DIR=$PAPER_ROOT/motion_corrected_resampled
WORK=/tmp/mcflirt_work_local
LOGS=$PAPER_ROOT/logs/mcflirt
HERE=$(dirname "$(realpath "$0")")

mkdir -p "$OUT_DIR" "$WORK" "$LOGS"

SESS_VOL0="$WORK/${SESSION}_run01_vol0.nii.gz"
SESS_MAT="$WORK/${SESSION}_to_boldref.mat"
echo "=== $SESSION: session-level BOLD->boldref transform ==="
fslroi "$RAW_BIDS/sub-005_${SESSION}_task-C_run-01_bold.nii.gz" "$SESS_VOL0" 0 1
flirt -in "$SESS_VOL0" -ref "$BOLDREF" -omat "$SESS_MAT" \
      -dof 6 -cost normmi -interp spline
echo "  saved $SESS_MAT"

# fsl_sub shell method runs synchronously; background each submission so all
# 11 runs spawn at once (well within 18 cores; mcflirt itself is single-threaded).
PIDS=()
for run in 01 02 03 04 05 06 07 08 09 10 11; do
    fsl_sub -N "mc_${SESSION}_${run}" -l "$LOGS" \
        "$HERE/mcflirt_one_run.sh" "$SESSION" "$run" \
        "$SESS_VOL0" "$SESS_MAT" "$WORK" "$OUT_DIR" "$RAW_BIDS" "$BOLDREF" &
    pid=$!
    PIDS+=($pid)
    echo "  spawned run-${run} (pid $pid)"
done

echo "=== waiting for ${#PIDS[@]} jobs ==="
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "  pid $pid exited non-zero"
done
echo "=== done ==="
ls "$OUT_DIR/${SESSION}_run-"*_mc_boldres.nii.gz | wc -l
echo "expected ~2112 TRs (11 runs * 192)"

rm -rf "$WORK"
