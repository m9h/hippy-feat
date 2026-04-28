#!/bin/bash
# Re-run mcflirt to capture motion params (.par files) only — frames already exist.
set -euo pipefail

export FSLDIR=/Users/mhough/fsl
export PATH=$FSLDIR/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ

SESSION=ses-03
PAPER_ROOT=/Users/mhough/Workspace/data/rtmindeye_paper
RAW_BIDS=$PAPER_ROOT/rt3t/data/raw_bids/sub-005/$SESSION/func
OUT_DIR=$PAPER_ROOT/motion_corrected_resampled
WORK=/tmp/mcflirt_par_only
LOGS=$PAPER_ROOT/logs/mcflirt_par
HERE=$(dirname "$(realpath "$0")")

mkdir -p "$OUT_DIR" "$WORK" "$LOGS"

SESS_VOL0="$WORK/${SESSION}_run01_vol0.nii.gz"
echo "=== $SESSION: extract session-level vol0 ==="
fslroi "$RAW_BIDS/sub-005_${SESSION}_task-C_run-01_bold.nii.gz" "$SESS_VOL0" 0 1

PIDS=()
for run in 01 02 03 04 05 06 07 08 09 10 11; do
    fsl_sub -N "mcpar_${SESSION}_${run}" -l "$LOGS" \
        "$HERE/mcflirt_par_only_one_run.sh" "$SESSION" "$run" \
        "$SESS_VOL0" "$WORK" "$OUT_DIR" "$RAW_BIDS" &
    pid=$!
    PIDS+=($pid)
    echo "  spawned run-${run} (pid $pid)"
done

echo "=== waiting for ${#PIDS[@]} jobs ==="
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "  pid $pid exited non-zero"
done

echo "=== done ==="
ls "$OUT_DIR"/*_motion.par 2>/dev/null | wc -l
echo "expected 11"

rm -rf "$WORK"
