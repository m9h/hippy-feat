#!/bin/bash
# Per-run motion correction. Args: session, run (zero-padded), session_vol0, session_mat, work_root, out_dir, raw_bids, boldref
set -euo pipefail

SESSION=$1
RUN=$2
SESS_VOL0=$3
SESS_MAT=$4
WORK_ROOT=$5
OUT_DIR=$6
RAW_BIDS=$7
BOLDREF=$8

export FSLDIR=/Users/mhough/fsl
export PATH=$FSLDIR/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ

RAW="$RAW_BIDS/sub-005_${SESSION}_task-C_run-${RUN}_bold.nii.gz"
[ -f "$RAW" ] || { echo "skip: $RAW missing"; exit 0; }

RUN_WORK="$WORK_ROOT/run-$RUN"
mkdir -p "$RUN_WORK"

mcflirt -in "$RAW" -out "$RUN_WORK/mc" -reffile "$SESS_VOL0" \
        -plots -cost normcorr -stages 4

applywarp -i "$RUN_WORK/mc.nii.gz" -r "$BOLDREF" \
          -o "$RUN_WORK/mc_boldres.nii.gz" \
          --premat="$SESS_MAT" --interp=spline

fslsplit "$RUN_WORK/mc_boldres.nii.gz" "$RUN_WORK/vol_" -t

N=0
for f in "$RUN_WORK"/vol_*.nii.gz; do
    idx=$(printf "%04d" $N)
    mv "$f" "$OUT_DIR/${SESSION}_run-${RUN}_${idx}_mc_boldres.nii.gz"
    N=$((N + 1))
done

cp "$RUN_WORK/mc.nii.gz.par" \
   "$OUT_DIR/${SESSION}_run-${RUN}_motion.par" 2>/dev/null || true

echo "[run-${RUN}] $N TRs"
