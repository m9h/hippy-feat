#!/bin/bash
# Per-run mcflirt that ONLY captures motion params (.par), skipping the
# applywarp + fslsplit work since the mc_boldres frames already exist.
set -euo pipefail

SESSION=$1
RUN=$2
SESS_VOL0=$3
WORK_ROOT=$4
OUT_DIR=$5
RAW_BIDS=$6

export FSLDIR=/Users/mhough/fsl
export PATH=$FSLDIR/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ

RAW="$RAW_BIDS/sub-005_${SESSION}_task-C_run-${RUN}_bold.nii.gz"
[ -f "$RAW" ] || { echo "skip: $RAW missing"; exit 0; }

RUN_WORK="$WORK_ROOT/run-$RUN"
mkdir -p "$RUN_WORK"

mcflirt -in "$RAW" -out "$RUN_WORK/mc" -reffile "$SESS_VOL0" \
        -plots -cost normcorr -stages 4

# mcflirt -plots writes <out>.par next to the .nii.gz
PAR_SRC="$RUN_WORK/mc.par"
PAR_DST="$OUT_DIR/${SESSION}_run-${RUN}_motion.par"
if [ -f "$PAR_SRC" ]; then
    cp "$PAR_SRC" "$PAR_DST"
    echo "[run-${RUN}] saved $(wc -l < "$PAR_DST") TR rows of motion params"
else
    echo "[run-${RUN}] ERROR: $PAR_SRC not found"
    exit 1
fi
