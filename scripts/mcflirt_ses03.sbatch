#!/bin/bash
#SBATCH --job-name=mcflirt-ses03
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# RT motion correction of sub-005 ses-03 matching the paper's Section 2.3.2:
#   1. Session start: FLIRT(vol0_of_run01, boldref) → 6-DOF T_bold2boldref
#   2. Per run:       MCFLIRT(4D BOLD, refvol=session-level vol0) → T_hmc
#   3. Compose:       T_total = T_bold2boldref ∘ T_hmc  (applied via applywarp)
#   4. Split:         4D → per-TR  ses-03_run-XX_NNNN_mc_boldres.nii.gz
#
# Output format matches what scripts/benchmark_variants.py and
# scripts/task_2_1_factorial.py already expect, so condition B
# (RT_motion + GLMsingle_HRF) becomes just "add the flag" after this runs.
#
# All FSL binaries are native arm64 at $FSLDIR/bin/.

set -euo pipefail

export FSLDIR=/home/mhough/fsl
export PATH=$FSLDIR/bin:$PATH
# Rishab 4/14: NIFTI (uncompressed) saves ~200-300 ms per volume gzip write
# during RT. Keep NIFTI_GZ here because we write once to long-term storage.
export FSLOUTPUTTYPE=NIFTI_GZ

SESSION=ses-03
RAW_BIDS=/data/3t/data/raw_bids/sub-005/$SESSION/func
BOLDREF=/data/3t/data/sub-005_ses-01_task-C_run-01_space-T1w_boldref.nii.gz
OUT_DIR=/data/3t/derivatives/motion_corrected_resampled
WORK=/tmp/mcflirt_work_${SLURM_JOB_ID:-local}
LOGS=/data/derivatives/rtmindeye_paper/logs

mkdir -p "$OUT_DIR" "$WORK" "$LOGS"

# --- Step 1: session-level BOLD → boldref registration (one-time) -----------
SESS_VOL0="$WORK/ses03_run01_vol0.nii.gz"
SESS_MAT="$WORK/ses03_to_boldref.mat"
echo "=== $SESSION: compute session-level BOLD→boldref transform ==="
fslroi "$RAW_BIDS/sub-005_${SESSION}_task-C_run-01_bold.nii.gz" "$SESS_VOL0" 0 1
flirt -in "$SESS_VOL0" -ref "$BOLDREF" -omat "$SESS_MAT" \
      -dof 6 -cost normmi -interp spline
echo "  saved $SESS_MAT"

# --- Step 2: per-run MCFLIRT + resample to boldref -------------------------
for run in 01 02 03 04 05 06 07 08 09 10 11; do
    RAW="$RAW_BIDS/sub-005_${SESSION}_task-C_run-${run}_bold.nii.gz"
    [ -f "$RAW" ] || { echo "skip: $RAW missing"; continue; }

    echo "=== $SESSION run-$run ==="
    RUN_WORK="$WORK/run-$run"
    mkdir -p "$RUN_WORK"

    # MCFLIRT with explicit external reference = session-level vol0. This
    # matches the paper's "reference to the first functional volume of the
    # session." -plots writes a 6-col motion params file (rot/trans xyz) for
    # use as confound regressors in Variant A-nuisance if we want.
    mcflirt -in "$RAW" -out "$RUN_WORK/mc" -reffile "$SESS_VOL0" \
            -plots -cost normcorr -stages 4

    # Apply the session-level BOLD→boldref transform to the 4D MC'd output.
    # applywarp handles 4D input natively — single interpolation step.
    applywarp -i "$RUN_WORK/mc.nii.gz" -r "$BOLDREF" \
              -o "$RUN_WORK/mc_boldres.nii.gz" \
              --premat="$SESS_MAT" --interp=spline

    # Split 4D → per-TR 3D with zero-padded index
    fslsplit "$RUN_WORK/mc_boldres.nii.gz" "$RUN_WORK/vol_" -t

    # Rename to the convention expected by downstream scripts:
    #   <session>_run-<run>_<NNNN>_mc_boldres.nii.gz
    N=0
    for f in "$RUN_WORK"/vol_*.nii.gz; do
        idx=$(printf "%04d" $N)
        mv "$f" "$OUT_DIR/${SESSION}_run-${run}_${idx}_mc_boldres.nii.gz"
        N=$((N + 1))
    done
    echo "  produced $N TRs at $OUT_DIR"

    # Archive the motion parameters for Variant A-nuisance use
    cp "$RUN_WORK/mc.nii.gz.par" \
       "$OUT_DIR/${SESSION}_run-${run}_motion.par" 2>/dev/null || true
done

# Keep the working dir on failure for debugging; only clean on success.
rm -rf "$WORK"

echo "=== done ==="
ls "$OUT_DIR/${SESSION}_run-"*_mc_boldres.nii.gz | wc -l
echo "TRs produced above should equal 11 * 192 = 2112 if all runs were present."
