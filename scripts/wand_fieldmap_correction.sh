#!/usr/bin/env bash
# WAND fieldmap-based distortion correction for resting fMRI
#
# Pipeline: magnitude brain extraction -> fsl_prepare_fieldmap -> FUGUE unwarping
# Uses dual-echo GRE fieldmap (TE1=4.92ms, TE2=7.38ms)
#
# Usage: bash scripts/wand_fieldmap_correction.sh [SUB] [SES]
#   Defaults: sub-08033 ses-03

set -euo pipefail

# --- Configuration ---
SUB="${1:-sub-08033}"
SES="${2:-ses-03}"

WAND="/Users/mhough/dev/wand"
FMAP_DIR="${WAND}/${SUB}/${SES}/fmap"
FUNC_DIR="${WAND}/${SUB}/${SES}/func"
DERIV="${WAND}/derivatives/fsl-fmri/${SUB}/${SES}"
OUT_DIR="${DERIV}/fieldmap_correction"

# Fieldmap parameters (from JSON sidecars)
DELTA_TE=2.46  # ms (TE2 - TE1 = 7.38 - 4.92)

# BOLD parameters (from JSON sidecar)
EPI_ECHO_SPACING=0.000274998  # seconds
EPI_TE=0.03  # seconds
UNWARP_DIR="y-"  # PhaseEncodingDirection j- = y-

# Input files
MAG1="${FMAP_DIR}/${SUB}_${SES}_magnitude1.nii.gz"
MAG2="${FMAP_DIR}/${SUB}_${SES}_magnitude2.nii.gz"
PHASE="${FMAP_DIR}/${SUB}_${SES}_phasediff.nii.gz"
BOLD_MC="${DERIV}/mc/task-rest_mc.nii.gz"
BOLD_RAW="${FUNC_DIR}/${SUB}_${SES}_task-rest_bold.nii.gz"

echo "============================================================"
echo "WAND Fieldmap Distortion Correction"
echo "============================================================"
echo "Subject: ${SUB}"
echo "Session: ${SES}"
echo "Delta TE: ${DELTA_TE} ms"
echo "EPI echo spacing: ${EPI_ECHO_SPACING} s"
echo "Unwarp direction: ${UNWARP_DIR}"
echo ""

# --- Create output directory ---
mkdir -p "${OUT_DIR}"

# ============================================================
# Step 1: Brain extraction of magnitude image
# ============================================================
echo "[1/6] Brain extraction of magnitude image..."

# Use magnitude1 (shorter TE = better SNR)
# BET with -f 0.5 and -B for bias field reduction
if [ ! -f "${OUT_DIR}/mag_brain.nii.gz" ]; then
    bet "${MAG1}" "${OUT_DIR}/mag_brain" -f 0.4 -B
    echo "  -> ${OUT_DIR}/mag_brain.nii.gz"
else
    echo "  -> Already exists, skipping"
fi

# Also erode mask slightly to avoid edge effects
if [ ! -f "${OUT_DIR}/mag_brain_mask_ero.nii.gz" ]; then
    fslmaths "${OUT_DIR}/mag_brain_mask.nii.gz" -ero "${OUT_DIR}/mag_brain_mask_ero.nii.gz"
    echo "  -> Eroded mask created"
fi

# ============================================================
# Step 2: Prepare fieldmap (rad/s)
# ============================================================
echo ""
echo "[2/6] Preparing fieldmap (phase diff -> rad/s)..."

if [ ! -f "${OUT_DIR}/fmap_rads.nii.gz" ]; then
    fsl_prepare_fieldmap SIEMENS \
        "${PHASE}" \
        "${OUT_DIR}/mag_brain.nii.gz" \
        "${OUT_DIR}/fmap_rads.nii.gz" \
        "${DELTA_TE}"
    echo "  -> ${OUT_DIR}/fmap_rads.nii.gz"
else
    echo "  -> Already exists, skipping"
fi

# ============================================================
# Step 3: Smooth fieldmap (median + Gaussian)
# ============================================================
echo ""
echo "[3/6] Smoothing fieldmap..."

if [ ! -f "${OUT_DIR}/fmap_rads_smooth.nii.gz" ]; then
    # Median filter (3mm) to remove spikes
    fugue --loadfmap="${OUT_DIR}/fmap_rads.nii.gz" \
          --savefmap="${OUT_DIR}/fmap_rads_smooth.nii.gz" \
          --median \
          --smooth3=2
    echo "  -> ${OUT_DIR}/fmap_rads_smooth.nii.gz"
else
    echo "  -> Already exists, skipping"
fi

# ============================================================
# Step 4: Register fieldmap to EPI
# ============================================================
echo ""
echo "[4/6] Registering fieldmap to EPI space..."

# Extract mean EPI as reference
if [ ! -f "${OUT_DIR}/mean_epi.nii.gz" ]; then
    fslmaths "${BOLD_MC}" -Tmean "${OUT_DIR}/mean_epi.nii.gz"
    echo "  -> Mean EPI extracted"
fi

# Register magnitude to EPI (6 DOF rigid body)
if [ ! -f "${OUT_DIR}/mag2epi.mat" ]; then
    flirt -in "${OUT_DIR}/mag_brain.nii.gz" \
          -ref "${OUT_DIR}/mean_epi.nii.gz" \
          -omat "${OUT_DIR}/mag2epi.mat" \
          -out "${OUT_DIR}/mag_brain_epi.nii.gz" \
          -dof 6 \
          -cost corratio
    echo "  -> Registration: mag -> EPI"
fi

# Apply registration to fieldmap
if [ ! -f "${OUT_DIR}/fmap_rads_epi.nii.gz" ]; then
    flirt -in "${OUT_DIR}/fmap_rads_smooth.nii.gz" \
          -ref "${OUT_DIR}/mean_epi.nii.gz" \
          -applyxfm -init "${OUT_DIR}/mag2epi.mat" \
          -out "${OUT_DIR}/fmap_rads_epi.nii.gz"
    echo "  -> Fieldmap registered to EPI space"
fi

# Also transform the mask
if [ ! -f "${OUT_DIR}/mag_brain_mask_epi.nii.gz" ]; then
    flirt -in "${OUT_DIR}/mag_brain_mask_ero.nii.gz" \
          -ref "${OUT_DIR}/mean_epi.nii.gz" \
          -applyxfm -init "${OUT_DIR}/mag2epi.mat" \
          -out "${OUT_DIR}/mag_brain_mask_epi.nii.gz" \
          -interp nearestneighbour
    # Threshold to binary
    fslmaths "${OUT_DIR}/mag_brain_mask_epi.nii.gz" -thr 0.5 -bin "${OUT_DIR}/mag_brain_mask_epi.nii.gz"
    echo "  -> Mask registered to EPI space"
fi

# ============================================================
# Step 5: FUGUE unwarping of motion-corrected BOLD
# ============================================================
echo ""
echo "[5/6] Applying FUGUE distortion correction to BOLD..."

if [ ! -f "${OUT_DIR}/task-rest_mc_dc.nii.gz" ]; then
    fugue -i "${BOLD_MC}" \
          --loadfmap="${OUT_DIR}/fmap_rads_epi.nii.gz" \
          --mask="${OUT_DIR}/mag_brain_mask_epi.nii.gz" \
          --dwell="${EPI_ECHO_SPACING}" \
          --unwarpdir="${UNWARP_DIR}" \
          -u "${OUT_DIR}/task-rest_mc_dc.nii.gz"
    echo "  -> ${OUT_DIR}/task-rest_mc_dc.nii.gz"
else
    echo "  -> Already exists, skipping"
fi

# ============================================================
# Step 6: Also correct the mean EPI for QC
# ============================================================
echo ""
echo "[6/6] Correcting mean EPI for quality check..."

if [ ! -f "${OUT_DIR}/mean_epi_dc.nii.gz" ]; then
    fugue -i "${OUT_DIR}/mean_epi.nii.gz" \
          --loadfmap="${OUT_DIR}/fmap_rads_epi.nii.gz" \
          --mask="${OUT_DIR}/mag_brain_mask_epi.nii.gz" \
          --dwell="${EPI_ECHO_SPACING}" \
          --unwarpdir="${UNWARP_DIR}" \
          -u "${OUT_DIR}/mean_epi_dc.nii.gz"
    echo "  -> ${OUT_DIR}/mean_epi_dc.nii.gz"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Corrected BOLD:    ${OUT_DIR}/task-rest_mc_dc.nii.gz"
echo "  Corrected mean:    ${OUT_DIR}/mean_epi_dc.nii.gz"
echo "  Fieldmap (rad/s):  ${OUT_DIR}/fmap_rads_epi.nii.gz"
echo "  Mag brain (EPI):   ${OUT_DIR}/mag_brain_epi.nii.gz"
echo ""
echo "QC: Compare mean_epi.nii.gz vs mean_epi_dc.nii.gz"
echo "    fsleyes ${OUT_DIR}/mean_epi.nii.gz ${OUT_DIR}/mean_epi_dc.nii.gz &"
echo ""
echo "To re-run FC analysis on corrected data:"
echo "  Set BOLD_MC='${OUT_DIR}/task-rest_mc_dc.nii.gz' in the analysis scripts"
