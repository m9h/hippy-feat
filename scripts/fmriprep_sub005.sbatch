#!/bin/bash
#SBATCH --job-name=fmriprep-sub005
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/data/derivatives/fmriprep_sub005/logs/%x-%j.out

# Sized to fit alongside a concurrent smri-fm job on the GPU partition
# that holds 10 CPUs + 64 GB. 8 CPUs + 40 GB fMRIPrep leaves node
# headroom and lets both jobs run simultaneously. Walltime bumped to
# 48h because fewer cores = longer wall-clock.

# fMRIPrep 24.1.1 on sub-005 ses-01/02/06 to produce training-prior
# compatible motion-corrected BOLDs in T1w space.
#
# GPU is not used: fMRIPrep is CPU-bound (mcflirt / flirt / ANTs / FS).
# Scheduled on the CPU partition so GPU remains free for other jobs.
#
# Prerequisite: FreeSurfer license at FS_LICENSE (already on disk).

set -euo pipefail

BIDS_DIR=/data/3t/data/raw_bids
OUT_DIR=/data/derivatives/fmriprep_sub005
WORK_DIR=/tmp/fmriprep_work_${SLURM_JOB_ID:-local}
SIF=/data/derivatives/containers/fmriprep_24.1.1.sif
FS_LICENSE=/data/raw/config/freesurfer-license.txt

mkdir -p "$OUT_DIR/logs" "$WORK_DIR" "$(dirname "$SIF")"

if [ ! -f "$SIF" ]; then
    echo "Pulling fMRIPrep 24.1.1 apptainer image…"
    apptainer pull "$SIF" docker://nipreps/fmriprep:24.1.1
fi

apptainer run --cleanenv \
    -B "$BIDS_DIR":/bids:ro \
    -B "$OUT_DIR":/out \
    -B "$WORK_DIR":/work \
    -B "$FS_LICENSE":/opt/freesurfer/license.txt:ro \
    "$SIF" \
    /bids /out participant \
    --participant-label 005 \
    --task-id C \
    --output-spaces T1w:res-2 \
    --nthreads 8 --omp-nthreads 4 --mem-mb 36000 \
    --skip-bids-validation \
    --stop-on-first-crash \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    -v

rm -rf "$WORK_DIR"
echo "Done. Outputs at $OUT_DIR/fmriprep/sub-005/"
