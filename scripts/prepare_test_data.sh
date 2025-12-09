#!/bin/bash
# Prepare test data for Vox2Cortex inference
# This script prepares data from BIDS_PNI fastsurfer outputs

set -e

# Configuration
SUBJECT="sub-PNA001_ses-01"
FASTSURFER_DIR="/data/mica3/BIDS_PNI/derivatives/fastsurfer/${SUBJECT}"
OUTPUT_DIR="/data_/mica1/03_projects/enning/vox2cortex_test"
MNI_TEMPLATE="/data/mica1/01_programs/mni_icbm152_2009/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"

# Tool paths
FREESURFER_HOME="/data/mica1/01_programs/freesurfer-7.4.1"
ANTS_DIR="/data/mica1/01_programs/ants-2.3.4/bin"

# Set up FreeSurfer
export FREESURFER_HOME
source ${FREESURFER_HOME}/SetUpFreeSurfer.sh

# Create output directories
mkdir -p ${OUTPUT_DIR}/${SUBJECT}
cd ${OUTPUT_DIR}/${SUBJECT}

echo "=== Step 1: Convert orig.mgz to NIfTI ==="
${FREESURFER_HOME}/bin/mri_convert \
    ${FASTSURFER_DIR}/mri/orig.mgz \
    ${OUTPUT_DIR}/${SUBJECT}/mri_native.nii.gz

echo "=== Step 2: Register to MNI152 using ANTs ==="
# Use ANTs for affine registration to MNI
${ANTS_DIR}/antsRegistrationSyNQuick.sh \
    -d 3 \
    -f ${MNI_TEMPLATE} \
    -m ${OUTPUT_DIR}/${SUBJECT}/mri_native.nii.gz \
    -o ${OUTPUT_DIR}/${SUBJECT}/mri2mni_ \
    -t a \
    -n 4

# Rename the warped image to mri.nii.gz (what Vox2Cortex expects)
mv ${OUTPUT_DIR}/${SUBJECT}/mri2mni_Warped.nii.gz ${OUTPUT_DIR}/${SUBJECT}/mri.nii.gz

echo "=== Step 3: Convert segmentation ==="
${FREESURFER_HOME}/bin/mri_convert \
    ${FASTSURFER_DIR}/mri/aseg.mgz \
    ${OUTPUT_DIR}/${SUBJECT}/aseg_native.nii.gz

# Apply transform to segmentation
${ANTS_DIR}/antsApplyTransforms \
    -d 3 \
    -i ${OUTPUT_DIR}/${SUBJECT}/aseg_native.nii.gz \
    -r ${MNI_TEMPLATE} \
    -o ${OUTPUT_DIR}/${SUBJECT}/aseg.nii.gz \
    -t ${OUTPUT_DIR}/${SUBJECT}/mri2mni_0GenericAffine.mat \
    -n NearestNeighbor

echo "=== Step 4: Convert and transform surfaces ==="
SURFACES=("lh.pial" "lh.white" "rh.pial" "rh.white")
SURF_NAMES=("lh_pial" "lh_white" "rh_pial" "rh_white")

for i in ${!SURFACES[@]}; do
    surf=${SURFACES[$i]}
    name=${SURF_NAMES[$i]}
    
    echo "Processing surface: ${surf} -> ${name}"
    
    # Convert to scanner coordinates
    ${FREESURFER_HOME}/bin/mris_convert \
        --to-scanner \
        ${FASTSURFER_DIR}/surf/${surf} \
        ${OUTPUT_DIR}/${SUBJECT}/${name}.scanner
done

echo "=== Step 5: Transform surfaces to MNI space (Python) ==="
# This requires a Python script to apply the affine transform to surfaces

echo "Done preparing data structure!"
echo "Output: ${OUTPUT_DIR}/${SUBJECT}/"
ls -la ${OUTPUT_DIR}/${SUBJECT}/
