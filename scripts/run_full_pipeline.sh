#!/bin/bash
# Full pipeline to prepare test data and run Vox2Cortex inference
# Run this on bb-compxg-01 after activating the vox2cortex conda environment

set -e

# Configuration
SUBJECT="sub-PNA001_ses-01"
FASTSURFER_DIR="/data/mica3/BIDS_PNI/derivatives/fastsurfer/${SUBJECT}"
OUTPUT_BASE="/data_/mica1/03_projects/enning/vox2cortex_test"
OUTPUT_DIR="${OUTPUT_BASE}/${SUBJECT}"
MNI_TEMPLATE="/data/mica1/01_programs/mni_icbm152_2009/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz"
VOX2CORTEX_DIR="${HOME}/Documents/Vox2Cortex"
CONDA_ENV="/data_/mica1/03_projects/enning/conda_envs/vox2cortex"

# Tool paths
FREESURFER_HOME="/data/mica1/01_programs/freesurfer-7.4.1"
ANTS_DIR="/data/mica1/01_programs/ants-2.3.4/bin"

echo "=============================================="
echo "Vox2Cortex Test Data Preparation Pipeline"
echo "=============================================="
echo "Subject: ${SUBJECT}"
echo "Output: ${OUTPUT_DIR}"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Set up FreeSurfer
export FREESURFER_HOME
source ${FREESURFER_HOME}/SetUpFreeSurfer.sh 2>/dev/null || true

# Create output directories
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

# ============================================
# STEP 1: Convert orig.mgz to NIfTI
# ============================================
echo ""
echo "=== Step 1: Convert orig.mgz to NIfTI ==="
if [ ! -f ${OUTPUT_DIR}/mri_native.nii.gz ]; then
    ${FREESURFER_HOME}/bin/mri_convert \
        ${FASTSURFER_DIR}/mri/orig.mgz \
        ${OUTPUT_DIR}/mri_native.nii.gz
    echo "Created: mri_native.nii.gz"
else
    echo "Skipping: mri_native.nii.gz already exists"
fi

# ============================================
# STEP 2: Register to MNI152 using ANTs
# ============================================
echo ""
echo "=== Step 2: Register to MNI152 (ANTs affine) ==="
if [ ! -f ${OUTPUT_DIR}/mri.nii.gz ]; then
    ${ANTS_DIR}/antsRegistrationSyNQuick.sh \
        -d 3 \
        -f ${MNI_TEMPLATE} \
        -m ${OUTPUT_DIR}/mri_native.nii.gz \
        -o ${OUTPUT_DIR}/mri2mni_ \
        -t a \
        -n 4
    
    # Rename the warped image
    mv ${OUTPUT_DIR}/mri2mni_Warped.nii.gz ${OUTPUT_DIR}/mri.nii.gz
    echo "Created: mri.nii.gz"
else
    echo "Skipping: mri.nii.gz already exists"
fi

# ============================================
# STEP 3: Convert and register segmentation
# ============================================
echo ""
echo "=== Step 3: Convert and register segmentation ==="
if [ ! -f ${OUTPUT_DIR}/aseg.nii.gz ]; then
    ${FREESURFER_HOME}/bin/mri_convert \
        ${FASTSURFER_DIR}/mri/aseg.mgz \
        ${OUTPUT_DIR}/aseg_native.nii.gz
    
    ${ANTS_DIR}/antsApplyTransforms \
        -d 3 \
        -i ${OUTPUT_DIR}/aseg_native.nii.gz \
        -r ${MNI_TEMPLATE} \
        -o ${OUTPUT_DIR}/aseg.nii.gz \
        -t ${OUTPUT_DIR}/mri2mni_0GenericAffine.mat \
        -n NearestNeighbor
    echo "Created: aseg.nii.gz"
else
    echo "Skipping: aseg.nii.gz already exists"
fi

# ============================================
# STEP 4: Convert surfaces to scanner coords
# ============================================
echo ""
echo "=== Step 4: Convert surfaces to scanner coordinates ==="
for surf in lh.pial lh.white rh.pial rh.white; do
    name=$(echo $surf | tr '.' '_')
    if [ ! -f ${OUTPUT_DIR}/${name}.scanner ]; then
        ${FREESURFER_HOME}/bin/mris_convert \
            --to-scanner \
            ${FASTSURFER_DIR}/surf/${surf} \
            ${OUTPUT_DIR}/${name}.scanner
        echo "Created: ${name}.scanner"
    else
        echo "Skipping: ${name}.scanner already exists"
    fi
done

# ============================================
# STEP 5: Transform surfaces to MNI space
# ============================================
echo ""
echo "=== Step 5: Transform surfaces to MNI space ==="
python ${VOX2CORTEX_DIR}/scripts/transform_surfaces.py ${OUTPUT_DIR}

# ============================================
# STEP 6: Create test file list
# ============================================
echo ""
echo "=== Step 6: Create test file list ==="
echo "${SUBJECT}" > ${OUTPUT_BASE}/test_ids.txt
echo "Created: ${OUTPUT_BASE}/test_ids.txt"

# ============================================
# Verify output
# ============================================
echo ""
echo "=== Output files ==="
ls -la ${OUTPUT_DIR}/

echo ""
echo "=== Checking required files for Vox2Cortex ==="
required_files=(
    "mri.nii.gz"
    "aseg.nii.gz"
    "lh_pial_reduced_0.3.ply"
    "lh_white_reduced_0.3.ply"
    "rh_pial_reduced_0.3.ply"
    "rh_white_reduced_0.3.ply"
)
all_present=true
for f in "${required_files[@]}"; do
    if [ -f ${OUTPUT_DIR}/${f} ]; then
        echo "✓ ${f}"
    else
        echo "✗ ${f} MISSING"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    echo ""
    echo "=============================================="
    echo "Data preparation complete!"
    echo "Ready for inference."
    echo "=============================================="
else
    echo ""
    echo "ERROR: Some required files are missing!"
    exit 1
fi
