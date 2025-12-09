#!/bin/bash
# Run Vox2Cortex inference on prepared test data
# Run this after run_full_pipeline.sh completes successfully

set -e

# Configuration
VOX2CORTEX_DIR="${HOME}/Documents/Vox2Cortex"
CONDA_ENV="/data_/mica1/03_projects/enning/conda_envs/vox2cortex"

echo "=============================================="
echo "Vox2Cortex Inference"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

cd ${VOX2CORTEX_DIR}

# Run inference using pretrained model
# Using the BIDS_PNI_test dataset we added to supported_datasets.py
echo ""
echo "Running inference with V2C-Flow-S-ADNI model..."
echo "Dataset: BIDS_PNI_test"
echo ""

python vox2organ/main.py \
    --test \
    -n V2C-Flow-S-ADNI \
    --dataset BIDS_PNI_test \
    --device cuda:0

echo ""
echo "=============================================="
echo "Inference complete!"
echo "Check results in experiments/ or pretrained_models/V2C-Flow-S-ADNI/"
echo "=============================================="
