# Vox2Cortex Setup and Usage Guide - MICA Server (bb-compxg-01)

## Overview
This guide provides step-by-step instructions for setting up and running Vox2Cortex on the MICA server (bb-compxg-01) for cortical surface reconstruction from T1-weighted MRI scans.

## Prerequisites
- Access to bb-compxg-01 server via SSH through `login.bic.mni.mcgill.ca`
- No sudo privileges required (conda environment approach)
- CUDA-compatible GPU (RTX A6000 available)

## Installation

### 1. Clone Repository
```bash
# On bb-compxg-01
cd ~/Documents
git clone https://github.com/zihuaihuai/Vox2Cortex.git
cd Vox2Cortex
```

### 2. Set Up Conda Environment

**Option A: Use Existing Environment (Recommended)**
```bash
# Try to activate the existing environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data_/mica1/03_projects/enning/conda_envs/vox2cortex

# Test if activation worked
python --version
```

If the activation succeeds, skip to step 3. If you get a permission error or the environment doesn't exist, use Option B.

**Option B: Create Your Own Environment**
```bash
# Create environment in your own directory
source ~/miniconda3/etc/profile.d/conda.sh
conda create -p ~/conda_envs/vox2cortex python=3.9 -y
conda activate ~/conda_envs/vox2cortex

# Install PyTorch with CUDA support
conda install pytorch=2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core dependencies
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install nibabel pymeshlab pandas scikit-learn matplotlib seaborn wandb

# Pin NumPy version for compatibility
pip install "numpy<2.0"
```

### 3. Install PyTorch3D with MeshesXD Patch
```bash
# Build PyTorch3D with custom MeshesXD class
cd ~/Documents/Vox2Cortex
bash setup_pytorch3d.sh
```

### 4. Verify Installation
```bash
# Set your environment path (choose based on Option A or B above)
export VOX2CORTEX_ENV="/data_/mica1/03_projects/enning/conda_envs/vox2cortex"  # or ~/conda_envs/vox2cortex

# Test basic functionality
cd ~/Documents/Vox2Cortex/vox2organ
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $VOX2CORTEX_ENV
python main.py --help
```

Expected output: Help message showing available command-line options.

## Data Preparation

### Directory Structure
Vox2Cortex expects data in BIDS format with the following structure:
```
/path/to/dataset/
├── sub-{subject}_ses-{session}/
│   ├── mri.nii.gz          # T1-weighted image in MNI space
│   ├── aseg.nii.gz         # FreeSurfer segmentation
│   ├── lh_pial.ply         # Left hemisphere pial surface
│   ├── lh_white.ply        # Left hemisphere white matter surface
│   ├── rh_pial.ply         # Right hemisphere pial surface
│   └── rh_white.ply        # Right hemisphere white matter surface
└── {dataset}_ids.txt       # List of subject IDs (one per line)
```

### Preprocessing Pipeline
For raw BIDS data, use the preprocessing scripts:

```bash
# 1. Convert FreeSurfer outputs to NIfTI and PLY
cd ~/Documents/Vox2Cortex/preprocessing/brain
bash preprocess_scans.sh /path/to/bids/dataset/sub-{subject}_ses-{session}

# 2. Register T1 to MNI152 template and transform surfaces
cd ~/Documents/Vox2Cortex
bash scripts/run_full_pipeline.sh /path/to/bids/dataset/sub-{subject}_ses-{session}
```

## Testing

### Quick Test with Preprocessed Data
```bash
# Set your environment path
export VOX2CORTEX_ENV="/data_/mica1/03_projects/enning/conda_envs/vox2cortex"  # or ~/conda_envs/vox2cortex

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $VOX2CORTEX_ENV

# Run inference on test subject
cd ~/Documents/Vox2Cortex/vox2organ
CUDA_VISIBLE_DEVICES=0 python main.py \
    --test \
    -n V2C-Flow-S-ADNI \
    --dataset BIDS_PNI_test \
    --experiment_base_dir pretrained_models/
```

### Expected Output
- Processing time: ~20 seconds per subject
- Output directory: `pretrained_models/V2C-Flow-S-ADNI/test_template_fsaverage-smooth-no-parc_BIDS_PNI_test_n_5/sub-{subject}_ses-{session}/`
- Generated files:
  - `lh_white_epoch_102.ply` - Left white matter surface
  - `lh_pial_epoch_102.ply` - Left pial surface
  - `rh_white_epoch_102.ply` - Right white matter surface
  - `rh_pial_epoch_102.ply` - Right pial surface
  - `pred_epoch_102.nii.gz` - Cortical thickness prediction volume

### Performance Metrics
Typical evaluation metrics for test subject:
- **ASSD**: ~11.0 mm (Average Surface-to-Surface Distance)
- **HD90**: ~33.2 mm (90th percentile Hausdorff Distance)
- **HD95**: ~38.1 mm (95th percentile Hausdorff Distance)
- **Self-Intersections**: ~1.3% (mesh quality)
- **VoxelDice**: ~0.92 (segmentation accuracy)

## Running on New Data

### 1. Configure Dataset Paths
Edit `vox2organ/data/supported_datasets.py`:
```python
from .supported_datasets import SupportedDatasets

# Add your dataset
SupportedDatasets.YOUR_DATASET_NAME.name: {
    'RAW_DATA_DIR': '/path/to/your/preprocessed/data',
    'FIXED_SPLIT': ["your_ids.txt", "your_ids.txt", "your_ids.txt"]
}
```

### 2. Create Subject ID File
```bash
# Create your_ids.txt with one subject per line
echo "sub-001_ses-01" > /path/to/your/data/your_ids.txt
echo "sub-002_ses-01" >> /path/to/your/data/your_ids.txt
```

### 3. Run Inference
```bash
# Set your environment path
export VOX2CORTEX_ENV="/data_/mica1/03_projects/enning/conda_envs/vox2cortex"  # or ~/conda_envs/vox2cortex

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $VOX2CORTEX_ENV

# Run on your dataset
cd ~/Documents/Vox2Cortex/vox2organ
CUDA_VISIBLE_DEVICES=0 python main.py \
    --test \
    -n V2C-Flow-S-ADNI \
    --dataset YOUR_DATASET_NAME \
    --experiment_base_dir pretrained_models/
```

## Troubleshooting

### Common Issues

**CUDA Version Compatibility**
- Server has CUDA 13.0 driver, but we use CUDA 12.1 wheels
- This is backwards compatible and works correctly

**PyTorch3D Build Issues**
- Ensure you're using the patched version with MeshesXD class
- Check that `python -c "import pytorch3d; print('OK')"` works

**Memory Issues**
- Use `CUDA_VISIBLE_DEVICES=0` to select specific GPU
- Monitor GPU memory with `nvidia-smi`

**Data Format Issues**
- Ensure surfaces are in MNI152 space
- Check that PLY files contain vertex and face data
- Verify NIfTI files have correct orientation

### Environment Variables
```bash
# For CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# For PyTorch memory debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## File Locations (MICA Server)

- **Code**: `~/Documents/Vox2Cortex/`
- **Conda Environment**:
  - Shared: `/data_/mica1/03_projects/enning/conda_envs/vox2cortex` (if accessible)
  - Personal: `~/conda_envs/vox2cortex` (create your own if needed)
- **Test Data**: `/data_/mica1/03_projects/enning/vox2cortex_test/`
- **Pretrained Models**: `~/Documents/Vox2Cortex/vox2organ/pretrained_models/V2C-Flow-S-ADNI/`
- **Raw BIDS Data**: `/data/mica3/BIDS_PNI/`

## Performance Notes

- **GPU**: RTX A6000 (48GB VRAM)
- **Inference Time**: ~20 seconds per subject
- **Memory Usage**: ~8GB GPU memory per subject
- **Batch Size**: Currently set to 1 (single subject processing)

## Support

For issues specific to the MICA server setup, contact the system administrators or check the original Vox2Cortex repository for updates.</content>
<parameter name="filePath">/Users/enningyang/CodeProj/Vox2Cortex/MICA_SETUP_GUIDE.md