# Vox2Cortex Environment Setup for bb-compxg-01

This guide helps set up the Vox2Cortex environment on bb-compxg-01 (BIC McGill server).

## Server Specifications
- **GPUs**: 8x NVIDIA RTX A6000
- **Driver**: 580.95.05
- **CUDA**: 13.0 (driver), 12.6 (toolkit at `/usr/local/cuda-12`)
- **No sudo access** - everything installed in user space

## Quick Setup (3 Steps)

### Step 1: Create Conda Environment
```bash
cd ~/Vox2Cortex
conda env create -f requirements_updated.yml
conda activate vox2cortex
```

### Step 2: Install PyTorch Geometric
```bash
conda activate vox2cortex
chmod +x setup_pyg.sh
./setup_pyg.sh
```

### Step 3: Install PyTorch3D (from fork)
```bash
conda activate vox2cortex
chmod +x setup_pytorch3d.sh
./setup_pytorch3d.sh
```
⚠️ This step takes 10-20 minutes to compile from source.

### Verify Installation
```bash
conda activate vox2cortex
chmod +x verify_env.sh
./verify_env.sh
```

## Alternative: All-in-One Script
```bash
chmod +x setup_vox2cortex_env.sh
./setup_vox2cortex_env.sh
```
This runs all steps automatically but takes longer.

## Usage

After setup, you can run Vox2Cortex:
```bash
conda activate vox2cortex
cd vox2organ
python main.py --help
```

## Troubleshooting

### CUDA not found during pytorch3d build
Make sure CUDA environment is set:
```bash
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### MeshesXD not found
This means pytorch3d was installed from the official repo instead of the fork.
Remove and reinstall:
```bash
pip uninstall pytorch3d
./setup_pytorch3d.sh
```

### PyTorch Geometric import errors
Make sure you installed the correct CUDA version wheels:
```bash
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
./setup_pyg.sh
```
