#!/bin/bash
# =============================================================================
# Vox2Cortex Environment Setup Script for bb-compxg-01
# =============================================================================
# Server: bb-compxg-01 (BIC McGill)
# GPUs: 8x NVIDIA RTX A6000
# Driver: 580.95.05, CUDA: 13.0 (driver), CUDA Toolkit: 12.6
#
# This script sets up a conda environment for Vox2Cortex without sudo rights.
# It handles the complexities of CUDA compatibility and the custom pytorch3d fork.
#
# Usage:
#   chmod +x setup_vox2cortex_env.sh
#   ./setup_vox2cortex_env.sh
#
# Requirements:
#   - conda/mamba installed and available in PATH
#   - CUDA-capable GPU with compatible driver
#   - Internet access to download packages
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 0: Check system environment
# =============================================================================
log_info "Checking system environment..."

# Set CUDA_HOME for bb-compxg-01
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check CUDA driver version
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    log_info "NVIDIA Driver Version: $DRIVER_VERSION"
    
    # Parse driver version to determine maximum supported CUDA version
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d'.' -f1)
    log_info "Driver Major Version: $DRIVER_MAJOR"
    
    # bb-compxg-01 has Driver 580+ with CUDA 13.0 support
    # We'll use PyTorch with CUDA 12.1 for best compatibility
    if [ "$DRIVER_MAJOR" -ge 525 ]; then
        log_info "Driver supports CUDA 12.x - using CUDA 12.1 for PyTorch"
        CUDA_TARGET="12.1"
    else
        log_warn "Using CUDA 11.8 for compatibility"
        CUDA_TARGET="11.8"
    fi
else
    log_error "nvidia-smi not found. Cannot detect GPU configuration."
    exit 1
fi

# Check for conda or mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    log_info "Using mamba for faster dependency resolution"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    log_info "Using conda (consider installing mamba for faster resolution)"
else
    log_error "Neither conda nor mamba found. Please install conda first."
    exit 1
fi

# =============================================================================
# Step 1: Create conda environment with core packages
# =============================================================================
ENV_NAME="vox2cortex"
PYTHON_VERSION="3.9"

log_info "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"

# Remove existing environment if it exists
$CONDA_CMD env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment with essential packages
# We use a minimal approach first, then add packages incrementally
$CONDA_CMD create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

log_info "Environment activated: $ENV_NAME"

# =============================================================================
# Step 2: Install PyTorch with correct CUDA version
# =============================================================================
log_info "Installing PyTorch with CUDA $CUDA_TARGET support..."

if [ "$CUDA_TARGET" == "12.1" ]; then
    # PyTorch 2.1.x with CUDA 12.1 (best for CUDA 12.6 toolkit on server)
    $CONDA_CMD install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
elif [ "$CUDA_TARGET" == "11.8" ]; then
    # PyTorch 2.0.x with CUDA 11.8 (good compatibility)
    $CONDA_CMD install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
elif [ "$CUDA_TARGET" == "11.3" ]; then
    # Original version from requirements.yml
    $CONDA_CMD install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch -y
fi

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# =============================================================================
# Step 3: Install PyTorch Geometric and related packages
# =============================================================================
log_info "Installing PyTorch Geometric..."

# PyG needs to match PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_MAJOR_MINOR=$(echo $TORCH_VERSION | cut -d'.' -f1,2)

if [[ "$TORCH_MAJOR_MINOR" == "2.1" ]]; then
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    pip install torch-geometric
elif [[ "$TORCH_MAJOR_MINOR" == "2.0" ]]; then
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    pip install torch-geometric
elif [[ "$TORCH_MAJOR_MINOR" == "1.10" ]]; then
    $CONDA_CMD install pyg=2.0.4 -c pyg -y
fi

# =============================================================================
# Step 4: Install core scientific packages
# =============================================================================
log_info "Installing scientific packages..."

$CONDA_CMD install -y \
    numpy=1.24.3 \
    scipy=1.10.1 \
    pandas \
    scikit-learn \
    scikit-image \
    matplotlib \
    nibabel \
    h5py \
    pillow \
    networkx \
    tqdm \
    pyyaml \
    -c conda-forge

# =============================================================================
# Step 5: Install additional required packages via pip
# =============================================================================
log_info "Installing additional packages via pip..."

pip install \
    monai>=0.8.1 \
    torchio \
    trimesh \
    pymeshlab \
    elasticdeform \
    geomloss \
    wandb \
    ipdb \
    tabulate \
    termcolor \
    yacs \
    fvcore \
    iopath \
    portalocker

# Install torchdiffeqpack (for neural ODEs)
pip install torchdiffeqpack

# =============================================================================
# Step 6: Install pytorch3d from the fork
# =============================================================================
log_info "Installing pytorch3d from fabibo3 fork..."

# Clone the fork
PYTORCH3D_DIR="$HOME/.cache/pytorch3d_fork"
rm -rf $PYTORCH3D_DIR
git clone https://github.com/fabibo3/pytorch3d.git $PYTORCH3D_DIR
cd $PYTORCH3D_DIR

# Install with CUDA support
# Set environment variables for building with CUDA
export FORCE_CUDA=1

# For bb-compxg-01, use the system CUDA
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

log_info "CUDA_HOME set to: $CUDA_HOME"
log_info "nvcc version: $(nvcc --version | grep release)"

# Build and install pytorch3d
# This may take 10-20 minutes
log_info "Building pytorch3d from source (this may take 10-20 minutes)..."
pip install -e . --no-build-isolation

cd -

# =============================================================================
# Step 7: Verify installation
# =============================================================================
log_info "Verifying installation..."

python << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import torch_geometric
print(f"PyTorch Geometric: {torch_geometric.__version__}")

import pytorch3d
print(f"PyTorch3D installed successfully")

# Check for MeshesXD (custom class from fork)
from pytorch3d.structures import MeshesXD
print("MeshesXD class available (fork working correctly)")

import monai
print(f"MONAI: {monai.__version__}")

import trimesh
print(f"trimesh: {trimesh.__version__}")

print("\n✓ All core packages installed successfully!")
EOF

log_info "Environment setup complete!"
log_info "To activate: conda activate $ENV_NAME"
log_info "To test Vox2Cortex: cd vox2organ && python main.py --help"
