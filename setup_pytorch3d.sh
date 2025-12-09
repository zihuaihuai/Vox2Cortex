#!/bin/bash
# =============================================================================
# Step 3: Install PyTorch3D from the fabibo3 fork
# =============================================================================
# This fork contains the MeshesXD class required by Vox2Cortex
# Run this AFTER installing PyTorch Geometric
# Usage: conda activate vox2cortex && ./setup_pytorch3d.sh
# =============================================================================

set -e

echo "[INFO] Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1

echo "[INFO] CUDA_HOME: $CUDA_HOME"
nvcc --version

echo "[INFO] Cloning pytorch3d fork from fabibo3..."
PYTORCH3D_DIR="$HOME/.cache/pytorch3d_fork"
rm -rf $PYTORCH3D_DIR
git clone https://github.com/fabibo3/pytorch3d.git $PYTORCH3D_DIR

cd $PYTORCH3D_DIR

echo "[INFO] Building pytorch3d from source (this may take 10-20 minutes)..."
pip install -e . --no-build-isolation -v

echo "[INFO] Verifying pytorch3d installation..."
python -c "
import pytorch3d
print('PyTorch3D installed successfully')
from pytorch3d.structures import MeshesXD
print('MeshesXD class available (fork working correctly!)')
"

echo "[INFO] pytorch3d setup complete!"
