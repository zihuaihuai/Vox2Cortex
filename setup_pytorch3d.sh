#!/bin/bash
# =============================================================================
# Step 3: Install PyTorch3D from the fabibo3 fork
# =============================================================================
# This fork contains the MeshesXD class required by Vox2Cortex
# Run this AFTER installing PyTorch Geometric
# 
# IMPORTANT: Run this in a screen/tmux session as it takes 10-20 minutes!
#   screen -S pytorch3d
#   ./setup_pytorch3d.sh
#   # Press Ctrl+A then D to detach
#   # Later: screen -r pytorch3d to reattach
#
# Usage: conda activate vox2cortex && ./setup_pytorch3d.sh
# =============================================================================

set -e

# Activate conda if not already active
if [ -z "$CONDA_PREFIX" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate /data_/mica1/03_projects/enning/conda_envs/vox2cortex
fi

echo "[INFO] Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
export MAX_JOBS=4  # Limit parallel jobs to avoid memory issues

echo "[INFO] CUDA_HOME: $CUDA_HOME"
nvcc --version

echo "[INFO] PyTorch info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

echo "[INFO] Cloning pytorch3d fork from fabibo3..."
PYTORCH3D_DIR="$HOME/pytorch3d_build/pytorch3d_fork"
if [ ! -d "$PYTORCH3D_DIR" ]; then
    mkdir -p "$HOME/pytorch3d_build"
    git clone https://github.com/fabibo3/pytorch3d.git "$PYTORCH3D_DIR"
fi

cd "$PYTORCH3D_DIR"

echo "[INFO] Building pytorch3d from source (this may take 10-20 minutes)..."
echo "[INFO] Started at: $(date)"
pip install -e . --no-build-isolation -v 2>&1 | tee ~/pytorch3d_build.log

echo "[INFO] Finished at: $(date)"

echo "[INFO] Verifying pytorch3d installation..."
python -c "
import pytorch3d
print('PyTorch3D installed successfully')
from pytorch3d.structures import MeshesXD
print('MeshesXD class available (fork working correctly!)')
"

echo "[INFO] pytorch3d setup complete!"
