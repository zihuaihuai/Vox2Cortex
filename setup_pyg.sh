#!/bin/bash
# =============================================================================
# Step 2: Install PyTorch Geometric
# =============================================================================
# Run this AFTER creating the conda environment
# Usage: conda activate vox2cortex && ./setup_pyg.sh
# =============================================================================

set -e

echo "[INFO] Installing PyTorch Geometric for PyTorch 2.1 + CUDA 12.1..."

# Verify we're in the right environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install PyG packages with correct CUDA version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install torch-geometric

echo "[INFO] PyTorch Geometric installed successfully!"
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
