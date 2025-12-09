#!/bin/bash
# Verify Vox2Cortex environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data_/mica1/03_projects/enning/conda_envs/vox2cortex

python << 'PYEOF'
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
from pytorch3d.structures import MeshesXD
print("pytorch3d with MeshesXD: OK")

import monai
print(f"MONAI: {monai.__version__}")

import trimesh
print(f"trimesh: {trimesh.__version__}")

print("\n✓ All imports successful!")
PYEOF

echo ""
echo "Testing Vox2Cortex main.py --help..."
cd ~/Documents/Vox2Cortex/vox2organ
python main.py --help | head -30
