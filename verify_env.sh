#!/bin/bash
# =============================================================================
# Verify Vox2Cortex Environment
# =============================================================================
# Run this to verify everything is installed correctly
# Usage: conda activate vox2cortex && ./verify_env.sh
# =============================================================================

set -e

echo "=============================================="
echo "Vox2Cortex Environment Verification"
echo "=============================================="

python << 'EOF'
import sys
print(f"Python: {sys.version}")

print("\n--- PyTorch ---")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

print("\n--- PyTorch Geometric ---")
import torch_geometric
print(f"PyTorch Geometric: {torch_geometric.__version__}")

print("\n--- PyTorch3D ---")
import pytorch3d
print(f"PyTorch3D installed")
from pytorch3d.structures import MeshesXD
print("MeshesXD class available ✓")

print("\n--- Medical Imaging ---")
import monai
print(f"MONAI: {monai.__version__}")
import nibabel
print(f"nibabel: {nibabel.__version__}")

print("\n--- Mesh Processing ---")
import trimesh
print(f"trimesh: {trimesh.__version__}")

print("\n--- Other Dependencies ---")
import torchdiffeqpack
print("torchdiffeqpack ✓")
import geomloss
print("geomloss ✓")
import wandb
print(f"wandb: {wandb.__version__}")

print("\n==============================================")
print("✓ All packages installed successfully!")
print("==============================================")
EOF

echo ""
echo "Testing Vox2Cortex main.py --help..."
cd "$(dirname "$0")/vox2organ"
python main.py --help | head -30

echo ""
echo "=============================================="
echo "Environment verification complete!"
echo "=============================================="
