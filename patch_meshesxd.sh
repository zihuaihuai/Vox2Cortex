#!/bin/bash
# Copy MeshesXD from fork to installed pytorch3d

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data_/mica1/03_projects/enning/conda_envs/vox2cortex

SITE_PACKAGES=$(python -c "import pytorch3d; print(pytorch3d.__path__[0])")
echo "pytorch3d location: $SITE_PACKAGES"

FORK_DIR="/export03/data/enning/pytorch3d_build/pytorch3d_fork/pytorch3d"

# Copy the modified files
cp "$FORK_DIR/structures/meshes.py" "$SITE_PACKAGES/structures/meshes.py"
cp "$FORK_DIR/structures/__init__.py" "$SITE_PACKAGES/structures/__init__.py"

# Also copy utils.py if it has changes
cp "$FORK_DIR/structures/utils.py" "$SITE_PACKAGES/structures/utils.py"

echo "Files copied!"

# Test import
python -c "from pytorch3d.structures import MeshesXD; print('MeshesXD imported successfully!')"
