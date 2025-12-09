#!/usr/bin/env python3
"""
Transform FreeSurfer surfaces to MNI space for Vox2Cortex inference.
This script applies the ANTs affine transform to surfaces and saves as PLY.
"""

import os
import sys
import numpy as np
import nibabel as nib
import trimesh

def read_ants_affine(mat_file):
    """Read ANTs affine transformation matrix (.mat file is actually a text file)."""
    # ANTs .mat files from antsRegistrationSyNQuick are ITK transform text files
    # Try reading as text first
    try:
        with open(mat_file, 'r') as f:
            content = f.read()
    except UnicodeDecodeError:
        # If binary, we need a different approach - use SimpleITK or scipy
        import scipy.io as sio
        # ANTs sometimes saves in MATLAB format
        try:
            mat_data = sio.loadmat(mat_file)
            # Extract transform
            if 'AffineTransform_double_3_3' in mat_data:
                params = mat_data['AffineTransform_double_3_3'].flatten()
                fixed_params = mat_data.get('fixed', np.zeros((3, 1))).flatten()
            else:
                raise ValueError(f"Unknown ANTs mat format: {list(mat_data.keys())}")
        except:
            # Try reading with h5py for HDF5 format
            import h5py
            with h5py.File(mat_file, 'r') as f:
                # Navigate HDF5 structure
                transform_group = f['TransformGroup']['0']
                params = np.array(transform_group['TransformParameters']).flatten()
                fixed_params = np.array(transform_group['TransformFixedParameters']).flatten()
        
        # Build matrix from params
        R = params[:9].reshape(3, 3)
        t = params[9:12]
        center = fixed_params[:3] if len(fixed_params) >= 3 else np.zeros(3)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center - R @ center + t
        
        # LPS to RAS conversion
        lps_to_ras = np.diag([-1, -1, 1, 1])
        T = lps_to_ras @ T @ lps_to_ras
        return T
    
    # Parse ITK transform text file
    lines = content.strip().split('\n')
    params = None
    fixed_params = None
    
    for line in lines:
        if line.startswith('Parameters:'):
            params = np.array([float(x) for x in line.split(':')[1].strip().split()])
        elif line.startswith('FixedParameters:'):
            fixed_params = np.array([float(x) for x in line.split(':')[1].strip().split()])
    
    if params is None:
        raise ValueError(f"Could not parse ANTs transform file: {mat_file}")
    
    # ITK/ANTs uses a 12-parameter affine (rotation matrix + translation)
    R = params[:9].reshape(3, 3)
    t = params[9:12]
    center = fixed_params[:3] if fixed_params is not None else np.zeros(3)
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center + t
    
    # ANTs/ITK uses LPS coordinate system, FreeSurfer uses RAS
    lps_to_ras = np.diag([-1, -1, 1, 1])
    T = lps_to_ras @ T @ lps_to_ras
    
    return T

def transform_surface(input_path, output_path, transform_matrix):
    """Transform a FreeSurfer surface and save as PLY."""
    # Read FreeSurfer surface
    verts, faces = nib.freesurfer.io.read_geometry(input_path)
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    # Apply transformation
    mesh.apply_transform(transform_matrix)
    
    # Export as PLY
    mesh.export(output_path)
    print(f"Saved: {output_path} ({len(verts)} vertices, {len(faces)} faces)")
    
    return mesh

def simplify_mesh(input_path, output_path, reduction=0.3):
    """Simplify mesh using quadric decimation."""
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)
        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=reduction,
            preserveboundary=True,
            preservetopology=True
        )
        ms.save_current_mesh(output_path)
        m = ms.current_mesh()
        print(f"Simplified: {output_path} ({m.vertex_number()} vertices)")
    except ImportError:
        print("Warning: pymeshlab not installed, skipping simplification")

def main():
    if len(sys.argv) < 2:
        print("Usage: transform_surfaces.py <subject_dir>")
        print("Example: transform_surfaces.py /data_/mica1/03_projects/enning/vox2cortex_test/sub-PNA001_ses-01")
        sys.exit(1)
    
    subject_dir = sys.argv[1]
    
    # Read ANTs affine transform
    mat_file = os.path.join(subject_dir, "mri2mni_0GenericAffine.mat")
    if not os.path.exists(mat_file):
        print(f"Error: Transform file not found: {mat_file}")
        sys.exit(1)
    
    T = read_ants_affine(mat_file)
    print(f"Loaded transform from {mat_file}")
    print(f"Transform matrix:\n{T}")
    
    # Transform surfaces
    surfaces = [
        ("lh_pial.scanner", "lh_pial.ply"),
        ("lh_white.scanner", "lh_white.ply"),
        ("rh_pial.scanner", "rh_pial.ply"),
        ("rh_white.scanner", "rh_white.ply"),
    ]
    
    for input_name, output_name in surfaces:
        input_path = os.path.join(subject_dir, input_name)
        output_path = os.path.join(subject_dir, output_name)
        
        if os.path.exists(input_path):
            transform_surface(input_path, output_path, T)
            
            # Also create reduced version
            reduced_path = output_path.replace('.ply', '_reduced_0.3.ply')
            simplify_mesh(output_path, reduced_path, reduction=0.3)
        else:
            print(f"Warning: Surface not found: {input_path}")

if __name__ == "__main__":
    main()
