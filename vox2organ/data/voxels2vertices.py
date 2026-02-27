"""voxels2vertices (svf branch) dataset support for Vox2Cortex."""

__author__ = "Codex"

import os
import glob
from enum import IntEnum
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes

import logger
from data.image_and_mesh_dataset import ImageAndMeshDataset
from utils.modes import DataModes
from utils.coordinate_transform import transform_mesh_affine


log = logger.get_std_logger(__name__)


class Voxels2VerticesLabels(IntEnum):
    """Label mapping used by brainscores/voxels2vertices import_data.py."""

    white_matter = 1
    gray_matter = 2


class Voxels2VerticesDataset(ImageAndMeshDataset):
    """Dataset for voxels2vertices svf-style hemi exports.

    Expected files per sample id/stem (id includes hemisphere suffix):
      <base>_hemi-L.nii.gz
      <base>_hemi-LfromR.nii.gz

    Surfaces:
      <base>_L_fs_white.surf.gii
      <base>_L_fs_pial.surf.gii
      <base>_LfromR_fs_white.surf.gii
      <base>_LfromR_fs_pial.surf.gii

    We map both hemi-L and hemi-LfromR samples to the same LEFT-key targets
    used by Vox2Cortex cortex-all (lh/rh white+pial after pairing in training).
    """

    image_file_name = "mri.nii.gz"  # not used for flat layout, kept for API compatibility
    seg_file_name = "aseg.nii.gz"   # segmentation is stored in the same flat volume

    LabelMap = Voxels2VerticesLabels

    @classmethod
    def _get_seg_and_mesh_label_names(cls, structure_type):
        if structure_type != "cortex-all":
            raise ValueError(
                "Voxels2VerticesDataset currently supports only structure_type='cortex-all'."
            )

        seg_label_names = {
            "white_matter": ("white_matter",),
            "gray_matter": ("gray_matter",),
        }
        mesh_label_names = {
            "lh_white": "L_fs_white",
            "rh_white": "R_fs_white",
            "lh_pial": "L_fs_pial",
            "rh_pial": "R_fs_pial",
        }
        return seg_label_names, mesh_label_names

    def __init__(self, structure_type: str, **kwargs):
        self.reduced_gt = kwargs.get("reduced_gt", False)
        self.registered_gt_meshes = kwargs.get("registered_gt_meshes", False)

        # voxels2vertices flat exports do not use these variants
        if self.reduced_gt or self.registered_gt_meshes:
            log.warning(
                "Ignoring reduced_gt=%r / registered_gt_meshes=%r for Voxels2VerticesDataset.",
                self.reduced_gt,
                self.registered_gt_meshes,
            )

        (self.voxel_label_names, self.mesh_label_names) = self._get_seg_and_mesh_label_names(
            structure_type
        )

        super().__init__(
            image_file_name=self.image_file_name,
            mesh_file_names=list(self.mesh_label_names.values()),
            seg_file_name=self.seg_file_name,
            **kwargs,
        )

    @staticmethod
    def _stem_no_nii(path: str) -> str:
        return path[:-7] if path.endswith(".nii.gz") else os.path.splitext(path)[0]

    @classmethod
    def discover_ids(cls, raw_data_dir: str) -> list[str]:
        """Discover valid sample ids (stems relative to raw_data_dir)."""
        root = os.path.realpath(os.path.expanduser(raw_data_dir))
        nii_files = sorted(
            glob.glob(os.path.join(root, "**", "*_hemi-L*.nii.gz"), recursive=True)
        )

        ids = []
        skipped = 0

        for nii_path in nii_files:
            base_name = os.path.basename(nii_path)
            if any(
                token in base_name
                for token in (
                    "_hemi-L",
                    "_hemi-LfromR",
                    "_fslabels_MNI.tmp",
                    "_huL_MNI.tmp",
                    "_huR_MNI.tmp",
                )
            ):
                continue

            stem_abs = cls._stem_no_nii(nii_path)
            if stem_abs.endswith("_hemi-LfromR"):
                prefix = stem_abs.replace("_hemi-LfromR", "")
                required_mesh_suffixes = (
                    "_LfromR_fs_white.surf.gii",
                    "_LfromR_fs_pial.surf.gii",
                )
            elif stem_abs.endswith("_hemi-L"):
                prefix = stem_abs.replace("_hemi-L", "")
                required_mesh_suffixes = (
                    "_L_fs_white.surf.gii",
                    "_L_fs_pial.surf.gii",
                )
            else:
                skipped += 1
                continue

            if not all(os.path.exists(prefix + sfx) for sfx in required_mesh_suffixes):
                skipped += 1
                continue

            rel_stem = os.path.relpath(stem_abs, root)
            ids.append(rel_stem)

        log.info(
            "Discovered %d voxels2vertices samples in %s (%d skipped without full cortex surfaces).",
            len(ids),
            root,
            skipped,
        )
        return ids

    @classmethod
    def split(
        cls,
        raw_data_dir,
        save_dir,
        augment_train: bool = False,
        dataset_seed: int = 0,
        all_ids_file: str = None,
        dataset_split_proportions: Sequence[int] = None,
        fixed_split=None,
        overfit: int = None,
        load_only=("train", "validation", "test"),
        **kwargs,
    ):
        # Auto-discover ids if not provided through fixed/all_ids.
        if fixed_split is None and all_ids_file is None:
            all_ids = cls.discover_ids(raw_data_dir)
            if not all_ids:
                raise RuntimeError(
                    "No voxels2vertices samples found. "
                    "Expected flat '<stem>.nii.gz' with L/R fs white/pial .surf.gii files."
                )
            auto_ids_file = os.path.join(raw_data_dir, "v2v_all_ids_auto.txt")
            with open(auto_ids_file, "w", encoding="utf-8") as f:
                for sid in all_ids:
                    f.write(sid + "\n")
            all_ids_file = os.path.basename(auto_ids_file)

        return super().split(
            raw_data_dir=raw_data_dir,
            save_dir=save_dir,
            augment_train=augment_train,
            dataset_seed=dataset_seed,
            all_ids_file=all_ids_file,
            dataset_split_proportions=dataset_split_proportions,
            fixed_split=fixed_split,
            overfit=overfit,
            load_only=load_only,
            **kwargs,
        )

    def _volume_path(self, sample_id: str) -> str:
        """Path to the flat .nii.gz volume for one sample id/stem."""
        return os.path.join(self._raw_data_dir, sample_id + ".nii.gz")

    def _mesh_path(self, sample_id: str, mesh_name: str) -> str:
        """Path to one svf-style .surf.gii mesh for one sample id/stem."""
        sample_abs = os.path.join(self._raw_data_dir, sample_id)
        if sample_abs.endswith("_hemi-LfromR"):
            prefix = sample_abs.replace("_hemi-LfromR", "")
            hemi_token = "LfromR"
        elif sample_abs.endswith("_hemi-L"):
            prefix = sample_abs.replace("_hemi-L", "")
            hemi_token = "L"
        else:
            raise ValueError(f"Unexpected sample id suffix for {sample_id}")

        # mesh_name is one of: L_fs_white, R_fs_white, L_fs_pial, R_fs_pial
        if mesh_name.endswith("fs_white"):
            surf_part = "fs_white"
        elif mesh_name.endswith("fs_pial"):
            surf_part = "fs_pial"
        else:
            raise ValueError(f"Unsupported mesh_name: {mesh_name}")

        # For svf hemispheric data we load L or LfromR targets from file names.
        return prefix + f"_{hemi_token}_{surf_part}.surf.gii"

    def image_affine(self, index: int):
        return nib.load(self._volume_path(self.ids[index])).affine

    def _load_data3D_and_transform(self, filename: str, is_label: bool):
        """Load flat .nii.gz data and transform to the configured patch size."""
        del filename  # flat layout ignores per-file names

        data = []
        transformations = []
        for sample_id in self.ids:
            img = nib.load(self._volume_path(sample_id))
            img_data = img.get_fdata()
            if self._orig_img_size is None:
                self._orig_img_size = img_data.shape
            else:
                assert np.array_equal(
                    np.array(img_data.shape), np.array(self._orig_img_size)
                ), "All images should be of equal size"
            img_data, trans_affine = self._get_single_patch(img_data, is_label)
            data.append(img_data)
            transformations.append(trans_affine)

        return data, transformations

    @staticmethod
    def _read_gifti_mesh(path: str):
        g = nib.load(path)
        pointset_intent = nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
        triangle_intent = nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]

        verts = None
        faces = None
        for da in g.darrays:
            if da.intent == pointset_intent:
                verts = np.asarray(da.data, dtype=np.float32)
            elif da.intent == triangle_intent:
                faces = np.asarray(da.data, dtype=np.int64)

        if verts is None and g.darrays:
            verts = np.asarray(g.darrays[0].data, dtype=np.float32)
        if faces is None and len(g.darrays) > 1:
            faces = np.asarray(g.darrays[1].data, dtype=np.int64)

        if verts is None or faces is None:
            raise ValueError(f"Could not parse GIFTI mesh arrays from {path}")

        return verts, faces

    def _load_dataMesh_raw(self, meshnames):
        """Load flat .surf.gii meshes and transform them to image voxel coordinates."""
        data = []
        assert len(self.trans_affine) == 0, "Should be empty."

        for sample_id in self.ids:
            orig = nib.load(self._volume_path(sample_id))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            self.trans_affine.append(world2vox_affine)

            file_vertices = []
            file_faces = []
            for mesh_name in meshnames:
                mesh_path = self._mesh_path(sample_id, mesh_name)
                try:
                    if os.path.exists(mesh_path):
                        vertices, faces = self._read_gifti_mesh(mesh_path)
                    elif os.path.exists(mesh_path.replace(".surf.gii", ".ply")):
                        ply_mesh = trimesh.load_mesh(
                            mesh_path.replace(".surf.gii", ".ply"), process=False
                        )
                        vertices = np.asarray(ply_mesh.vertices, dtype=np.float32)
                        faces = np.asarray(ply_mesh.faces, dtype=np.int64)
                    elif os.path.exists(mesh_path.replace(".surf.gii", ".stl")):
                        stl_mesh = trimesh.load_mesh(mesh_path.replace(".surf.gii", ".stl"))
                        vertices = np.asarray(stl_mesh.vertices, dtype=np.float32)
                        faces = np.asarray(stl_mesh.faces, dtype=np.int64)
                    else:
                        raise FileNotFoundError(mesh_path)
                except Exception as e:
                    if self.mode != DataModes.TEST:
                        raise e
                    dummy = trimesh.creation.icosahedron()
                    vertices = np.asarray(dummy.vertices, dtype=np.float32)
                    faces = np.asarray(dummy.faces, dtype=np.int64)
                    log.warning(
                        "No mesh for file %s (%s), inserting dummy.",
                        sample_id,
                        mesh_name,
                    )

                vertices, faces = transform_mesh_affine(vertices, faces, world2vox_affine)

                self.n_max_vertices = (
                    np.maximum(vertices.shape[0], self.n_max_vertices)
                    if (self.n_max_vertices is not None)
                    else vertices.shape[0]
                )
                self.n_min_vertices = (
                    np.minimum(vertices.shape[0], self.n_min_vertices)
                    if (self.n_min_vertices is not None)
                    else vertices.shape[0]
                )

                file_vertices.append(torch.from_numpy(vertices).float())
                file_faces.append(torch.from_numpy(faces).long())

            data.append(Meshes(file_vertices, file_faces))

        return data
