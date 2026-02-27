#!/usr/bin/env python3
"""voxels2vertices-compatible inference entrypoint for Vox2Cortex.

Usage mirrors brainscores/voxels2vertices:
  python inference.py --input <nii_or_dir> --output-dir <dir> --ckpt <checkpoint.pt>
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    prefix = "V2V_"
    for key, val in os.environ.items():
        if not key.startswith(prefix) or key == "V2V_CONFIG":
            continue
        parts = key[len(prefix):].lower().split("__")
        target = cfg
        for p in parts[:-1]:
            if p not in target or not isinstance(target[p], dict):
                target[p] = {}
            target = target[p]
        leaf = parts[-1]

        parsed: Any = val
        if val.lower() in {"true", "false"}:
            parsed = val.lower() == "true"
        else:
            try:
                if "." in val:
                    parsed = float(val)
                    if parsed.is_integer():
                        parsed = int(parsed)
                else:
                    parsed = int(val)
            except Exception:
                pass
        target[leaf] = parsed


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _apply_env_overrides(cfg)
    return cfg


def _normalize_min_max(x: torch.Tensor) -> torch.Tensor:
    mn = float(x.min())
    mx = float(x.max())
    if mx <= mn:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


def _box_in_bounds(box, image_shape):
    newbox = []
    pad_width = []
    for box_i, shape_i in zip(box, image_shape):
        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))
        newbox.append(newbox_i)
        pad_width.append(pad_width_i)
    needs_padding = any(i != (0, 0) for i in pad_width)
    return newbox, pad_width, needs_padding


def _crop_indices(image_shape, patch_shape, center):
    box = [(i - ps // 2, i - ps // 2 + ps) for i, ps in zip(center, patch_shape)]
    box, pad_width, needs_padding = _box_in_bounds(box, image_shape)
    slices = tuple(slice(i[0], i[1]) for i in box)
    return slices, pad_width, needs_padding


def _crop(image, patch_shape, center, pad_value=0):
    slices, pad_width, needs_padding = _crop_indices(image.shape, patch_shape, center)
    patch = image[slices]
    if needs_padding:
        if len(pad_width) < patch.ndim:
            pad_width.append((0, 0))
        patch = np.pad(patch, pad_width, mode="constant", constant_values=pad_value)
    return patch


def img_with_patch_size(
    img: np.ndarray | torch.Tensor,
    patch_size,
    is_label: bool,
    mode="crop",
    crop_at=None,
    pad_value=0,
):
    d, h, w = img.shape
    d_new, h_new, w_new = patch_size
    transform_affine = np.eye(4)

    if mode == "crop":
        if crop_at is None:
            center_z, center_y, center_x = d // 2, h // 2, w // 2
        else:
            center_z, center_y, center_x = crop_at
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = _crop(img, (d_new, h_new, w_new), (center_z, center_y, center_x), pad_value=pad_value)
        img = torch.from_numpy(img).long() if is_label else torch.from_numpy(img).float()
        transform_affine[:-1, -1] = -np.array(
            [center_z - d_new // 2, center_y - h_new // 2, center_x - w_new // 2]
        )

    elif mode == "interpolate":
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        transform_affine[np.diag_indices(3)] = np.array(
            patch_size, dtype=float
        ) / np.array(img.shape, dtype=float)
        if is_label:
            img = torch.nn.functional.interpolate(img[None, None].float(), patch_size, mode="nearest")[0, 0].long()
        else:
            img = torch.nn.functional.interpolate(
                img[None, None], patch_size, mode="trilinear", align_corners=False
            )[0, 0]
    else:
        raise ValueError("Unknown mode.")

    assert tuple(img.shape) == tuple(patch_size)
    return img, transform_affine


def _set_vram_cap(device_name: str, max_vram_gb: float) -> None:
    if max_vram_gb <= 0 or not torch.cuda.is_available() or not device_name.startswith("cuda"):
        return

    dev = torch.device(device_name)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024.0 ** 3)
    fraction = min(max_vram_gb / total_gb, 1.0)
    if fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, device=idx)
        print(
            f"Applied VRAM cap {max_vram_gb:.2f} GB on {device_name} "
            f"({100.0 * fraction:.2f}% of {total_gb:.2f} GB)."
        )


def _list_niftis(path_str: str) -> list[str]:
    p = Path(path_str)
    if p.is_file() and str(p).endswith(".nii.gz"):
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "**" / "*.nii.gz"), recursive=True))
    raise FileNotFoundError(f"Input path not found: {path_str}")


def _nii_stem(path_str: str) -> str:
    name = Path(path_str).name
    return name[:-7] if name.endswith(".nii.gz") else Path(path_str).stem


def _save_surface_gifti(vertices_world: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    coords_da = nib.gifti.GiftiDataArray(
        data=vertices_world.astype(np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"],
    )
    faces_da = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"],
    )
    gi = nib.gifti.GiftiImage(darrays=[coords_da, faces_da])
    nib.save(gi, str(out_path))


def _check_bin(name: str) -> bool:
    return shutil.which(name) is not None


def _register_to_mni_if_possible(
    in_path: str,
    out_path: Path,
    ref_mni_path: Path | None,
    nthreads: int,
    skip_register: bool,
) -> str:
    if skip_register:
        return in_path

    if ref_mni_path is None or not ref_mni_path.exists():
        print("No MNI reference found, skipping ANTs registration.")
        return in_path

    if not (_check_bin("antsRegistrationSyNQuick.sh") and _check_bin("antsApplyTransforms")):
        print("ANTs binaries not found, skipping registration.")
        return in_path

    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "reg_"
        cmd = [
            "antsRegistrationSyNQuick.sh",
            "-d",
            "3",
            "-f",
            str(ref_mni_path),
            "-m",
            in_path,
            "-o",
            str(prefix),
            "-n",
            str(max(1, nthreads)),
            "-t",
            "s",
        ]
        subprocess.check_call(cmd)

        warped = Path(str(prefix) + "Warped.nii.gz")
        if not warped.exists():
            raise RuntimeError(f"ANTs registration did not produce warped image: {warped}")

        shutil.copyfile(warped, out_path)
    return str(out_path)


def _prepare_input_tensor(
    image_path: str,
    hps: dict[str, Any],
    normalize_vertices_fn,
) -> tuple[torch.Tensor, np.ndarray]:
    img = nib.load(image_path)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    world2vox = np.linalg.inv(img.affine)

    patch_size = tuple(hps["PATCH_SIZE"])
    select_patch_size = tuple(hps.get("SELECT_PATCH_SIZE", hps["PATCH_SIZE"]))
    patch_origin = np.array(hps.get("PATCH_ORIGIN", [0, 0, 0]), dtype=int)

    lower_limit = patch_origin
    upper_limit = patch_origin + np.array(select_patch_size, dtype=int)
    crop_at = tuple(((lower_limit + upper_limit) // 2).tolist())

    vol_patch, trans_affine_1 = img_with_patch_size(
        vol,
        select_patch_size,
        is_label=False,
        mode="crop",
        crop_at=crop_at,
        pad_value=0,
    )

    if patch_size != select_patch_size:
        vol_patch, trans_affine_2 = img_with_patch_size(
            vol_patch,
            patch_size,
            is_label=False,
            mode="interpolate",
        )
    else:
        trans_affine_2 = np.eye(4)

    trans_affine_img = trans_affine_2 @ trans_affine_1

    if isinstance(vol_patch, np.ndarray):
        vol_patch = torch.from_numpy(vol_patch).float()
    else:
        vol_patch = vol_patch.float()

    vol_patch = _normalize_min_max(vol_patch)

    _, norm_affine = normalize_vertices_fn(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        patch_size,
        return_affine=True,
    )
    trans_affine_label = norm_affine @ trans_affine_img @ world2vox

    x = vol_patch.unsqueeze(0).unsqueeze(0)
    return x, trans_affine_label


def _load_hps(
    group: str,
    ckpt_path: Path,
    params_path: Path | None,
    assemble_group_params_fn,
    update_dict_fn,
    param_file_name: str,
) -> dict[str, Any]:
    hps = assemble_group_params_fn(group)

    inferred_params = params_path
    if inferred_params is None:
        candidate = ckpt_path.parent / param_file_name
        if candidate.exists():
            inferred_params = candidate

    if inferred_params is not None and inferred_params.exists():
        with open(inferred_params, "r", encoding="utf-8") as f:
            params_json = yaml.safe_load(f) or {}
        hps = update_dict_fn(hps, params_json)

    # Stored params may serialize classes as strings.
    gc_val = hps["MODEL_CONFIG"].get("GC")
    if isinstance(gc_val, str):
        hps["MODEL_CONFIG"]["GC"] = eval(gc_val)

    return hps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register to MNI (optional), run Vox2Cortex inference, and save GIFTI meshes."
    )
    parser.add_argument("-i", "--input", required=True, help="Input .nii.gz file OR directory containing .nii.gz files")
    parser.add_argument("-o", "--output-dir", required=True, help="Where to save outputs")
    parser.add_argument("-ckpt", "--ckpt", required=True, help="Checkpoint (.pt) produced by Vox2Cortex training")
    parser.add_argument("--config", default=None, help="Path to config.yml (default: V2V_CONFIG or ./config.yml)")
    parser.add_argument("--params", default=None, help="Optional params.json to recreate model hyperparameters")
    parser.add_argument("--group", default="V2C-Flow-S", help="Fallback Vox2Cortex parameter group")
    parser.add_argument("-device", "--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use torch.autocast(float16) on CUDA")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-vram-gb", type=float, default=48.0)
    parser.add_argument("--skip-register", action="store_true", help="Skip ANTs registration and run on input space")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    vox2organ_dir = REPO_ROOT / "vox2organ"
    if str(vox2organ_dir) not in sys.path:
        sys.path.insert(0, str(vox2organ_dir))

    import logger  # type: ignore
    from models.model_handler import ModelHandler  # type: ignore
    from params.groups import assemble_group_params  # type: ignore
    from utils.coordinate_transform import normalize_vertices  # type: ignore
    from utils.graph_conv import SparseGraphConv, GraphConvNorm, LinearLayer  # noqa: F401 # type: ignore
    from utils.template import MeshTemplate, TEMPLATE_SPECS  # type: ignore
    from utils.utils import load_checkpoint, update_dict  # type: ignore

    log = logger.get_std_logger(__name__)

    config_path = Path(
        args.config
        or os.environ.get("V2V_CONFIG", str(REPO_ROOT / "config.yml"))
    ).resolve()
    cfg = _load_config(config_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        _set_vram_cap(device, args.max_vram_gb)

    hps = _load_hps(
        args.group,
        Path(args.ckpt),
        Path(args.params).resolve() if args.params else None,
        assemble_group_params,
        update_dict,
        logger.PARAM_FILE_NAME,
    )

    model_config = {k.lower(): v for k, v in hps["MODEL_CONFIG"].items()}
    model = ModelHandler[hps["ARCHITECTURE"]].value(
        ndims=hps["NDIMS"],
        n_v_classes=hps["N_V_CLASSES"],
        n_m_classes=hps["N_M_CLASSES"],
        patch_size=hps["PATCH_SIZE"],
        **model_config,
    ).float()
    model, _, _, _ = load_checkpoint(model, args.ckpt, "cpu")
    model = model.to(device)
    model.eval()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    refsurfs_dir = Path(cfg.get("paths", {}).get("refsurfs_dir", "refsurfs"))
    if not refsurfs_dir.is_absolute():
        refsurfs_dir = (config_path.parent / refsurfs_dir).resolve()
    ref_mni_path = refsurfs_dir / "MNI152_T1_0.6mm_brain.nii.gz"

    inputs = _list_niftis(args.input)
    if not inputs:
        raise RuntimeError("No input NIfTI files found.")

    mesh_names = ["lh_white", "rh_white", "lh_pial", "rh_pial"]
    out_name_tpl = {
        "lh_white": "hemi-L_surf-fsLR-32k_label-white.surf.gii",
        "rh_white": "hemi-R_surf-fsLR-32k_label-white.surf.gii",
        "lh_pial": "hemi-L_surf-fsLR-32k_label-pial.surf.gii",
        "rh_pial": "hemi-R_surf-fsLR-32k_label-pial.surf.gii",
    }

    for in_path in inputs:
        stem = _nii_stem(in_path)
        log.info("[inference] %s", in_path)

        warped_path = output_dir / f"{stem}_space-MNI152.nii.gz"
        model_input_path = _register_to_mni_if_possible(
            in_path=in_path,
            out_path=warped_path,
            ref_mni_path=ref_mni_path,
            nthreads=args.num_threads,
            skip_register=args.skip_register,
        )

        x, trans_affine_label = _prepare_input_tensor(model_input_path, hps, normalize_vertices)

        template = MeshTemplate(
            mesh_label_names=mesh_names,
            trans_affine=trans_affine_label,
            **TEMPLATE_SPECS[hps["MESH_TEMPLATE_ID"]],
        )
        input_meshes = template.create_template_batch_size(1, device=device)

        with torch.inference_mode():
            if args.amp and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(x.to(device), input_meshes)
            else:
                pred = model(x.to(device), input_meshes)

        model_class = model.__class__
        mesh_pred = model_class.pred_to_final_mesh_pred(pred)

        vox2ras = np.linalg.inv(trans_affine_label)
        mesh_pred = mesh_pred.transform(torch.tensor(vox2ras, dtype=torch.float32, device=device))

        for name, verts, faces in zip(mesh_names, mesh_pred.verts_list(), mesh_pred.faces_list()):
            out_name = f"{stem}_{out_name_tpl[name]}"
            out_path = output_dir / out_name
            _save_surface_gifti(
                verts.detach().cpu().numpy(),
                faces.detach().cpu().numpy(),
                out_path,
            )
            log.info("wrote: %s", out_path)

        # Free graph memory between inputs.
        del pred, mesh_pred, input_meshes
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
