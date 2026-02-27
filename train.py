#!/usr/bin/env python3
"""voxels2vertices-compatible training entrypoint for Vox2Cortex.

Usage mirrors brainscores/voxels2vertices:
  python train.py

Configuration is loaded from config.yml (or V2V_CONFIG), with optional
V2V_* environment overrides in the same style as voxels2vertices.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SPLIT_FILES = [
    "v2v_vox2cortex_train_ids.txt",
    "v2v_vox2cortex_val_ids.txt",
    "v2v_vox2cortex_test_ids.txt",
]
DEFAULT_VOLUME_GLOB = "*_hemi-L*.nii.gz"
DEFAULT_STEM_SUFFIX_LEFT = "_hemi-L"
DEFAULT_STEM_SUFFIX_R2L = "_hemi-LfromR"

DEFAULT_HEMI_PATTERNS = {
    "left": {
        "white": "{prefix}_L_fs_white.surf.gii",
        "pial": "{prefix}_L_fs_pial.surf.gii",
    },
    "left_from_right": {
        "white": "{prefix}_LfromR_fs_white.surf.gii",
        "pial": "{prefix}_LfromR_fs_pial.surf.gii",
    },
}


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Apply V2V_SECTION__KEY=value env overrides."""
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


def _stem_no_nii(path: Path) -> str:
    s = str(path)
    return s[:-7] if s.endswith(".nii.gz") else str(path.with_suffix(""))


def discover_v2v_ids(training_root: Path, cfg: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    skipped_missing = 0
    dcfg = cfg.get("data", {})
    volume_glob = str(dcfg.get("volume_glob", DEFAULT_VOLUME_GLOB))
    suffix_left = str(dcfg.get("stem_suffix_left", DEFAULT_STEM_SUFFIX_LEFT))
    suffix_r2l = str(dcfg.get("stem_suffix_right_to_left", DEFAULT_STEM_SUFFIX_R2L))
    hemi_patterns = dcfg.get("hemi_file_patterns", DEFAULT_HEMI_PATTERNS)
    left_patterns = hemi_patterns.get("left", DEFAULT_HEMI_PATTERNS["left"])
    r2l_patterns = hemi_patterns.get("left_from_right", DEFAULT_HEMI_PATTERNS["left_from_right"])

    for nii in sorted(training_root.rglob(volume_glob)):
        name = nii.name
        if any(
            token in name
            for token in (
                "_fslabels_MNI.tmp",
                "_huL_MNI.tmp",
                "_huR_MNI.tmp",
            )
        ):
            continue

        stem_abs = Path(_stem_no_nii(nii))
        sample_stem = str(stem_abs)
        if sample_stem.endswith(suffix_left):
            prefix = sample_stem[: -len(suffix_left)]
            required = [left_patterns["white"], left_patterns["pial"]]
        elif sample_stem.endswith(suffix_r2l):
            prefix = sample_stem[: -len(suffix_r2l)]
            required = [r2l_patterns["white"], r2l_patterns["pial"]]
        else:
            skipped_missing += 1
            continue

        required_paths = [Path(pattern.format(prefix=prefix)) for pattern in required]
        if not all(p.exists() for p in required_paths):
            skipped_missing += 1
            continue

        ids.append(str(stem_abs.relative_to(training_root)))

    print(
        f"Discovered {len(ids)} training samples in {training_root} "
        f"({skipped_missing} volumes skipped without full cortex surfaces)."
    )
    return ids


def write_split_files(
    ids: list[str],
    split_dir: Path,
    split_ratio: tuple[int, int, int],
    seed: int,
    split_files: list[str],
) -> tuple[int, int, int]:
    if len(split_files) != 3:
        raise ValueError("split_files must contain 3 file names.")
    if sum(split_ratio) != 100:
        raise ValueError("split_ratio must sum to 100.")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    counts = [split_ratio[0] * n // 100, split_ratio[1] * n // 100, split_ratio[2] * n // 100]
    counts[0] += n - sum(counts)  # assign remainder to train

    # Keep all splits non-empty when possible so Vox2Cortex validation works.
    if n >= 3:
        for idx in (1, 2):  # val, test
            if counts[idx] == 0:
                donor = 0 if counts[0] > 1 else (2 if idx == 1 and counts[2] > 1 else 1)
                if counts[donor] <= 1:
                    raise RuntimeError(
                        f"Cannot create non-empty train/val/test splits from {n} samples."
                    )
                counts[donor] -= 1
                counts[idx] = 1

    n_train, n_val, n_test = counts

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:n_train + n_val + n_test]

    split_dir.mkdir(parents=True, exist_ok=True)
    for fn, subset in zip(split_files, (train_ids, val_ids, test_ids)):
        with open(split_dir / fn, "w", encoding="utf-8") as f:
            for sid in subset:
                f.write(sid + "\n")

    return len(train_ids), len(val_ids), len(test_ids)


def _as_device_list(device_cfg: Any) -> list[str]:
    if isinstance(device_cfg, list):
        return [str(d) for d in device_cfg]
    if isinstance(device_cfg, str):
        parts = [p.strip() for p in device_cfg.split(",") if p.strip()]
        return parts if parts else ["cuda:0"]
    return ["cuda:0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vox2Cortex with voxels2vertices-style config.")
    parser.add_argument("--config", default=None, help="Path to config.yml (default: V2V_CONFIG or ./config.yml)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = Path(
        args.config
        or os.environ.get("V2V_CONFIG", str(repo_root / "config.yml"))
    ).resolve()
    cfg = _load_config(config_path)

    paths_cfg = cfg.get("paths", {})
    train_cfg = cfg.get("train", {})
    v2c_cfg = cfg.get("vox2cortex", {})

    training_root = Path(v2c_cfg.get("training_root", paths_cfg.get("training_root", "trainingdata"))).expanduser()
    if not training_root.is_absolute():
        training_root = (config_path.parent / training_root).resolve()
    if not training_root.exists():
        raise FileNotFoundError(f"training_root not found: {training_root}")

    # Split files may need a writable location different from training_root.
    split_dir_cfg = v2c_cfg.get("split_dir", None)
    if split_dir_cfg is None:
        split_dir = training_root
    else:
        split_dir = Path(str(split_dir_cfg)).expanduser()
        if not split_dir.is_absolute():
            split_dir = (config_path.parent / split_dir).resolve()

    split_seed = int(v2c_cfg.get("split_seed", train_cfg.get("seed", 1337)))
    split_ratio_cfg = v2c_cfg.get("split_ratio", [80, 10, 10])
    split_ratio = (int(split_ratio_cfg[0]), int(split_ratio_cfg[1]), int(split_ratio_cfg[2]))
    split_files = list(v2c_cfg.get("split_files", DEFAULT_SPLIT_FILES))

    ids = discover_v2v_ids(training_root, cfg)
    if len(ids) < 3:
        raise RuntimeError("Need at least 3 discovered samples to create train/val/test splits.")

    try:
        n_train, n_val, n_test = write_split_files(ids, split_dir, split_ratio, split_seed, split_files)
    except PermissionError:
        fallback_split_dir = (repo_root / "splits").resolve()
        print(
            f"Permission denied writing split files to {split_dir}. "
            f"Falling back to {fallback_split_dir}."
        )
        split_dir = fallback_split_dir
        n_train, n_val, n_test = write_split_files(ids, split_dir, split_ratio, split_seed, split_files)
    print(f"Wrote split files in {split_dir}: train={n_train}, val={n_val}, test={n_test}")

    group = str(v2c_cfg.get("group", "V2C-Flow-S"))
    device_list = _as_device_list(v2c_cfg.get("device", "cuda:0"))
    exp_name = str(v2c_cfg.get("experiment_name", "vox2cortex_v2v"))
    exp_base_dir = str(v2c_cfg.get("experiment_base_dir", "../experiments"))
    run_test_after_train = bool(v2c_cfg.get("run_test_after_train", True))
    max_vram_gb = float(v2c_cfg.get("max_vram_gb", 48))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 500))
    pretrained_model = v2c_cfg.get("pretrained_model", None)

    cmd = [
        sys.executable,
        "vox2organ/main.py",
        "--train",
        "--group",
        group,
        "--dataset",
        "VOXELS2VERTICES",
        "--n_epochs",
        str(epochs),
        "--experiment_base_dir",
        exp_base_dir,
        "-n",
        exp_name,
        "--no-wandb",
        "--device",
        *device_list,
    ]
    if run_test_after_train:
        cmd.append("--test")
    if pretrained_model:
        cmd.extend(["--pretrained_model", str(pretrained_model)])

    env = os.environ.copy()
    env["VOX2CORTEX_V2V_DATA_ROOT"] = str(training_root)
    env["VOX2CORTEX_V2V_SPLIT_DIR"] = str(split_dir)
    env["VOX2CORTEX_MAX_VRAM_GB"] = str(max_vram_gb)

    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
