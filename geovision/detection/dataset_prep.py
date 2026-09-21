"""Dataset preparation and train/val splitting for YOLO detection training."""

import shutil
from pathlib import Path

import numpy as np
import yaml

from geovision.constants import (
    NWPU_VHR10_CLASSES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from geovision.data.downloaders import download_nwpu_vhr10
from geovision.logger import get_logger

logger = get_logger("geovision.detection.dataset_prep")


def prepare_yolo_dataset(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    val_split: float = 0.2,
    smoke: bool = False,
    seed: int = 42,
) -> Path:
    """Prepare YOLO-formatted dataset with train/val splits and data.yaml configuration.

    Args:
        raw_dir: Path to raw NWPU VHR-10 data.
        output_dir: Destination path for YOLO-formatted dataset.
        val_split: Proportion of data to reserve for validation (0.0 to 1.0).
        smoke: If True, uses/generates lightweight smoke subset.
        seed: Random seed for reproducible splitting.

    Returns:
        Path to the generated `data.yaml` configuration file.
    """
    raw_path = Path(raw_dir) if raw_dir else RAW_DATA_DIR / "NWPU_VHR10"
    out_path = Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "yolo_nwpu"

    if not raw_path.exists() or len(list(raw_path.rglob("*.jpg"))) == 0:
        logger.info(f"Raw NWPU dataset not found at {raw_path}. Downloading/generating (Smoke: {smoke})...")
        download_nwpu_vhr10(RAW_DATA_DIR, smoke=smoke)

    images_src = raw_path / "images"
    labels_src = raw_path / "labels"

    # Setup YOLO directory structure
    train_img_dir = out_path / "images" / "train"
    val_img_dir = out_path / "images" / "val"
    train_lbl_dir = out_path / "labels" / "train"
    val_lbl_dir = out_path / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(images_src.glob("*.jpg")) + list(images_src.glob("*.png")))
    if smoke and len(all_images) > 10:
        all_images = all_images[:10]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_images))
    split_idx = max(1, int(len(all_images) * (1.0 - val_split)))

    train_files = [all_images[i] for i in indices[:split_idx]]
    val_files = [all_images[i] for i in indices[split_idx:]] if split_idx < len(all_images) else [all_images[0]]

    # Copy train files
    for img_file in train_files:
        shutil.copy2(img_file, train_img_dir / img_file.name)
        lbl_file = labels_src / f"{img_file.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, train_lbl_dir / lbl_file.name)
        else:
            (train_lbl_dir / f"{img_file.stem}.txt").touch()

    # Copy val files
    for img_file in val_files:
        shutil.copy2(img_file, val_img_dir / img_file.name)
        lbl_file = labels_src / f"{img_file.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, val_lbl_dir / lbl_file.name)
        else:
            (val_lbl_dir / f"{img_file.stem}.txt").touch()

    # Generate data.yaml with absolute posix paths
    data_yaml_path = out_path / "data.yaml"
    data_dict = {
        "path": out_path.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(NWPU_VHR10_CLASSES),
        "names": NWPU_VHR10_CLASSES,
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

    logger.info(
        f"YOLO dataset prepared at '{out_path}' (Train: {len(train_files)}, Val: {len(val_files)}). Config: {data_yaml_path}"
    )
    return data_yaml_path
