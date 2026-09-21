"""Dataset statistical analysis and class distribution calculation."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import numpy as np

from geovision.logger import get_logger

logger = get_logger("geovision.data.statistics")


def calculate_dataset_stats(dataset_path: Path | str, task: str) -> Dict[str, Any]:
    """Calculate statistical distribution for a dataset directory.

    Args:
        dataset_path: Path to dataset root.
        task: One of 'classification', 'detection', 'segmentation', 'change'.

    Returns:
        Dictionary containing counts, dimensions, class frequencies, and storage footprint.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    stats: Dict[str, Any] = {
        "dataset_path": str(path),
        "task": task,
        "total_samples": 0,
        "class_counts": {},
        "image_shapes": [],
        "total_disk_mb": 0.0,
    }

    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    stats["total_disk_mb"] = round(total_bytes / (1024 * 1024), 2)

    if task in ("classification", "retrieval"):
        # Subdirectory per class structure (like EuroSAT)
        class_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        total = 0
        for cd in class_dirs:
            imgs = [f for f in cd.iterdir() if f.suffix.lower() in [".jpg", ".png", ".tif", ".tiff"]]
            stats["class_counts"][cd.name] = len(imgs)
            total += len(imgs)
            if imgs and len(stats["image_shapes"]) < 5:
                with Image.open(imgs[0]) as im:
                    stats["image_shapes"].append(im.size)
        stats["total_samples"] = total

    elif task == "detection":
        images_dir = path / "images"
        labels_dir = path / "labels"
        if images_dir.exists():
            imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            stats["total_samples"] = len(imgs)
            if labels_dir.exists():
                class_counter = Counter()
                for lbl_file in labels_dir.glob("*.txt"):
                    with open(lbl_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_counter[int(parts[0])] += 1
                stats["class_counts"] = dict(class_counter)

    elif task == "segmentation":
        images_dir = path / "images"
        masks_dir = path / "masks"
        if images_dir.exists():
            imgs = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            stats["total_samples"] = len(imgs)
        if masks_dir.exists():
            mask_files = list(masks_dir.glob("*.png"))
            if mask_files:
                # Sample first few masks to check class distribution
                sample_masks = [np.array(Image.open(m)) for m in mask_files[:5]]
                unique_classes = np.unique(np.concatenate([m.flatten() for m in sample_masks]))
                stats["unique_classes_found"] = unique_classes.tolist()

    elif task == "change":
        a_dir = path / "A"
        b_dir = path / "B"
        if a_dir.exists() and b_dir.exists():
            pairs = list(a_dir.glob("*.png")) + list(a_dir.glob("*.jpg"))
            stats["total_samples"] = len(pairs)

    logger.info(f"Dataset stats computed for '{path.name}' ({task}): {stats['total_samples']} samples, {stats['total_disk_mb']} MB")
    return stats
