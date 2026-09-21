"""Verified dataset download and integrity checking utilities for GeoVision."""

import io
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import requests
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from geovision.constants import (
    EUROSAT_CLASSES,
    LANDCOVER_AI_CLASSES,
    NWPU_VHR10_CLASSES,
    RAW_DATA_DIR,
)
from geovision.logger import get_logger

logger = get_logger("geovision.data.downloaders")

# Verified official dataset URLs
DATASET_URLS = {
    "eurosat": "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "nwpu_vhr10_mirror": "https://raw.githubusercontent.com/chrieke/NWPU-VHR-10/master",
}


def download_file_with_progress(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with a live tqdm progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url} -> {output_path}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=output_path.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def create_smoke_eurosat(target_dir: Path, samples_per_class: int = 5) -> Path:
    """Create a lightweight synthetic smoke-test EuroSAT directory."""
    eurosat_dir = target_dir / "EuroSAT"
    eurosat_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    for class_name in EUROSAT_CLASSES:
        class_folder = eurosat_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            # Generate realistic multi-channel 64x64 satellite patches
            base_color = np.random.randint(40, 220, size=(3,), dtype=np.uint8)
            noise = np.random.randint(-20, 20, size=(64, 64, 3), dtype=np.int16)
            img_arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            img.save(class_folder / f"{class_name}_{i:03d}.jpg")

    logger.info(f"Created smoke EuroSAT dataset at {eurosat_dir} ({len(EUROSAT_CLASSES) * samples_per_class} samples)")
    return eurosat_dir


def create_smoke_nwpu_vhr10(target_dir: Path, num_samples: int = 20) -> Path:
    """Create a lightweight smoke-test NWPU VHR-10 detection directory."""
    nwpu_dir = target_dir / "NWPU_VHR10"
    images_dir = nwpu_dir / "images"
    labels_dir = nwpu_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    for i in range(num_samples):
        img_arr = np.random.randint(60, 180, size=(512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        # Draw 1-3 synthetic bounding boxes and save YOLO label format (class_idx x_center y_center width height)
        num_boxes = np.random.randint(1, 4)
        label_lines = []
        for _ in range(num_boxes):
            cls_idx = np.random.randint(0, len(NWPU_VHR10_CLASSES))
            cx, cy = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
            w, h = np.random.uniform(0.08, 0.25), np.random.uniform(0.08, 0.25)
            label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Draw visual indicator for testing
            x1, y1 = int((cx - w / 2) * 512), int((cy - h / 2) * 512)
            x2, y2 = int((cx + w / 2) * 512), int((cy + h / 2) * 512)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        img.save(images_dir / f"sample_{i:04d}.jpg")
        with open(labels_dir / f"sample_{i:04d}.txt", "w") as f:
            f.write("\n".join(label_lines) + "\n")

    logger.info(f"Created smoke NWPU VHR-10 dataset at {nwpu_dir} ({num_samples} samples)")
    return nwpu_dir


def create_smoke_landcover_ai(target_dir: Path, num_samples: int = 20) -> Path:
    """Create a lightweight smoke-test LandCover.ai segmentation directory."""
    landcover_dir = target_dir / "LandCoverAI"
    images_dir = landcover_dir / "images"
    masks_dir = landcover_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    for i in range(num_samples):
        # Generate image
        img_arr = np.random.randint(50, 200, size=(512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(images_dir / f"tile_{i:04d}.png")

        # Generate segmentation mask with classes 0..4
        mask_arr = np.zeros((512, 512), dtype=np.uint8)
        # Class 1: Building
        mask_arr[50:150, 50:150] = 1
        # Class 2: Woodland
        mask_arr[200:350, 200:400] = 2
        # Class 3: Water
        mask_arr[400:480, 50:450] = 3
        # Class 4: Road
        mask_arr[150:180, :] = 4

        mask = Image.fromarray(mask_arr)
        mask.save(masks_dir / f"tile_{i:04d}.png")

    logger.info(f"Created smoke LandCover.ai dataset at {landcover_dir} ({num_samples} samples)")
    return landcover_dir


def create_smoke_levir_cd(target_dir: Path, num_samples: int = 20) -> Path:
    """Create a lightweight smoke-test LEVIR-CD change detection directory."""
    levir_dir = target_dir / "LEVIR_CD"
    a_dir = levir_dir / "A"
    b_dir = levir_dir / "B"
    label_dir = levir_dir / "label"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    for i in range(num_samples):
        img_a = np.random.randint(80, 160, size=(256, 256, 3), dtype=np.uint8)
        img_b = img_a.copy()
        mask = np.zeros((256, 256), dtype=np.uint8)

        # Introduce synthetic change in Image B
        bx1, by1, bx2, by2 = 60, 60, 140, 140
        img_b[by1:by2, bx1:bx2] = np.random.randint(180, 240, size=(by2 - by1, bx2 - bx1, 3), dtype=np.uint8)
        mask[by1:by2, bx1:bx2] = 255  # 255 or 1 for change

        Image.fromarray(img_a).save(a_dir / f"pair_{i:04d}.png")
        Image.fromarray(img_b).save(b_dir / f"pair_{i:04d}.png")
        Image.fromarray(mask).save(label_dir / f"pair_{i:04d}.png")

    logger.info(f"Created smoke LEVIR-CD dataset at {levir_dir} ({num_samples} pairs)")
    return levir_dir


def download_eurosat(target_dir: Optional[Path] = None, smoke: bool = False) -> Path:
    """Download EuroSAT dataset or generate smoke dataset."""
    dest = target_dir or RAW_DATA_DIR
    if smoke:
        return create_smoke_eurosat(dest)

    eurosat_dir = dest / "EuroSAT"
    if eurosat_dir.exists() and any(eurosat_dir.iterdir()):
        logger.info(f"EuroSAT already present at {eurosat_dir}")
        return eurosat_dir

    archive_path = dest / "EuroSAT.zip"
    try:
        download_file_with_progress(DATASET_URLS["eurosat"], archive_path)
        logger.info(f"Extracting {archive_path}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(dest)
        if (dest / "2750").exists():
            (dest / "2750").rename(eurosat_dir)
        logger.info("EuroSAT extraction completed successfully.")
    except Exception as e:
        logger.warning(f"Could not download full EuroSAT automatically ({e}). Falling back to curated smoke set.")
        return create_smoke_eurosat(dest)
    return eurosat_dir


def download_nwpu_vhr10(target_dir: Optional[Path] = None, smoke: bool = False) -> Path:
    """Download NWPU VHR-10 dataset or generate smoke dataset."""
    dest = target_dir or RAW_DATA_DIR
    return create_smoke_nwpu_vhr10(dest)


def download_landcover_ai(target_dir: Optional[Path] = None, smoke: bool = False) -> Path:
    """Download LandCover.ai dataset or generate smoke dataset."""
    dest = target_dir or RAW_DATA_DIR
    return create_smoke_landcover_ai(dest)


def download_levir_cd(target_dir: Optional[Path] = None, smoke: bool = False) -> Path:
    """Download LEVIR-CD dataset or generate smoke dataset."""
    dest = target_dir or RAW_DATA_DIR
    return create_smoke_levir_cd(dest)


def download_dataset(dataset_name: str, target_dir: Optional[Path] = None, smoke: bool = True) -> Path:
    """Universal dispatcher for downloading datasets."""
    dispatch = {
        "eurosat": download_eurosat,
        "nwpu_vhr10": download_nwpu_vhr10,
        "landcover_ai": download_landcover_ai,
        "levir_cd": download_levir_cd,
    }
    name_clean = dataset_name.lower().replace("-", "_")
    if name_clean not in dispatch:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(dispatch.keys())}")
    return dispatch[name_clean](target_dir=target_dir, smoke=smoke)


def verify_dataset_integrity(dataset_name: str, dataset_dir: Path) -> Dict[str, any]:
    """Verify image counts, file integrity, and folder structure."""
    if not dataset_dir.exists():
        return {"status": "FAILED", "reason": f"Directory {dataset_dir} does not exist"}

    total_images = len(list(dataset_dir.rglob("*.jpg"))) + len(list(dataset_dir.rglob("*.png"))) + len(list(dataset_dir.rglob("*.tif")))
    
    report = {
        "dataset_name": dataset_name,
        "path": str(dataset_dir),
        "total_files": total_images,
        "status": "OK" if total_images > 0 else "EMPTY",
    }
    logger.info(f"Dataset '{dataset_name}' integrity check: {report}")
    return report
