"""PyTorch Dataset classes for all GeoVision satellite tasks."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from geovision.constants import (
    EUROSAT_CLASSES,
    LANDCOVER_AI_CLASSES,
    NWPU_VHR10_CLASSES,
)
from geovision.logger import get_logger

logger = get_logger("geovision.data.datasets")

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    class Dataset:
        pass


class EuroSATDataset(Dataset):
    """EuroSAT Sentinel-2 Scene Classification & Retrieval Dataset."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = "all",
        split_seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(EUROSAT_CLASSES)}

        # Load samples per class
        all_samples = []
        for cls_name in EUROSAT_CLASSES:
            cls_dir = self.root_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    all_samples.append((img_path, self.class_to_idx[cls_name]))

        # Deterministic split
        if split != "all" and all_samples:
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(len(all_samples))
            val_idx = int(0.8 * len(all_samples))
            if split == "train":
                self.samples = [all_samples[i] for i in indices[:val_idx]]
            elif split in ("val", "test"):
                self.samples = [all_samples[i] for i in indices[val_idx:]]
        else:
            self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as im:
            img_arr = np.array(im.convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=img_arr)
            img_tensor = augmented["image"]
        elif _HAS_TORCH:
            img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float() / 255.0
        else:
            img_tensor = img_arr

        label_tensor = torch.tensor(label, dtype=torch.long) if _HAS_TORCH else label

        return {
            "image": img_tensor,
            "label": label_tensor,
            "class_name": EUROSAT_CLASSES[label],
            "path": str(img_path),
        }


class NWPUVHR10Dataset(Dataset):
    """NWPU VHR-10 Optical Remote Sensing Object Detection Dataset."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        image_size: int = 512,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        ) if self.images_dir.exists() else []

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"

        with Image.open(img_path) as im:
            img_arr = np.array(im.convert("RGB"))

        bboxes = []
        class_labels = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        # Clamp YOLO coordinates
                        cx = min(max(cx, 0.0), 1.0)
                        cy = min(max(cy, 0.0), 1.0)
                        w = min(max(w, 0.01), 1.0)
                        h = min(max(h, 0.01), 1.0)
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(cls_id)

        if self.transform is not None and bboxes:
            augmented = self.transform(image=img_arr, bboxes=bboxes, class_labels=class_labels)
            img_out = augmented["image"]
            bboxes_out = augmented["bboxes"]
            classes_out = augmented["class_labels"]
        elif self.transform is not None:
            augmented = self.transform(image=img_arr, bboxes=[], class_labels=[])
            img_out = augmented["image"]
            bboxes_out = []
            classes_out = []
        elif _HAS_TORCH:
            img_out = torch.from_numpy(img_arr.transpose(2, 0, 1)).float() / 255.0
            bboxes_out = bboxes
            classes_out = class_labels
        else:
            img_out = img_arr
            bboxes_out = bboxes
            classes_out = class_labels

        return {
            "image": img_out,
            "bboxes": bboxes_out,
            "class_labels": classes_out,
            "path": str(img_path),
        }


class LandCoverAIDataset(Dataset):
    """LandCover.ai High-Resolution Land-Cover Semantic Segmentation Dataset."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Check direct or nested directory
        if (self.root_dir / "LandCoverAI" / "images").exists():
            self.images_dir = self.root_dir / "LandCoverAI" / "images"
            self.masks_dir = self.root_dir / "LandCoverAI" / "masks"
        elif (self.root_dir / "landcover_ai" / "images").exists():
            self.images_dir = self.root_dir / "landcover_ai" / "images"
            self.masks_dir = self.root_dir / "landcover_ai" / "masks"
        else:
            self.images_dir = self.root_dir / "images"
            self.masks_dir = self.root_dir / "masks"

        self.image_files = []
        if self.images_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                self.image_files.extend(list(self.images_dir.glob(ext)))
            self.image_files = sorted(list(set(self.image_files)))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        mask_path = None
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            candidate = self.masks_dir / f"{img_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        with Image.open(img_path) as im:
            img_arr = np.array(im.convert("RGB"))

        if mask_path is not None and mask_path.exists():
            with Image.open(mask_path) as m:
                mask_arr = np.array(m, dtype=np.uint8)
        else:
            mask_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img_arr, mask=mask_arr)
            img_tensor = augmented["image"]
            mask_tensor = (
                torch.from_numpy(augmented["mask"]).long()
                if _HAS_TORCH
                else augmented["mask"]
            )
        elif _HAS_TORCH:
            img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_arr).long()
        else:
            img_tensor = img_arr
            mask_tensor = mask_arr

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "path": str(img_path),
        }


class LEVIRCDDataset(Dataset):
    """LEVIR-CD Bi-temporal Remote Sensing Change Detection Dataset."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.a_dir = self.root_dir / "A"
        self.b_dir = self.root_dir / "B"
        self.label_dir = self.root_dir / "label"

        self.pair_files = sorted(
            list(self.a_dir.glob("*.png")) + list(self.a_dir.glob("*.jpg"))
        ) if self.a_dir.exists() else []

    def __len__(self) -> int:
        return len(self.pair_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        a_path = self.pair_files[idx]
        b_path = self.b_dir / a_path.name
        lbl_path = self.label_dir / a_path.name

        with Image.open(a_path) as im:
            img_a = np.array(im.convert("RGB"))
        with Image.open(b_path) as im:
            img_b = np.array(im.convert("RGB"))

        if lbl_path.exists():
            with Image.open(lbl_path) as im:
                mask_raw = np.array(im.convert("L"), dtype=np.uint8)
                # Convert 255/nonzero to binary 1
                mask_arr = (mask_raw > 0).astype(np.uint8)
        else:
            mask_arr = np.zeros(img_a.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img_a, image_b=img_b, mask=mask_arr)
            img_a_tensor = augmented["image"]
            img_b_tensor = (
                torch.from_numpy(augmented["image_b"].transpose(2, 0, 1)).float() / 255.0
                if not isinstance(augmented["image_b"], torch.Tensor)
                else augmented["image_b"]
            ) if _HAS_TORCH else augmented["image_b"]
            mask_tensor = (
                torch.from_numpy(augmented["mask"]).long()
                if _HAS_TORCH
                else augmented["mask"]
            )
        elif _HAS_TORCH:
            img_a_tensor = torch.from_numpy(img_a.transpose(2, 0, 1)).float() / 255.0
            img_b_tensor = torch.from_numpy(img_b.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_arr).long()
        else:
            img_a_tensor = img_a
            img_b_tensor = img_b
            mask_tensor = mask_arr

        return {
            "image_a": img_a_tensor,
            "image_b": img_b_tensor,
            "mask": mask_tensor,
            "name": a_path.name,
        }
