"""Satellite-specific data augmentations and preprocessing pipelines using Albumentations."""

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

# Standard optical remote sensing / ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_classification_transforms(
    image_size: int = 64,
    is_train: bool = True,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Any:
    """Satellite scene classification / retrieval augmentation pipeline (D4 Dihedral symmetries)."""
    if not _HAS_ALBUMENTATIONS:
        return None

    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_detection_transforms(
    image_size: int = 512,
    is_train: bool = True,
) -> Any:
    """Satellite object detection augmentation pipeline preserving bounding box geometries."""
    if not _HAS_ALBUMENTATIONS:
        return None

    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_area=4.0,
        min_visibility=0.1,
    )

    if is_train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ],
            bbox_params=bbox_params,
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
        ],
        bbox_params=bbox_params,
    )


def get_segmentation_transforms(
    image_size: int = 512,
    is_train: bool = True,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Any:
    """Land-cover semantic segmentation pipeline with synchronized image and mask augmentations."""
    if not _HAS_ALBUMENTATIONS:
        return None

    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_change_detection_transforms(
    image_size: int = 256,
    is_train: bool = True,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Any:
    """Bi-temporal change detection pipeline synchronizing Pair A, Pair B, and binary Change Mask."""
    if not _HAS_ALBUMENTATIONS:
        return None

    additional_targets = {"image_b": "image"}

    if is_train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )
