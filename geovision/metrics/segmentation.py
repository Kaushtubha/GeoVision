"""Semantic segmentation evaluation metrics and land-cover area distribution calculators."""

from typing import Any

import numpy as np

from geovision.constants import LANDCOVER_AI_CLASSES
from geovision.logger import get_logger

logger = get_logger("geovision.metrics.segmentation")


class SegmentationMetricsCalculator:
    """Multi-class confusion matrix and segmentation metrics calculator."""

    def __init__(
        self,
        num_classes: int = 5,
        class_names: dict[int, str] | list[str] | None = None,
        ignore_index: int | None = None,
    ):
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = [LANDCOVER_AI_CLASSES.get(i, f"Class_{i}") for i in range(num_classes)]
        elif isinstance(class_names, dict):
            self.class_names = [class_names.get(i, f"Class_{i}") for i in range(num_classes)]
        else:
            self.class_names = list(class_names)

        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset accumulated confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        preds: np.ndarray | Any,
        targets: np.ndarray | Any,
    ) -> None:
        """Accumulate predictions and ground-truth targets into confusion matrix.

        Args:
            preds: Predicted class labels (N, H, W) or (H, W), integer dtype.
            targets: Ground-truth class labels (N, H, W) or (H, W), integer dtype.
        """
        # Convert PyTorch tensors to NumPy arrays if necessary
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        if hasattr(targets, "detach"):
            targets = targets.detach().cpu().numpy()

        p_flat = np.asarray(preds, dtype=np.int64).flatten()
        t_flat = np.asarray(targets, dtype=np.int64).flatten()

        if p_flat.shape != t_flat.shape:
            raise ValueError(f"Shape mismatch: preds {p_flat.shape} vs targets {t_flat.shape}")

        mask = (t_flat >= 0) & (t_flat < self.num_classes) & (p_flat >= 0) & (p_flat < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (t_flat != self.ignore_index)

        p_valid = p_flat[mask]
        t_valid = t_flat[mask]

        # Fast bincount 2D confusion matrix accumulation
        indices = self.num_classes * t_valid + p_valid
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))

    def compute(self) -> dict[str, Any]:
        """Compute mIoU, per-class IoU, Dice/F1 scores, and overall accuracy."""
        cm = self.confusion_matrix.astype(np.float64)
        total_pixels = cm.sum()

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Per-class IoU: TP / (TP + FP + FN)
        denominator_iou = tp + fp + fn
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)
        valid_iou_mask = denominator_iou > 0
        per_class_iou[valid_iou_mask] = tp[valid_iou_mask] / denominator_iou[valid_iou_mask]

        # Per-class Dice: 2*TP / (2*TP + FP + FN)
        denominator_dice = 2.0 * tp + fp + fn
        per_class_dice = np.zeros(self.num_classes, dtype=np.float64)
        valid_dice_mask = denominator_dice > 0
        per_class_dice[valid_dice_mask] = (2.0 * tp[valid_dice_mask]) / denominator_dice[valid_dice_mask]

        # Overall Accuracy
        overall_acc = tp.sum() / total_pixels if total_pixels > 0 else 0.0

        # Frequency-Weighted IoU
        freq = cm.sum(axis=1) / total_pixels if total_pixels > 0 else np.zeros(self.num_classes)
        fwiou = float(np.sum(freq[valid_iou_mask] * per_class_iou[valid_iou_mask]))

        # Mean IoU and Mean Dice over active classes
        miou = float(np.mean(per_class_iou[valid_iou_mask])) if np.any(valid_iou_mask) else 0.0
        mean_dice = float(np.mean(per_class_dice[valid_dice_mask])) if np.any(valid_dice_mask) else 0.0

        per_class_iou_dict = {
            self.class_names[i]: float(per_class_iou[i])
            for i in range(self.num_classes)
        }
        per_class_dice_dict = {
            self.class_names[i]: float(per_class_dice[i])
            for i in range(self.num_classes)
        }

        return {
            "mIoU": miou,
            "mean_dice": mean_dice,
            "overall_accuracy": float(overall_acc),
            "freq_weighted_iou": fwiou,
            "per_class_iou": per_class_iou_dict,
            "per_class_dice": per_class_dice_dict,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def compute_landcover_distribution(
    mask: np.ndarray | Any,
    num_classes: int = 5,
    class_names: dict[int, str] | list[str] | None = None,
    gsd_meters: float | None = None,
) -> dict[str, Any]:
    """Calculate pixel counts, percentages, and ground area coverage per land cover class.

    Args:
        mask: 2D integer segmentation mask (H, W) or flattened array.
        num_classes: Number of distinct classes.
        class_names: Optional mapping or list of class names.
        gsd_meters: Ground Sampling Distance in meters per pixel (e.g., 0.25m or 0.5m).

    Returns:
        Dictionary with per-class statistics and totals.
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask_arr = np.asarray(mask, dtype=np.int64)
    total_pixels = int(mask_arr.size)

    if class_names is None:
        names = [LANDCOVER_AI_CLASSES.get(i, f"Class_{i}") for i in range(num_classes)]
    elif isinstance(class_names, dict):
        names = [class_names.get(i, f"Class_{i}") for i in range(num_classes)]
    else:
        names = list(class_names)

    counts = np.bincount(mask_arr.flatten(), minlength=num_classes)[:num_classes]

    pixel_counts = {}
    percentages = {}
    area_sq_m = {}
    area_hectares = {}
    area_sq_km = {}

    pixel_area_m2 = (gsd_meters ** 2) if gsd_meters is not None else None

    for i in range(num_classes):
        c_name = names[i]
        c_count = int(counts[i])
        pixel_counts[c_name] = c_count
        pct = (c_count / total_pixels * 100.0) if total_pixels > 0 else 0.0
        percentages[c_name] = round(pct, 2)

        if pixel_area_m2 is not None:
            sq_m = c_count * pixel_area_m2
            area_sq_m[c_name] = round(sq_m, 2)
            area_hectares[c_name] = round(sq_m / 10000.0, 4)
            area_sq_km[c_name] = round(sq_m / 1_000_000.0, 6)

    result: dict[str, Any] = {
        "total_pixels": total_pixels,
        "pixel_counts": pixel_counts,
        "percentages": percentages,
    }

    if pixel_area_m2 is not None:
        result["gsd_meters"] = gsd_meters
        result["total_area_sq_km"] = round((total_pixels * pixel_area_m2) / 1_000_000.0, 6)
        result["area_sq_meters"] = area_sq_m
        result["area_hectares"] = area_hectares
        result["area_sq_km"] = area_sq_km

    return result
