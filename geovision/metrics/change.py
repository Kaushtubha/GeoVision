"""Bi-temporal change detection evaluation metrics and statistical shift calculators."""

from typing import Any

import numpy as np

from geovision.logger import get_logger

logger = get_logger("geovision.metrics.change")


class ChangeMetricsCalculator:
    """Accumulator for bi-temporal binary change detection metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self) -> None:
        """Reset metric accumulators."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(
        self,
        preds: np.ndarray | Any,
        targets: np.ndarray | Any,
    ) -> None:
        """Update confusion counts with a new batch of predictions and targets.

        Args:
            preds: Predicted change probability map or binary integer predictions (N, H, W) or (H, W).
            targets: Ground-truth binary change masks with values in {0, 1}.
        """
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        if hasattr(targets, "detach"):
            targets = targets.detach().cpu().numpy()

        p_arr = np.asarray(preds)
        t_arr = np.asarray(targets)

        # Binarize if float probabilities
        if np.issubdtype(p_arr.dtype, np.floating):
            p_bin = (p_arr >= self.threshold).astype(np.int64)
        else:
            p_bin = (p_arr > 0).astype(np.int64)

        t_bin = (t_arr > 0).astype(np.int64)

        p_flat = p_bin.flatten()
        t_flat = t_bin.flatten()

        if p_flat.shape != t_flat.shape:
            raise ValueError(f"Shape mismatch: preds {p_flat.shape} vs targets {t_flat.shape}")

        self.tp += int(np.sum((t_flat == 1) & (p_flat == 1)))
        self.fp += int(np.sum((t_flat == 0) & (p_flat == 1)))
        self.tn += int(np.sum((t_flat == 0) & (p_flat == 0)))
        self.fn += int(np.sum((t_flat == 1) & (p_flat == 0)))

    def compute(self) -> dict[str, Any]:
        """Compute binary precision, recall, F1-score (Dice), Change-IoU, and overall accuracy."""
        eps = 1e-7

        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        total = tp + fp + tn + fn

        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        f1_score = (2.0 * precision * recall) / (precision + recall + eps) if (precision + recall) > 0 else 0.0

        # IoU for Change class: TP / (TP + FP + FN)
        change_denom = tp + fp + fn
        change_iou = (tp / (change_denom + eps)) if change_denom > 0 else 0.0

        # IoU for No-Change class: TN / (TN + FP + FN)
        no_change_denom = tn + fp + fn
        no_change_iou = (tn / (no_change_denom + eps)) if no_change_denom > 0 else 0.0

        miou = (change_iou + no_change_iou) / 2.0
        overall_accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "change_iou": float(change_iou),
            "no_change_iou": float(no_change_iou),
            "mIoU": float(miou),
            "overall_accuracy": float(overall_accuracy),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
        }


def compute_change_statistics(
    change_mask: np.ndarray | Any,
    gsd_meters: float | None = 0.5,
) -> dict[str, Any]:
    """Compute physical ground area shift and quantitative change stats from a binary change mask.

    Args:
        change_mask: 2D binary integer mask (H, W) or boolean array where 1 indicates change.
        gsd_meters: Ground Sampling Distance in meters/pixel (e.g., 0.5m for LEVIR-CD).

    Returns:
        Dictionary containing pixel counts, change ratio, and ground area coverage in m2, ha, and km2.
    """
    if hasattr(change_mask, "detach"):
        change_mask = change_mask.detach().cpu().numpy()

    mask_arr = np.asarray(change_mask)
    binary_mask = (mask_arr > 0).astype(np.uint8)

    total_pixels = int(binary_mask.size)
    changed_pixels = int(np.sum(binary_mask))
    unchanged_pixels = total_pixels - changed_pixels

    change_pct = (changed_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    stats: dict[str, Any] = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "change_percentage": round(change_pct, 4),
    }

    if gsd_meters is not None:
        pixel_area_m2 = gsd_meters ** 2
        changed_sq_m = changed_pixels * pixel_area_m2
        total_sq_m = total_pixels * pixel_area_m2

        stats["gsd_meters"] = gsd_meters
        stats["changed_area_sq_meters"] = round(changed_sq_m, 2)
        stats["changed_area_hectares"] = round(changed_sq_m / 10000.0, 4)
        stats["changed_area_sq_km"] = round(changed_sq_m / 1_000_000.0, 6)
        stats["total_area_sq_km"] = round(total_sq_m / 1_000_000.0, 6)

    return stats
