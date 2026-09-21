"""Evaluation metrics for all GeoVision tasks."""

from geovision.metrics.detection import (
    compute_ap_from_pr,
    compute_iou_matrix,
    evaluate_detection_metrics,
)

__all__ = [
    "compute_iou_matrix",
    "compute_ap_from_pr",
    "evaluate_detection_metrics",
]
