"""Object detection evaluation metrics: IoU, Precision-Recall Curves, and mAP (COCO / Pascal VOC standards)."""

from typing import Any

import numpy as np

from geovision.constants import NWPU_VHR10_CLASSES
from geovision.logger import get_logger

logger = get_logger("geovision.metrics.detection")


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute Intersection-over-Union (IoU) matrix between two sets of bounding boxes.

    Args:
        boxes1: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M) with values in [0.0, 1.0].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Intersection coordinates
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = area1 + area2 - inter_area
    iou = inter_area / np.maximum(union_area, 1e-8)
    return iou


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate Average Precision (AP) via all-point envelope interpolation (COCO/Pascal VOC standard)."""
    # Append boundary points
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute monotonic precision envelope (from right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate_detection_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    num_classes: int = 10,
    iou_thresholds: list[float] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate mAP@50, mAP@75, mAP@50-95, and per-class AP breakdown across test dataset.

    Args:
        predictions: List of dicts with keys 'boxes' (N, 4), 'scores' (N,), 'labels' (N,).
        ground_truths: List of dicts with keys 'boxes' (M, 4), 'labels' (M,).
        num_classes: Number of unique object classes.
        iou_thresholds: List of IoU thresholds (defaults to [0.5, 0.55, ..., 0.95]).
        class_names: Optional list of class names.

    Returns:
        Structured evaluation report dictionary.
    """
    if iou_thresholds is None:
        iou_thresholds = list(np.arange(0.5, 1.0, 0.05))

    names = class_names or NWPU_VHR10_CLASSES

    # Results dictionary: ap_matrix[class_idx, iou_thresh_idx]
    ap_matrix = np.zeros((num_classes, len(iou_thresholds)), dtype=np.float32)
    per_class_counts = {c: 0 for c in range(num_classes)}

    # Count total ground truths per class
    for gt in ground_truths:
        for lbl in gt.get("labels", []):
            if 0 <= int(lbl) < num_classes:
                per_class_counts[int(lbl)] += 1

    for c in range(num_classes):
        total_gt_c = per_class_counts[c]
        if total_gt_c == 0:
            continue

        # Extract all detections for class c across entire dataset
        c_preds_scores = []
        c_preds_img_idx = []
        c_preds_boxes = []

        for img_idx, pred in enumerate(predictions):
            boxes = np.array(pred.get("boxes", []), dtype=np.float32)
            scores = np.array(pred.get("scores", []), dtype=np.float32)
            labels = np.array(pred.get("labels", []), dtype=np.int32)

            mask = (labels == c)
            if np.any(mask):
                for b, s in zip(boxes[mask], scores[mask]):
                    c_preds_boxes.append(b)
                    c_preds_scores.append(s)
                    c_preds_img_idx.append(img_idx)

        if not c_preds_scores:
            continue

        # Sort detections descending by confidence score
        sort_order = np.argsort(-np.array(c_preds_scores))
        c_preds_boxes = np.array(c_preds_boxes)[sort_order]
        c_preds_img_idx = np.array(c_preds_img_idx)[sort_order]

        # Evaluate against ground truths for each IoU threshold
        for t_idx, iou_thresh in enumerate(iou_thresholds):
            tp = np.zeros(len(c_preds_scores), dtype=np.float32)
            fp = np.zeros(len(c_preds_scores), dtype=np.float32)

            # Track detected ground truths per image to avoid double-counting
            detected_gts = {img_idx: set() for img_idx in range(len(ground_truths))}

            for d_idx, (p_box, img_idx) in enumerate(zip(c_preds_boxes, c_preds_img_idx)):
                gt = ground_truths[img_idx]
                gt_boxes = np.array(gt.get("boxes", []), dtype=np.float32)
                gt_labels = np.array(gt.get("labels", []), dtype=np.int32)

                c_gt_mask = (gt_labels == c)
                c_gt_boxes = gt_boxes[c_gt_mask]

                if len(c_gt_boxes) == 0:
                    fp[d_idx] = 1.0
                    continue

                # Compute IoU between prediction and all GTs of class c in this image
                ious = compute_iou_matrix(p_box[np.newaxis, :], c_gt_boxes)[0]
                best_gt_idx = int(np.argmax(ious))
                best_iou = float(ious[best_gt_idx])

                if best_iou >= iou_thresh and best_gt_idx not in detected_gts[img_idx]:
                    tp[d_idx] = 1.0
                    detected_gts[img_idx].add(best_gt_idx)
                else:
                    fp[d_idx] = 1.0

            # Compute cumulative precision and recall
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / total_gt_c
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

            ap_matrix[c, t_idx] = compute_ap_from_pr(recalls, precisions)

    # Derive macro metrics
    # IoU@0.5 is at index 0 (assuming thresholds start at 0.5)
    map50 = float(np.mean(ap_matrix[:, 0]))
    # IoU@0.75 is at index 5 (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    map75 = float(np.mean(ap_matrix[:, 5])) if ap_matrix.shape[1] > 5 else 0.0
    map50_95 = float(np.mean(ap_matrix))

    per_class_ap50 = {
        names[c] if c < len(names) else f"class_{c}": float(ap_matrix[c, 0])
        for c in range(num_classes)
    }

    report = {
        "mAP@50": round(map50, 4),
        "mAP@75": round(map75, 4),
        "mAP@50-95": round(map50_95, 4),
        "per_class_AP@50": per_class_ap50,
        "total_ground_truths": per_class_counts,
    }

    logger.info(f"Detection Evaluation: mAP@50={report['mAP@50']:.4f}, mAP@50-95={report['mAP@50-95']:.4f}")
    return report
