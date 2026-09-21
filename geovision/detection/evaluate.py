"""Object detection evaluation and benchmarking pipeline."""

from pathlib import Path
from typing import Any

from geovision.constants import NWPU_VHR10_CLASSES
from geovision.detection.dataset_prep import prepare_yolo_dataset
from geovision.detection.model import YOLODetector
from geovision.logger import get_logger
from geovision.metrics.detection import evaluate_detection_metrics
from geovision.tracking.mlflow_tracker import get_tracker

logger = get_logger("geovision.detection.evaluate")


def evaluate_yolo_detector(
    weights_path: str | Path | None = None,
    data_yaml: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    smoke: bool = False,
    experiment_name: str = "GeoVision_Detection_Eval",
) -> dict[str, Any]:
    """Run full evaluation on validation/test set and report mAP metrics.

    Args:
        weights_path: Path to trained weights (defaults to 'yolov8n.pt').
        data_yaml: Path to data.yaml dataset config.
        conf_threshold: Prediction confidence threshold.
        iou_threshold: NMS IoU threshold.
        smoke: If True, uses smoke dataset.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary of computed metrics (mAP@50, mAP@75, mAP@50-95, per-class AP).
    """
    model_path = weights_path or "yolov8n.pt"
    detector = YOLODetector(model_path=model_path, device="cpu")

    if data_yaml is None:
        data_yaml = prepare_yolo_dataset(smoke=smoke)

    data_dir = Path(data_yaml).parent
    val_images_dir = data_dir / "images" / "val"
    val_labels_dir = data_dir / "labels" / "val"

    image_files = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
    logger.info(f"Evaluating {len(image_files)} validation images...")

    predictions = []
    ground_truths = []

    for img_path in image_files:
        # 1. Run prediction
        preds = detector.predict(
            img_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        predictions.append(preds)

        # 2. Parse ground truth
        lbl_file = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        gt_labels = []

        if lbl_file.exists():
            h, w = preds["image_shape"][:2]
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(cls_id)

        ground_truths.append({"boxes": gt_boxes, "labels": gt_labels})

    # 3. Compute metrics
    report = evaluate_detection_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        num_classes=len(NWPU_VHR10_CLASSES),
        class_names=NWPU_VHR10_CLASSES,
    )

    tracker = get_tracker(experiment_name=experiment_name, enabled=True)
    with tracker.start_run(run_name="detection_evaluation") as active_tracker:
        active_tracker.log_params({
            "model_path": str(model_path),
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "total_val_images": len(image_files),
        })
        active_tracker.log_metrics({
            "mAP@50": report["mAP@50"],
            "mAP@75": report["mAP@75"],
            "mAP@50-95": report["mAP@50-95"],
        })

    return report
