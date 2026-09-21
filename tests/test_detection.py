"""Unit tests for object detection metrics, YOLO dataset preparation, and inference."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from geovision.data.downloaders import download_dataset
from geovision.detection.dataset_prep import prepare_yolo_dataset
from geovision.detection.model import YOLODetector
from geovision.geo.raster import write_geotiff
from geovision.metrics.detection import (
    compute_ap_from_pr,
    compute_iou_matrix,
    evaluate_detection_metrics,
)


def test_iou_matrix_computation():
    """Verify exact IoU calculations between known bounding boxes."""
    # Box 1: [0, 0, 10, 10] (Area 100)
    # Box 2: [0, 0, 10, 10] (Identical -> IoU 1.0)
    # Box 3: [0, 0, 5, 10] (Half overlap -> Inter 50, Union 100 -> IoU 0.5)
    # Box 4: [20, 20, 30, 30] (No overlap -> IoU 0.0)
    boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    boxes2 = np.array([
        [0, 0, 10, 10],
        [0, 0, 5, 10],
        [20, 20, 30, 30],
    ], dtype=np.float32)

    ious = compute_iou_matrix(boxes1, boxes2)
    assert ious.shape == (1, 3)
    assert pytest.approx(ious[0, 0], 1e-5) == 1.0
    assert pytest.approx(ious[0, 1], 1e-5) == 0.5
    assert pytest.approx(ious[0, 2], 1e-5) == 0.0


def test_ap_from_pr_curve():
    """Verify all-point interpolated Average Precision calculation."""
    # Perfect curve: precision is 1.0 across all recalls
    recalls = np.array([0.2, 0.5, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0])
    ap = compute_ap_from_pr(recalls, precisions)
    assert pytest.approx(ap, 1e-5) == 1.0

    # Step curve
    recalls_step = np.array([0.5, 1.0])
    precisions_step = np.array([1.0, 0.5])
    ap_step = compute_ap_from_pr(recalls_step, precisions_step)
    assert 0.5 < ap_step <= 1.0


def test_evaluate_detection_metrics_mock():
    """Verify evaluation metric computation across structured predictions and ground truths."""
    ground_truths = [
        {"boxes": [[10, 10, 50, 50], [100, 100, 150, 150]], "labels": [0, 1]},
        {"boxes": [[20, 20, 60, 60]], "labels": [0]},
    ]

    # Exact matching predictions
    predictions = [
        {"boxes": [[10, 10, 50, 50], [100, 100, 150, 150]], "scores": [0.95, 0.90], "labels": [0, 1]},
        {"boxes": [[20, 20, 60, 60]], "scores": [0.85], "labels": [0]},
    ]

    report = evaluate_detection_metrics(predictions, ground_truths, num_classes=2)
    assert report["mAP@50"] == 1.0
    assert report["mAP@50-95"] > 0.90
    assert report["total_ground_truths"][0] == 2
    assert report["total_ground_truths"][1] == 1


def test_yolo_dataset_preparation():
    """Verify YOLO dataset splitting and data.yaml creation in smoke mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        out_dir = Path(tmpdir) / "yolo"
        download_dataset("nwpu_vhr10", target_dir=raw_dir, smoke=True)

        yaml_path = prepare_yolo_dataset(
            raw_dir=raw_dir / "NWPU_VHR10",
            output_dir=out_dir,
            val_split=0.2,
            smoke=True,
        )

        assert yaml_path.exists()
        assert (out_dir / "images" / "train").exists()
        assert (out_dir / "images" / "val").exists()
        assert (out_dir / "labels" / "train").exists()
        assert (out_dir / "labels" / "val").exists()


def test_yolo_detector_inference_pipeline():
    """Verify detector initialization, prediction formatting, and georeferenced output."""
    detector = YOLODetector(model_path="yolov8n.pt", device="cpu")

    # 1. Standard prediction on synthetic RGB image
    sample_img = np.random.randint(50, 200, size=(256, 256, 3), dtype=np.uint8)
    preds = detector.predict(sample_img, conf_threshold=0.1)

    assert "boxes" in preds
    assert "scores" in preds
    assert "labels" in preds
    assert "class_names" in preds
    assert preds["image_shape"] == (256, 256, 3)

    # 2. Georeferenced prediction on synthetic GeoTIFF
    with tempfile.TemporaryDirectory() as tmpdir:
        geotiff_path = Path(tmpdir) / "sample_geotiff.tif"
        transform = Affine(0.5, 0.0, 500000.0, 0.0, -0.5, 4000000.0)
        write_geotiff(geotiff_path, sample_img, transform, crs="EPSG:3857")

        geo_preds = detector.predict_georeferenced(geotiff_path, conf_threshold=0.1)
        assert geo_preds["crs"] == "EPSG:3857"
        assert "detections" in geo_preds
        assert "geojson" in geo_preds
        assert geo_preds["geojson"]["type"] == "FeatureCollection"
