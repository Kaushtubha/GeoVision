"""Object detection module for optical remote sensing satellite imagery."""

from geovision.detection.dataset_prep import prepare_yolo_dataset
from geovision.detection.evaluate import evaluate_yolo_detector
from geovision.detection.model import YOLODetector
from geovision.detection.train import train_yolo_detector

__all__ = [
    "YOLODetector",
    "prepare_yolo_dataset",
    "train_yolo_detector",
    "evaluate_yolo_detector",
]
