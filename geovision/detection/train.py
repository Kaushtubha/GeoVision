"""Config-driven YOLO training pipeline with MLflow tracking and smoke mode."""

from pathlib import Path
from typing import Any

from geovision.config import DetectionConfig, GeoVisionConfig, load_config
from geovision.detection.dataset_prep import prepare_yolo_dataset
from geovision.logger import get_logger
from geovision.tracking.mlflow_tracker import get_tracker

logger = get_logger("geovision.detection.train")

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


def train_yolo_detector(
    cfg: DetectionConfig | GeoVisionConfig | None = None,
    smoke: bool = False,
    experiment_name: str = "GeoVision_Detection",
) -> dict[str, Any]:
    """Train or fine-tune YOLO model on satellite detection dataset.

    Args:
        cfg: Optional configuration object.
        smoke: If True, runs fast 1-epoch smoke test with tiny subset on CPU.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary of final metrics and artifact paths.
    """
    if cfg is None:
        master_cfg = load_config(smoke=smoke)
        det_cfg = master_cfg.detection
    elif isinstance(cfg, GeoVisionConfig):
        det_cfg = cfg.detection
        smoke = smoke or cfg.smoke
    else:
        det_cfg = cfg

    epochs = det_cfg.smoke_epochs if smoke else det_cfg.epochs
    batch_size = 4 if smoke else det_cfg.batch_size
    img_size = det_cfg.image_size
    device = det_cfg.device

    save_dir = Path(det_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare YOLO dataset
    data_yaml = prepare_yolo_dataset(smoke=smoke)

    tracker = get_tracker(experiment_name=experiment_name, enabled=True)

    with tracker.start_run(
        run_name=f"yolo_train_{'smoke' if smoke else 'full'}",
        tags={"task": "detection", "smoke": smoke, "model": det_cfg.model_name},
    ) as active_tracker:
        active_tracker.log_params({
            "model_name": det_cfg.model_name,
            "dataset": det_cfg.dataset_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": img_size,
            "learning_rate": det_cfg.learning_rate,
            "device": device,
            "smoke": smoke,
        })

        if not _HAS_ULTRALYTICS:
            logger.warning("Ultralytics not installed. Operating in stub execution mode.")
            dummy_metrics = {"mAP@50": 0.0, "mAP@50-95": 0.0, "epochs_completed": 0}
            active_tracker.log_metrics(dummy_metrics)
            return dummy_metrics

        # 2. Initialize YOLO
        logger.info(f"Initializing YOLO model: '{det_cfg.model_name}'...")
        model = YOLO(det_cfg.model_name)

        # 3. Train
        logger.info(f"Starting YOLO training: {epochs} epoch(s), batch_size={batch_size}, device={device}...")
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(save_dir.parent),
            name="yolo_run",
            exist_ok=True,
            verbose=False,
            plots=False,
        )

        # 4. Extract metrics
        metrics = {}
        try:
            val_results = model.val(data=str(data_yaml), device=device, verbose=False)
            metrics = {
                "mAP@50": round(float(val_results.box.map50), 4),
                "mAP@75": round(float(val_results.box.map75), 4),
                "mAP@50-95": round(float(val_results.box.map), 4),
                "precision": round(float(val_results.box.mp), 4),
                "recall": round(float(val_results.box.mr), 4),
            }
        except Exception as e:
            logger.warning(f"Could not extract automated validation metrics: {e}")
            metrics = {"mAP@50": 0.0, "mAP@50-95": 0.0}

        active_tracker.log_metrics(metrics)

        # 5. Export weights
        best_weight = save_dir.parent / "yolo_run" / "weights" / "best.pt"
        target_weight = save_dir / "best.pt"
        if best_weight.exists():
            import shutil
            shutil.copy2(best_weight, target_weight)
            active_tracker.log_artifact(target_weight)
            logger.info(f"Exported best weights to: {target_weight}")

        logger.info(f"Training completed successfully! Metrics: {metrics}")
        return {
            "metrics": metrics,
            "weights_path": str(target_weight if target_weight.exists() else best_weight),
            "smoke": smoke,
        }
