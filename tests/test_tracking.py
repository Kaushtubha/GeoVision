"""Unit tests for experiment tracking with MLflow."""

import tempfile
from pathlib import Path

from geovision.tracking.mlflow_tracker import ExperimentTracker


def test_tracker_run_lifecycle():
    """Verify tracker starts runs, logs params, metrics, artifacts, and finishes cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file:///{Path(tmpdir).as_posix()}"
        tracker = ExperimentTracker(
            experiment_name="TestGeoVision",
            tracking_uri=tracking_uri,
            enabled=True,
        )

        with tracker.start_run(run_name="smoke_test_run") as active_tracker:
            active_tracker.log_params({"model": "yolov8n", "lr": 0.001, "batch_size": 4})
            active_tracker.log_metrics({"train_loss": 0.45, "val_mIoU": 0.78}, step=1)
            active_tracker.log_metrics({"train_loss": 0.30, "val_mIoU": 0.82}, step=2)

            # Create a dummy artifact to log
            dummy_file = Path(tmpdir) / "metrics_summary.json"
            dummy_file.write_text('{"status": "success"}', encoding="utf-8")
            active_tracker.log_artifact(dummy_file)

        assert tracker._active_run is None


def test_offline_tracker():
    """Verify tracker functions gracefully when MLflow is disabled."""
    tracker = ExperimentTracker(enabled=False)
    with tracker.start_run(run_name="offline_run") as t:
        t.log_params({"key": "value"})
        t.log_metrics({"acc": 0.99})
    assert tracker.enabled is False
