"""MLflow experiment tracking integration for GeoVision."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from geovision.constants import MLRUNS_DIR
from geovision.logger import get_logger

# Opt into local filesystem tracking in MLflow 3.x
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logger = get_logger("geovision.tracking")

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


class ExperimentTracker:
    """Manages MLflow experiment lifecycle, metric logging, and artifact persistence."""

    def __init__(
        self,
        experiment_name: str = "GeoVision",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ):
        self.experiment_name = experiment_name
        self.enabled = enabled and _HAS_MLFLOW
        self.tracking_uri = tracking_uri or str(MLRUNS_DIR.resolve())
        self._active_run = None

        if self.enabled:
            try:
                # Ensure tracking directory exists
                Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                logger.info(
                    f"MLflow tracking initialized for experiment '{self.experiment_name}' at {self.tracking_uri}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}. Running in local tracking mode.")
                self.enabled = False
        else:
            if not _HAS_MLFLOW:
                logger.info("MLflow not installed. Operating in offline metrics logging mode.")

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
        nested: bool = False,
    ):
        """Context manager to start and cleanly conclude an MLflow run."""
        if self.enabled:
            try:
                with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
                    self._active_run = run
                    logger.info(f"Started MLflow run '{run_name or run.info.run_id}' (ID: {run.info.run_id})")
                    yield self
            finally:
                self._active_run = None
        else:
            logger.info(f"[Offline Run] Started run '{run_name or 'unnamed'}'")
            try:
                yield self
            finally:
                logger.info(f"[Offline Run] Finished run '{run_name or 'unnamed'}'")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log key-value hyperparameters."""
        # Flatten dictionary for MLflow compatibility
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                flat_params[k] = json.dumps(v)
            else:
                flat_params[k] = v

        if self.enabled:
            try:
                mlflow.log_params(flat_params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")
        else:
            logger.info(f"[Offline Params] {flat_params}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric evaluation or training metrics."""
        import re
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                # MLflow metrics only permit alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ) and slashes (/)
                sanitized_key = re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", str(k))
                clean_metrics[sanitized_key] = float(v)

        if self.enabled:
            try:
                mlflow.log_metrics(clean_metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
        else:
            step_str = f" [Step {step}]" if step is not None else ""
            logger.info(f"[Offline Metrics{step_str}] {clean_metrics}")

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file or directory as an MLflow artifact."""
        target = Path(local_path)
        if not target.exists():
            logger.warning(f"Artifact not found at {target}, skipping log_artifact")
            return

        if self.enabled:
            try:
                if target.is_file():
                    mlflow.log_artifact(str(target), artifact_path=artifact_path)
                elif target.is_dir():
                    mlflow.log_artifacts(str(target), artifact_path=artifact_path)
                logger.info(f"Logged artifact: {target.name}")
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")
        else:
            logger.info(f"[Offline Artifact] Saved at: {target}")


def get_tracker(
    experiment_name: str = "GeoVision",
    tracking_uri: str | None = None,
    enabled: bool = True,
) -> ExperimentTracker:
    """Factory helper for ExperimentTracker."""
    return ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        enabled=enabled,
    )
