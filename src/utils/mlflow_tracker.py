"""MLflow experiment tracking integration."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.core.config import get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for ALPR system.
    Tracks model performance, parameters, and artifacts.
    """

    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the experiment
        """
        self.settings = get_settings()
        self.mlflow_config = self.settings.mlflow

        if not self.mlflow_config.enabled:
            logger.warning("MLflow tracking is disabled")
            self.enabled = False
            return

        self.enabled = True

        # Set tracking URI
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)

        # Set experiment
        self.experiment_name = experiment_name or self.mlflow_config.experiment_name
        mlflow.set_experiment(self.experiment_name)

        # Get experiment
        self.client = MlflowClient()
        self.experiment = self.client.get_experiment_by_name(self.experiment_name)

        logger.info(f"MLflow tracking initialized: {self.experiment_name}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Dictionary of tags

        Returns:
            Active run
        """
        if not self.enabled:
            return None

        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Default tags
        default_tags = {
            "mlflow.runName": run_name,
            "system": "ALPR",
            "version": self.settings.app_version,
        }

        if tags:
            default_tags.update(tags)

        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")

        return run

    def end_run(self):
        """End the current MLflow run."""
        if not self.enabled:
            return

        mlflow.end_run()
        logger.info("Ended MLflow run")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return

        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.enabled:
            return

        mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.enabled:
            return

        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.enabled:
            return

        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Log an artifact file.

        Args:
            artifact_path: Path to artifact file
            artifact_name: Optional name for the artifact
        """
        if not self.enabled:
            return

        mlflow.log_artifact(artifact_path, artifact_name)
        logger.debug(f"Logged artifact: {artifact_path}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
        """
        if not self.enabled:
            return

        mlflow.log_dict(dictionary, filename)

    def log_model_info(self, model_name: str, model_version: str, model_params: Dict):
        """
        Log model information.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_params: Model parameters
        """
        if not self.enabled:
            return

        self.log_param("model_name", model_name)
        self.log_param("model_version", model_version)
        self.log_params({f"model_{k}": v for k, v in model_params.items()})

    def log_detection_metrics(
        self,
        total_frames: int,
        total_detections: int,
        successful_reads: int,
        avg_confidence: float,
        processing_time: float,
    ):
        """
        Log detection metrics.

        Args:
            total_frames: Total frames processed
            total_detections: Total detections made
            successful_reads: Number of successful plate reads
            avg_confidence: Average OCR confidence
            processing_time: Total processing time
        """
        if not self.enabled:
            return

        metrics = {
            "total_frames": float(total_frames),
            "total_detections": float(total_detections),
            "successful_reads": float(successful_reads),
            "detection_rate": successful_reads / total_detections if total_detections > 0 else 0,
            "avg_confidence": avg_confidence,
            "processing_time": processing_time,
            "fps": total_frames / processing_time if processing_time > 0 else 0,
        }

        self.log_metrics(metrics)

    def log_video_processing(
        self,
        video_path: str,
        video_duration: float,
        video_fps: float,
        results: Dict,
    ):
        """
        Log video processing results.

        Args:
            video_path: Path to video
            video_duration: Video duration in seconds
            video_fps: Video FPS
            results: Processing results dictionary
        """
        if not self.enabled:
            return

        # Log video info as params
        self.log_params({
            "video_filename": Path(video_path).name,
            "video_duration": video_duration,
            "video_fps": video_fps,
        })

        # Log results as metrics
        self.log_metrics({
            "frames_processed": float(results.get("frames_processed", 0)),
            "detections": float(results.get("detections", 0)),
            "processing_time": results.get("processing_time", 0),
        })

    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        mAP: float,
        precision: float,
        recall: float,
    ):
        """
        Log training metrics (for model training).

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            mAP: Mean Average Precision
            precision: Precision score
            recall: Recall score
        """
        if not self.enabled:
            return

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mAP": mAP,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0,
        }

        self.log_metrics(metrics, step=epoch)

    def save_model(self, model, model_name: str, metadata: Optional[Dict] = None):
        """
        Save a PyTorch model to MLflow.

        Args:
            model: PyTorch model
            model_name: Name for the model
            metadata: Optional metadata
        """
        if not self.enabled:
            return

        try:
            mlflow.pytorch.log_model(
                model,
                model_name,
                registered_model_name=model_name,
            )

            if metadata:
                self.log_dict(metadata, f"{model_name}_metadata.json")

            logger.info(f"Saved model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def get_best_run(self, metric_name: str, ascending: bool = False):
        """
        Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to compare
            ascending: Sort in ascending order (True) or descending (False)

        Returns:
            Best run information
        """
        if not self.enabled:
            return None

        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if runs:
            return runs[0]
        return None

    def compare_runs(self, run_ids: list, metrics: list):
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to compare

        Returns:
            Comparison data
        """
        if not self.enabled:
            return None

        comparison = []

        for run_id in run_ids:
            run = self.client.get_run(run_id)

            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
            }

            for metric in metrics:
                run_data[metric] = run.data.metrics.get(metric)

            comparison.append(run_data)

        return comparison


# Global tracker instance
_tracker: Optional[MLflowTracker] = None


def get_tracker(experiment_name: Optional[str] = None) -> MLflowTracker:
    """Get or create MLflow tracker instance."""
    global _tracker
    if _tracker is None or (experiment_name and experiment_name != _tracker.experiment_name):
        _tracker = MLflowTracker(experiment_name)
    return _tracker
