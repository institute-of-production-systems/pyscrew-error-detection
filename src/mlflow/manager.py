import logging
import os

import numpy as np
import pandas as pd

import mlflow
from src.evaluation.results import (
    DatasetResult,
    ExperimentResult,
    FoldResult,
    ModelResult,
)
from src.utils import get_logger
from src.utils.exceptions import ExperimentEvaluationError


class MLflowManager:
    """Handles MLflow tracking with clean start/update pattern for 4-level hierarchy."""

    def __init__(self, port: int = 5000):
        self.port = port
        self.logger = get_logger(__name__)

        # Suppress MLflow's verbose logging
        self._suppress_mlflow_logging()

    def _suppress_mlflow_logging(self):
        """Suppress MLflow's auto-generated messages."""
        # Set MLflow logging level to ERROR to suppress INFO messages
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("mlflow.tracking").setLevel(logging.ERROR)
        logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

        # Disable MLflow's run start/end messages via environment variable
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

        # Additional environment variables to suppress MLflow output
        os.environ["MLFLOW_TRACKING_SILENT"] = "true"

    def setup_tracking(self, sampling_selection: str) -> None:
        """Initialize MLflow tracking connection."""
        try:
            mlflow.set_tracking_uri(f"http://localhost:{self.port}")
            mlflow.set_experiment(sampling_selection)
            # Disable auto-logging for explicit control
            self._suppress_mlflow_logging()
            self.logger.info(f"MLflow tracking initialized for '{sampling_selection}'")
        except Exception as e:
            raise ExperimentEvaluationError(
                f"MLflow initialization failed: {str(e)}. "
                "Please ensure MLflow server is running."
            ) from e

    # ================================
    # START METHODS - Initialize runs
    # ================================

    def start_experiment_run(self, experiment_result: ExperimentResult) -> None:
        """Initialize main experiment run with basic structure."""
        mlflow.start_run(run_name=experiment_result.run_name, nested=False)
        mlflow.set_tags(experiment_result.get_tags())
        self.logger.info(f"Started experiment run: '{experiment_result.run_name}'")

    def start_dataset_run(self, dataset_result: DatasetResult) -> None:
        """Initialize dataset run with metadata."""
        mlflow.start_run(run_name=dataset_result.name, nested=True)
        mlflow.set_tags(dataset_result.get_tags())
        self.logger.info(f"Started dataset run: {dataset_result.name}")

    def start_model_run(self, model_result: ModelResult) -> None:
        """Initialize model run with placeholder metrics."""
        mlflow.start_run(run_name=model_result.name, nested=True)
        mlflow.set_tags(model_result.get_tags())
        mlflow.log_metrics(model_result.get_metrics())
        self.logger.debug(f"Started model run: {model_result.name}")

    def start_fold_run(self, fold_result: FoldResult) -> None:
        """Initialize fold run structure."""
        mlflow.start_run(run_name=fold_result.name, nested=True)
        mlflow.set_tags(fold_result.get_tags())
        mlflow.log_metrics(fold_result.get_metrics())
        mlflow.log_params(fold_result.get_params())
        self.logger.debug(f"Started fold run: {fold_result.name}")

    # ================================
    # UPDATE METHODS - Push results
    # ================================

    def update_model_run(self, model_result: ModelResult) -> None:
        """Update model run with current metrics, tags, and confusion matrix."""
        mlflow.set_tags(model_result.get_tags())
        mlflow.log_metrics(model_result.get_metrics())
        # Log a summed-up matrix if available
        if model_result.fold_results:
            self._log_model_confusion_matrix(model_result)

    def update_fold_run(self, fold_result: FoldResult) -> None:
        """Update fold run with actual results after fold completion."""
        mlflow.set_tags(fold_result.get_tags())
        mlflow.log_metrics(fold_result.get_metrics())
        # Log confusion matrix if available
        if fold_result.y_true is not None and fold_result.y_pred is not None:
            self._log_fold_confusion_matrix(fold_result)

    def update_dataset_run(self, dataset_result: DatasetResult) -> None:
        mlflow.set_tags(dataset_result.get_tags())

    def update_experiment_run(self, experiment_result: ExperimentResult) -> None:
        """Update experiment run with progress counters."""
        mlflow.set_tags(experiment_result.get_tags())

    # ================================
    # HELPER METHODS
    # ================================

    def _save_and_log_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        """Save confusion matrix as CSV and log as MLflow artifact."""
        filename = "confusion_matrix.csv"
        pd.DataFrame(confusion_matrix).to_csv(filename, index=False)
        mlflow.log_artifact(filename)

        # Clean up temp file
        if os.path.exists(filename):
            os.remove(filename)

    def _log_fold_confusion_matrix(self, fold_result: FoldResult) -> None:
        """Log confusion matrix for a single fold."""
        self._save_and_log_confusion_matrix(fold_result.get_confusion_matrix())

    def _log_model_confusion_matrix(self, model_result: ModelResult) -> None:
        """Log summed confusion matrix across all folds as artifact."""
        if not model_result.fold_results:
            return

        # Sum up confusion matrices from all folds
        summed_cm = None
        for fold_result in model_result.fold_results:
            fold_cm = fold_result.get_confusion_matrix()
            if summed_cm is None:
                summed_cm = fold_cm
            else:
                summed_cm += fold_cm

        if summed_cm is not None:
            self._save_and_log_confusion_matrix(summed_cm)

    def end_run(self) -> None:
        """End current MLflow run safely."""
        try:
            mlflow.end_run()
        except Exception as e:
            self.logger.warning(f"Error ending MLflow run: {str(e)}")
