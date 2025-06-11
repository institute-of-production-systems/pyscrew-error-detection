import os
import logging
from typing import Optional

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
from src.utils.exceptions import FatalExperimentError


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
            mlflow.autolog(disable=True)
            mlflow.sklearn.autolog(disable=True)
            self.logger.info(f"MLflow tracking initialized for '{sampling_selection}'")
        except Exception as e:
            raise FatalExperimentError(
                f"MLflow initialization failed: {str(e)}. "
                "Please ensure MLflow server is running."
            ) from e

    # ================================
    # START METHODS - Initialize runs
    # ================================

    def start_experiment_run(self, experiment_result: ExperimentResult) -> None:
        """Initialize main experiment run with basic structure."""

        scenario_selection = experiment_result.get_scenario_selection()
        modeling_selection = experiment_result.get_modeling_selection()
        sampling_selection = experiment_result.get_sampling_selection()
        run_name = f"run_{scenario_selection}_{modeling_selection}"

        # Initialize a result dict to hold all experiment tags (both static and dynamic)
        experiment_tags_dict = {
            "scenario_selection": scenario_selection,  # e.g. "s04", "s05", "s06"...
            "modeling_selection": modeling_selection,  # e.g. "fast", "paper", "full", etc.
            "experiment_type": experiment_result.sampling_selection,  # e.g. "binary_vs_ref"
            "start_time": experiment_result.start_time,
            "status": "initialized",
            "completed_datasets": "0",
            "completed_trainings": "0",
        }

        mlflow.start_run(run_name=run_name, nested=False)

        # Set initial tags
        mlflow.set_tags(experiment_tags_dict)

        self.logger.info(f"Started experiment run: '{run_name}'")

    def start_dataset_run(self, dataset_result: DatasetResult) -> None:
        """Initialize dataset run with metadata."""
        run_name = dataset_result.dataset_name
        dataset_tags = dataset_result.get_tags()

        mlflow.start_run(run_name=run_name, nested=True)
        mlflow.set_tags(dataset_tags)

        # Initialize placeholder metrics (will be updated after models complete)
        mlflow.log_metrics(
            {
                "best_f1_score": 0.0,
                "best_accuracy": 0.0,
                "avg_f1_score": 0.0,
                "avg_accuracy": 0.0,
                "models_completed": 0,
            }
        )

        self.logger.debug(f"Started dataset run: {run_name}")

    def start_model_run(self, model_result: ModelResult) -> None:
        """Initialize model run with placeholder metrics."""
        run_name = model_result.model_name

        mlflow.start_run(run_name=run_name, nested=True)

        # Initialize with placeholder metrics (will be updated after each fold)
        mlflow.log_metrics(
            {
                "f1_score": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "folds_completed": 0,
            }
        )

        # Log model metadata
        mlflow.log_params(
            {
                "model_name": model_result.model_name,
                "dataset_name": model_result.dataset_name,
            }
        )

        self.logger.debug(f"Started model run: {run_name}")

    def start_fold_run(self, fold_index: int) -> None:
        """Initialize fold run structure."""
        run_name = f"fold_{fold_index}"

        mlflow.start_run(run_name=run_name, nested=True)

        # Log fold parameters
        mlflow.log_params({"fold_index": fold_index, "fold_type": "cross_validation"})

    # ================================
    # UPDATE METHODS - Push results
    # ================================

    def update_fold_run(self, fold_result: FoldResult) -> None:
        """Update fold run with actual results after fold completion."""
        try:
            # Log fold metrics
            mlflow.log_metrics(fold_result.get_mlflow_metrics())

            # Log fold parameters
            mlflow.log_params(fold_result.get_mlflow_params())

            # Log confusion matrix as artifact
            try:
                confusion_matrix = fold_result.get_confusion_matrix()
                if confusion_matrix is not None:
                    cm_path = f"fold_{fold_result.fold_index}_cm.csv"
                    pd.DataFrame(confusion_matrix).to_csv(cm_path, index=False)
                    mlflow.log_artifact(cm_path)
                    os.remove(cm_path)  # Clean up
            except Exception as e:
                self.logger.warning(f"Failed to log fold confusion matrix: {str(e)}")

        except Exception as e:
            self.logger.warning(
                f"Failed to update fold {fold_result.fold_index}: {str(e)}"
            )

    def update_model_run(self, model_result: ModelResult) -> None:
        """Update model run with current averages after each fold completion."""
        try:
            # Get current metrics (recalculated from all completed folds)
            current_metrics = model_result.get_mlflow_metrics()

            # Update the model run with current averages
            mlflow.log_metrics(current_metrics)

            # Update fold completion count
            mlflow.log_metric("folds_completed", len(model_result.fold_results))

            # Log aggregated confusion matrix
            if model_result.confusion_matrix is not None:
                self._log_confusion_matrix(model_result)

            # Log stability analysis
            self._log_fold_stability_analysis(model_result)

        except Exception as e:
            self.logger.warning(
                f"Failed to update model {model_result.model_name}: {str(e)}"
            )

    def update_dataset_run(self, dataset_result: DatasetResult) -> None:
        """Update dataset run with aggregates across all models."""
        try:
            if not dataset_result.model_results:
                return

            # Calculate aggregates across all models
            f1_scores = [
                mr.get_mean_metric("f1_score") for mr in dataset_result.model_results
            ]
            accuracies = [
                mr.get_mean_metric("accuracy") for mr in dataset_result.model_results
            ]

            # Best performing model metrics
            best_f1 = max(f1_scores) if f1_scores else 0.0
            best_accuracy = max(accuracies) if accuracies else 0.0

            # Average across all models (to identify easy/hard datasets)
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0

            # Update dataset metrics
            mlflow.log_metrics(
                {
                    "best_f1_score": best_f1,
                    "best_accuracy": best_accuracy,
                    "avg_f1_score": avg_f1,
                    "avg_accuracy": avg_accuracy,
                    "models_completed": len(dataset_result.model_results),
                    "f1_score_std": np.std(f1_scores) if f1_scores else 0.0,
                    "accuracy_std": np.std(accuracies) if accuracies else 0.0,
                }
            )

            # Tag dataset difficulty based on average performance
            difficulty = (
                "easy" if avg_f1 > 0.8 else "medium" if avg_f1 > 0.6 else "hard"
            )
            mlflow.set_tag("dataset_difficulty", difficulty)

        except Exception as e:
            self.logger.warning(
                f"Failed to update dataset {dataset_result.dataset_name}: {str(e)}"
            )

    def update_experiment_run(self, experiment_result: ExperimentResult) -> None:
        """Update experiment run with progress counters."""
        try:
            # Count completed datasets and total trained models
            completed_datasets = len(experiment_result.dataset_results)
            total_trained_models = sum(
                len(dr.model_results) for dr in experiment_result.dataset_results
            )

            # Update progress tags
            mlflow.set_tags(
                {
                    "completed_datasets": str(completed_datasets),
                    "trained_models": str(total_trained_models),
                    "status": "running",  # Will be set to "completed" when experiment finishes
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to update experiment progress: {str(e)}")

    def finalize_experiment_run(self, experiment_result: ExperimentResult) -> None:
        """Finalize experiment run with completion status."""
        try:
            # Update final tags
            mlflow.set_tags(
                {
                    "status": "completed",
                    "finish_time": experiment_result.finish_time,
                    "total_datasets": str(len(experiment_result.dataset_results)),
                    "total_trained_models": str(
                        sum(
                            len(dr.model_results)
                            for dr in experiment_result.dataset_results
                        )
                    ),
                }
            )

            self.logger.info("Finalized experiment run")

        except Exception as e:
            self.logger.warning(f"Failed to finalize experiment: {str(e)}")

    # ================================
    # HELPER METHODS
    # ================================

    def _log_fold_stability_analysis(self, model_result: ModelResult) -> None:
        """Log fold stability analysis as metrics and tags."""
        try:
            for metric_name in ["f1_score", "accuracy", "precision", "recall"]:
                if (
                    hasattr(model_result, "mean_metrics")
                    and metric_name in model_result.mean_metrics
                ):
                    stability = model_result.get_fold_stability(metric_name)
                    mlflow.set_tag(f"{metric_name}_stability", stability)

                    # Log coefficient of variation
                    summary = model_result.get_metric_summary(metric_name)
                    mlflow.log_metric(f"{metric_name}_cv", summary["cv"])

        except Exception as e:
            self.logger.warning(f"Failed to log stability analysis: {str(e)}")

    def _log_confusion_matrix(self, model_result: ModelResult) -> None:
        """Log aggregated confusion matrix as MLflow artifact."""
        cm_path = f"{model_result.model_name}_aggregated_cm.csv"
        try:
            pd.DataFrame(model_result.confusion_matrix).to_csv(cm_path, index=False)
            mlflow.log_artifact(cm_path)
        except Exception as e:
            self.logger.warning(f"Failed to log confusion matrix: {str(e)}")
        finally:
            if os.path.exists(cm_path):
                os.remove(cm_path)

    def end_run(self) -> None:
        """End current MLflow run safely."""
        try:
            mlflow.end_run()
        except Exception as e:
            self.logger.warning(f"Error ending MLflow run: {str(e)}")
