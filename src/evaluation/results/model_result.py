from typing import Any, Dict, List

import numpy as np

from .fold_result import FoldResult


class ModelResult:
    """Results from evaluating a model across all cross-validation folds."""

    def __init__(self, model_name: str, dataset_name: str):
        """Initialize model result to hold multiple fold results."""
        # Set names for model and dataset
        self.model_name = model_name
        self.dataset_name = dataset_name

        # Initialize an empty list to hold all fold results
        self.fold_results: List[FoldResult] = []

        # Initialize empty metrics dictionaries (will be computed when folds are added)
        self.mean_metrics = {}
        self.std_metrics = {}
        self.min_metrics = {}
        self.max_metrics = {}
        self.confusion_matrix = None

    def add_result(self, fold_result: FoldResult) -> None:
        """Add a fold result and recompute aggregated metrics."""
        if not isinstance(fold_result, FoldResult):
            raise TypeError("fold_result must be a FoldResult instance")

        self.fold_results.append(fold_result)

        # Recompute aggregated metrics after each fold addition
        # This enables real-time updates in MLflow
        self._compute_aggregated_metrics()
        self._compute_aggregated_confusion_matrix()

    def _compute_aggregated_metrics(self):
        """Compute mean, std, min, max across all folds."""
        if not self.fold_results:
            self.mean_metrics = {}
            self.std_metrics = {}
            self.min_metrics = {}
            self.max_metrics = {}
            return

        # Get all metric names from first fold
        metric_names = self.fold_results[0].metrics.keys()

        self.mean_metrics = {}
        self.std_metrics = {}
        self.min_metrics = {}
        self.max_metrics = {}

        for metric_name in metric_names:
            values = [fold.metrics.get(metric_name, 0) for fold in self.fold_results]
            self.mean_metrics[metric_name] = np.mean(values)
            self.std_metrics[metric_name] = np.std(values)
            self.min_metrics[metric_name] = np.min(values)
            self.max_metrics[metric_name] = np.max(values)

    def _compute_aggregated_confusion_matrix(self):
        """Compute confusion matrix aggregated across all folds."""
        if not self.fold_results:
            self.confusion_matrix = None
            return

        # Concatenate all predictions and true labels
        all_y_true = np.concatenate([fold.y_true for fold in self.fold_results])
        all_y_pred = np.concatenate([fold.y_pred for fold in self.fold_results])

        from sklearn.metrics import confusion_matrix

        self.confusion_matrix = confusion_matrix(all_y_true, all_y_pred)

    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary for a specific metric."""
        return {
            "mean": self.mean_metrics.get(metric_name, 0.0),
            "std": self.std_metrics.get(metric_name, 0.0),
            "min": self.min_metrics.get(metric_name, 0.0),
            "max": self.max_metrics.get(metric_name, 0.0),
            "cv": self.std_metrics.get(metric_name, 0.0)
            / max(
                self.mean_metrics.get(metric_name, 1.0), 1e-10
            ),  # Coefficient of variation
        }

    def get_mean_metric(self, metric_name: str) -> float:
        """Get mean value for a specific metric."""
        return self.mean_metrics.get(metric_name, 0.0)

    def get_fold_stability(self, metric_name: str) -> str:
        """
        Assess stability of a metric across folds.

        Returns:
        --------
        str: "stable", "moderate", or "unstable"
        """
        if not self.fold_results or metric_name not in self.std_metrics:
            return "unknown"

        mean_val = self.mean_metrics.get(metric_name, 0.0)
        std_val = self.std_metrics.get(metric_name, 0.0)

        # Use coefficient of variation to assess stability
        cv = std_val / max(mean_val, 1e-10)

        if cv < 0.1:
            return "stable"
        elif cv < 0.2:
            return "moderate"
        else:
            return "unstable"

    def get_mlflow_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics suitable for MLflow logging."""
        metrics = {}

        # Add mean metrics (primary metrics)
        for metric_name, value in self.mean_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics[metric_name] = float(value)

        # Add statistical measures with suffix
        for metric_name, value in self.std_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics[f"{metric_name}_std"] = float(value)

        for metric_name, value in self.min_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics[f"{metric_name}_min"] = float(value)

        for metric_name, value in self.max_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics[f"{metric_name}_max"] = float(value)

        # Add timing information if available
        if self.fold_results:
            train_times = [fold.training_time for fold in self.fold_results]
            pred_times = [fold.prediction_time for fold in self.fold_results]

            metrics.update(
                {
                    "avg_training_time": np.mean(train_times),
                    "avg_prediction_time": np.mean(pred_times),
                    "total_training_time": np.sum(train_times),
                    "total_prediction_time": np.sum(pred_times),
                }
            )

        return metrics

    def __repr__(self) -> str:
        f1 = self.get_mean_metric("f1_score")
        f1_std = self.std_metrics.get("f1_score", 0)
        return f"ModelResult({self.model_name}, f1={f1:.2f}±{f1_std:.2f}, {len(self.fold_results)} folds)"
