from typing import Dict, List

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset
from src.models import get_model_family, get_model_library

from .base_result import BaseResult
from .fold_result import FoldResult


class ModelResult(BaseResult):
    """
    Results from evaluating a model across all cross-validation folds.

    Container for all fold results for a specific model. Inherits static configuration
    from BaseResult and dataset metadata from the ExperimentDataset.
    """

    def __init__(self, experiment_dataset: ExperimentDataset, model_name: str) -> None:
        """
        Initialize model result for a specific model on a dataset.

        Parameters:
        -----------
        experiment_dataset : ExperimentDataset
            Dataset containing data and metadata
        model_name : str
            Name of the model (e.g., 'RandomForestClassifier')
        """
        # Call parent constructor with the core configuration
        super().__init__(
            experiment_dataset.scenario_selection,
            experiment_dataset.sampling_selection,
            experiment_dataset.modeling_selection,
        )
        self.name = model_name

        # Model metadata
        self.model_family = get_model_family(model_name)
        self.model_library = get_model_library(model_name)

        # Dataset metadata (propagated to fold level)
        self.n_samples = len(experiment_dataset.x_values)
        self.n_classes = experiment_dataset.class_count
        unique_classes, counts = np.unique(
            experiment_dataset.y_values, return_counts=True
        )
        self.samples_per_class = ":".join(map(str, counts))

        # Initialize empty list to hold fold results
        self.fold_results: List[FoldResult] = []

    def add_result(self, fold_result: FoldResult) -> None:
        """Add a fold result and recompute aggregated metrics."""
        if not isinstance(fold_result, FoldResult):
            raise TypeError("fold_result must be a FoldResult instance")
        self.fold_results.append(fold_result)

    def get_tags(self) -> Dict[str, str]:
        """
        Get model-specific tags plus inherited tags.

        Returns:
        --------
        Dict[str, str]
            Tags including:
            - Inherited: scenario, sampling, modeling, timing
            - Dataset: n_samples, n_classes, samples_per_class
            - Model: model_name, model_family, model_library
        """
        tags = super().get_tags()
        tags.update(
            {
                # Add dataset metadata (2: Dataset Level)
                "n_samples": self.n_samples,
                "n_classes": self.n_classes,
                "samples_per_class": self.samples_per_class,
                # Add model metadata (3: Model Level)
                "model_name": self.name,
                "model_family": self.model_family,
                "model_library": self.model_library,
            }
        )

        return self.format_tags(tags)

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from successful folds only."""

        # Filter out failed folds for metric calculation
        successful_folds = [fr for fr in self.fold_results if not fr.failed]

        # No successful folds
        if not successful_folds:
            from src.evaluation.get_metrics import get_metrics

            return get_metrics()  # Returns initialized empty metrics

        # Single successful fold - return as-is with std=0
        if len(successful_folds) == 1:
            metrics = successful_folds[0].metrics.copy()
            # Add std=0 for all metrics
            for key in list(metrics.keys()):
                metrics[f"{key}_std"] = 0.0
            return metrics

        # Multiple successful folds - calculate mean and std
        metrics = {}
        metric_names = successful_folds[0].metrics.keys()

        for metric_name in metric_names:
            values = [fold.metrics[metric_name] for fold in successful_folds]
            metrics[metric_name] = float(np.mean(values))
            metrics[f"{metric_name}_std"] = float(np.std(values))

        return metrics

    def __repr__(self) -> str:
        status = "completed" if self.end_time else "running"

        if self.failed:
            return f"ModelResult({self.name}, FAILED: {self.failure_reason}, {status})"

        if not self.fold_results:
            return f"ModelResult({self.name}, no folds, {status})"

        # Count successful vs failed folds
        successful_folds = [fr for fr in self.fold_results if not fr.failed]
        failed_count = len(self.fold_results) - len(successful_folds)

        if not successful_folds:
            return f"ModelResult({self.name}, all {len(self.fold_results)} folds failed, {status})"

        # Get metrics from successful folds only
        metrics = self.get_metrics()
        f1 = metrics.get("f1_score", 0)
        f1_std = metrics.get("f1_score_std", 0)

        fold_info = f"{len(successful_folds)} folds"
        if failed_count > 0:
            fold_info += f" ({failed_count} failed)"

        if len(successful_folds) == 1:
            return f"ModelResult({self.name}, f1={f1:.3f}, {fold_info}, {status})"
        else:
            return f"ModelResult({self.name}, f1={f1:.3f}±{f1_std:.3f}, {fold_info}, {status})"
