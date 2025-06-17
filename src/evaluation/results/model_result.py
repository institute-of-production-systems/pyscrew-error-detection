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
                # Add dataset metadata
                "n_samples": str(self.n_samples),
                "n_classes": str(self.n_classes),
                "samples_per_class": self.samples_per_class,
                # Add model metadata
                "model_name": self.name,
                "model_family": self.model_family,
                "model_library": self.model_library,
            }
        )

        return tags

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics. Empty if no folds, single fold as-is, multiple folds averaged."""

        # No folds completed
        if not self.fold_results:
            return {}

        # Single fold - return as-is with std=0
        if len(self.fold_results) == 1:
            metrics = self.fold_results[0].metrics.copy()
            # Add std=0 for all metrics
            for key in list(metrics.keys()):
                metrics[f"{key}_std"] = 0.0
            return metrics

        # Multiple folds - calculate mean and std
        metrics = {}
        metric_names = self.fold_results[0].metrics.keys()

        for metric_name in metric_names:
            values = [fold.metrics[metric_name] for fold in self.fold_results]
            metrics[metric_name] = np.mean(values)
            metrics[f"{metric_name}_std"] = np.std(values)

        return metrics

    def __repr__(self) -> str:
        if not self.fold_results:
            return f"ModelResult({self.name}, no folds yet)"

        metrics = self.get_metrics()
        f1_key = "f1_score" if len(self.fold_results) == 1 else "f1_score_avg"
        f1 = metrics.get(f1_key, 0)

        if len(self.fold_results) == 1:
            return f"ModelResult({self.name}, f1={f1:.2f}, 1 fold)"
        else:
            f1_std = metrics.get("f1_score_std", 0)
            return f"ModelResult({self.name}, f1={f1:.2f}±{f1_std:.2f}, {len(self.fold_results)} folds)"
