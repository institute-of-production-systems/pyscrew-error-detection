from typing import Any, Dict, List

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset

from .base_result import BaseResult
from .fold_result import FoldResult


class ModelResult(BaseResult):
    """
    Results from evaluating a model across all cross-validation folds.

    Container for all fold results in an experiment. Inherits static configuration
    from BaseResult for MLflow tagging and cross-experiment queries.
    """

    def __init__(self, experiment_dataset: ExperimentDataset, model_name: str) -> None:
        """Initialize model result to hold multiple fold results."""

        # Call parent constructor with the core configuration
        super().__init__(
            experiment_dataset.scenario_selection,
            experiment_dataset.sampling_selection,
            experiment_dataset.modeling_selection,
        )
        self.name = model_name
        self.tags = experiment_dataset.get_tags()

        # Initialize an empty list to hold all fold results
        self.fold_results: List[FoldResult] = []

    def add_result(self, fold_result: FoldResult) -> None:
        """Add a fold result and recompute aggregated metrics."""
        if not isinstance(fold_result, FoldResult):
            raise TypeError("fold_result must be a FoldResult instance")
        self.fold_results.append(fold_result)

    def get_result_tags(self):
        return {
            "run_name": self.run_name,
            "scenario_selection": self.scenario_selection,
            "sampling_selection": self.sampling_selection,
            "modeling_selection": self.modeling_selection,
            "start_time": self.get_start_time(),
            "end_time": self.get_end_time(),
            "run_time": self.get_run_time(),
            **self.tags,
        }

    def get_avg_metrics(self):
        if not self.fold_results:
            return {}

        if len(self.fold_results) == 1:
            return self.fold_results[0].metrics

        # Calculate averages across all folds
        all_metrics = {}
        metric_names = self.fold_results[0].metrics.keys()

        for metric_name in metric_names:
            values = [fold.metrics.get(metric_name, 0) for fold in self.fold_results]
            all_metrics[f"avg_{metric_name}"] = np.mean(values)
            all_metrics[f"std_{metric_name}"] = np.std(values)
            # all_metrics[f"min_{metric_name}"] = np.min(values)
            # all_metrics[f"max_{metric_name}"] = np.max(values)

        # Add timing info
        train_times = [fold.training_time for fold in self.fold_results]
        pred_times = [fold.prediction_time for fold in self.fold_results]
        all_metrics["avg_training_time"] = np.mean(train_times)
        all_metrics["avg_prediction_time"] = np.mean(pred_times)

        return all_metrics

    def __repr__(self) -> str:
        if not self.fold_results:
            return f"ModelResult({self.name}, no folds yet)"

        metrics = self.get_avg_metrics()
        f1_key = "f1_score" if len(self.fold_results) == 1 else "f1_score_avg"
        f1 = metrics.get(f1_key, 0)

        if len(self.fold_results) == 1:
            return f"ModelResult({self.name}, f1={f1:.2f}, 1 fold)"
        else:
            f1_std = metrics.get("f1_score_std", 0)
            return f"ModelResult({self.name}, f1={f1:.2f}±{f1_std:.2f}, {len(self.fold_results)} folds)"
