from typing import Dict, List

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset
from src.utils.exceptions import ModelEvaluationError

from .base_result import BaseResult
from .model_result import ModelResult


class DatasetResult(BaseResult):
    """
    Results from evaluating all models on a single dataset.

    Container for all model results on a dataset. Inherits static configuration
    from BaseResult for MLflow tagging and cross-experiment queries.
    """

    def __init__(self, experiment_dataset: ExperimentDataset) -> None:
        """
        Initialize dataset result from an ExperimentDataset.

        Parameters:
        -----------
        experiment_dataset : ExperimentDataset
            Dataset containing:
            - Core configuration (scenario, sampling, modeling selections)
            - Data arrays (x_values, y_values)
            - Metadata (class_count, class_names, description)

        Computes and stores:
        - n_samples: Total number of samples
        - n_classes: Number of unique classes
        - samples_per_class: Comma-separated string of class distributions
        """
        # Call parent constructor with the core configuration
        super().__init__(
            experiment_dataset.scenario_selection,
            experiment_dataset.sampling_selection,
            experiment_dataset.modeling_selection,
        )
        self.name = experiment_dataset.name

        # Compute and store only what we need for tags
        self.n_samples = len(experiment_dataset.x_values)
        self.n_classes = experiment_dataset.class_count

        # Calculate samples_per_class
        _, counts = np.unique(experiment_dataset.y_values, return_counts=True)
        self.samples_per_class = ":".join(map(str, counts))

        # Store only lightweight metadata for plotting/analysis
        self.class_names = experiment_dataset.class_names  # Just the dict
        self.description = experiment_dataset.description

        # Initialize empty list to hold model results
        self.model_results: List[ModelResult] = []

    def add_result(self, model_result: ModelResult) -> None:
        """Add a model result to this dataset."""
        if not isinstance(model_result, ModelResult):
            raise TypeError("'model_result' must be a ModelResult instance")
        self.model_results.append(model_result)

    def get_tags(self) -> Dict[str, str]:
        """
        Get dataset-specific tags plus inherited tags.

        Returns:
        --------
        Dict[str, str]
            Tags including:
            - Inherited: scenario, sampling, modeling, timing
            - Dataset-specific: n_samples, n_classes, samples_per_class
        """
        tags = super().get_tags()
        tags.update(
            {
                # Add dataset metadata (2: Dataset Level)
                "n_samples": self.n_samples,
                "n_classes": self.n_classes,
                "samples_per_class": self.samples_per_class,
            }
        )
        return self.format_tags(tags)

    def __repr__(self) -> str:
        """String representation showing dataset name, model count, and failure info."""
        status = "completed" if self.end_time else "running"

        failure_info = ""
        if self.failed:
            status = "failed"
        elif any(mr.failed for mr in self.model_results):
            failed_count = sum(1 for mr in self.model_results if mr.failed)
            failure_info = f", {failed_count} models failed"

        return f"DatasetResult({self.name}, {len(self.model_results)} models, {status}{failure_info})"
