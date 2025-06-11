from typing import Any, Dict, List

from .base_result import BaseResult
from .model_result import ModelResult


class DatasetResult(BaseResult):
    """
    Results from evaluating all models on a single dataset.

    Container for all model results on a dataset. Inherits static configuration
    from BaseResult for MLflow tagging and cross-experiment queries.
    """

    def __init__(
        self,
        scenario_selection: str,
        sampling_selection: str,
        modeling_selection: str,
        dataset_name: str,
        dataset_tags: Dict[str, Any],
    ) -> None:
        """
        Initialize dataset result with core configuration.

        Parameters:
        -----------
        scenario_selection : str
            Dataset scenario (e.g., "s06")
        sampling_selection : str
            Sampling strategy (e.g., "binary_for_extremes")
        modeling_selection : str
            Model selection (e.g., "paper")
        dataset_name : str
            Name of the dataset (e.g., "group_1_cooling_time")
        dataset_tags : Dict[str, Any]
            Dataset-specific metadata tags
        """
        # Call parent constructor with the core configuration
        super().__init__(scenario_selection, sampling_selection, modeling_selection)
        # Dataset-specific properties
        self.name = dataset_name
        self.tags = dataset_tags
        # Initialize an empty list to hold all model results
        self.model_results: List[ModelResult] = []

    def add_result(self, model_result: ModelResult) -> None:
        """Add a model result to this dataset."""
        if not isinstance(model_result, ModelResult):
            raise TypeError("'model_result' must be a ModelResult instance")
        self.model_results.append(model_result)

    def get_result_tags(self) -> Dict[str, str]:
        """
        Get static tags that inherit down MLflow hierarchy for cross-experiment queries.
        Enables SQL queries like: WHERE tags.run_name LIKE 'run_s06_%'
        """
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

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all models for this dataset."""
        # TODO: Implement aggregation of model results...
        pass

    def __repr__(self) -> str:
        """String representation showing dataset name and model count."""
        status = "completed" if self.end_time else "running"
        return f"DatasetResult({self.name}, {len(self.model_results)} models, {status})"
