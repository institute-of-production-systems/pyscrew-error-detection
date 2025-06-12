from typing import List

import pandas as pd

from .base_result import BaseResult
from .dataset_result import DatasetResult


class ExperimentResult(BaseResult):
    """
    Results from running a complete experiment across multiple datasets.

    Container for all dataset results in an experiment. Inherits static configuration
    from BaseResult for MLflow tagging and cross-experiment queries.
    """

    def __init__(
        self,
        scenario_selection: str,
        sampling_selection: str,
        modeling_selection: str,
    ) -> None:
        """
        Initialize experiment result with core configuration.

        Parameters:
        -----------
        scenario_selection : str
            Dataset scenario (e.g., "s06")
        sampling_selection : str
            Sampling strategy (e.g., "binary_for_extremes")
        modeling_selection : str
            Model selection (e.g., "paper")
        """
        # Call parent constructor with the core configuration
        super().__init__(scenario_selection, sampling_selection, modeling_selection)

        # Initialize an empty list to hold all dataset results
        self.dataset_results: List[DatasetResult] = []

    def add_result(self, dataset_result: DatasetResult) -> None:
        """Add a dataset result to this experiment."""
        if not isinstance(dataset_result, DatasetResult):
            raise TypeError("'dataset_result' must be a DatasetResult instance")
        self.dataset_results.append(dataset_result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        # TODO: Implement an iteration of all dataset, model, fold results...
        pass

    def __repr__(self) -> str:
        """String representation showing experiment status and dataset count."""
        status = "completed" if self.end_time else "running"
        return f"ExperimentResult({self.sampling_selection}, {len(self.dataset_results)} datasets, {status})"
