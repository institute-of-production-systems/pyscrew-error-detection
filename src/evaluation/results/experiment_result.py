from typing import List

import pandas as pd

from .base_result import BaseResult
from .dataset_result import DatasetResult


class ExperimentResult(BaseResult):
    """
    Results from running a complete experiment across multiple datasets.

    Container for all dataset results in an experiment. Inherits static configuration
    from BaseResult for MLflow tagging and cross-experiment queries.

    This is the top level of the 4-level hierarchy:
    ExperimentResult -> DatasetResult -> ModelResult -> FoldResult
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
            Dataset scenario (e.g., "s04", "s05", "s06")
        sampling_selection : str
            Sampling strategy (e.g., "binary_vs_ref", "multiclass_with_groups")
        modeling_selection : str
            Model set selection (e.g., "debug", "fast", "paper", "full")
        """
        # Call parent constructor with the core configuration
        super().__init__(scenario_selection, sampling_selection, modeling_selection)

        # Initialize an empty list to hold all dataset results
        self.dataset_results: List[DatasetResult] = []

    def add_result(self, dataset_result: DatasetResult) -> None:
        """
        Add a dataset result to this experiment.

        Parameters:
        -----------
        dataset_result : DatasetResult
            Completed dataset result containing model evaluations

        Raises:
        -------
        TypeError
            If dataset_result is not a DatasetResult instance
        """
        if not isinstance(dataset_result, DatasetResult):
            raise TypeError("'dataset_result' must be a DatasetResult instance")
        self.dataset_results.append(dataset_result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert hierarchical results to flat DataFrame for analysis."""
        rows = []
        for dataset_result in self.dataset_results:
            for model_result in dataset_result.model_results:
                for fold_result in model_result.fold_results:
                    row = {
                        # 1: Experiment level
                        "experiment": self.sampling_selection,
                        "scenario": self.scenario_selection,
                        "modeling": self.modeling_selection,
                        # 2: Dataset level
                        "dataset": dataset_result.name,
                        "n_samples": dataset_result.n_samples,
                        "n_classes": dataset_result.n_classes,
                        # 3: Model level
                        "model": model_result.name,
                        "model_family": model_result.model_family,
                        # 4: Fold level
                        "fold": fold_result.fold_index,
                        **fold_result.metrics,  # All metrics
                        # F: Failure tags
                        "failed_experiment": self.failed,
                        "failed_dataset": dataset_result.failed,
                        "failed_model": model_result.failed,
                        "failed_fold": fold_result.failed,
                    }
                    rows.append(row)
        return pd.DataFrame(rows)

    def get_failure_summary(self) -> dict:
        """Get summary of failures across the experiment."""
        total_datasets = len(self.dataset_results)
        failed_datasets = sum(1 for dr in self.dataset_results if dr.failed)

        total_models = sum(len(dr.model_results) for dr in self.dataset_results)
        failed_models = sum(
            sum(1 for mr in dr.model_results if mr.failed)
            for dr in self.dataset_results
        )

        total_folds = sum(
            sum(len(mr.fold_results) for mr in dr.model_results)
            for dr in self.dataset_results
        )
        failed_folds = sum(
            sum(
                sum(1 for fr in mr.fold_results if fr.failed) for mr in dr.model_results
            )
            for dr in self.dataset_results
        )

        return {
            "experiment_failed": self.failed,
            "datasets": {"total": total_datasets, "failed": failed_datasets},
            "models": {"total": total_models, "failed": failed_models},
            "folds": {"total": total_folds, "failed": failed_folds},
        }

    def end(self) -> None:
        """Mark pipeline complete, calculate runtime, and log failure summary."""
        # Call parent's end() method to handle timing
        super().end()

        # Get logger for this module
        from src.utils import get_logger

        logger = get_logger(__name__)

        # Log completion with failure summary
        summary = self.get_failure_summary()

        # Save to csv (TODO: with with proper naming)
        if True:
            df_experiment_result = self.to_dataframe()
            file_name = "results/experiment_result.csv"
            df_experiment_result.to_csv(file_name)

        if summary["experiment_failed"]:
            logger.error(f"Experiment FAILED: {self.failure_reason}")
        elif (
            summary["datasets"]["failed"] > 0
            or summary["models"]["failed"] > 0
            or summary["folds"]["failed"] > 0
        ):
            logger.warning(f"Experiment completed with failures:")
            logger.warning(
                f"  Datasets: {summary['datasets']['failed']}/{summary['datasets']['total']} failed"
            )
            logger.warning(
                f"  Models: {summary['models']['failed']}/{summary['models']['total']} failed"
            )
            logger.warning(
                f"  Folds: {summary['folds']['failed']}/{summary['folds']['total']} failed"
            )
        else:
            logger.info(f"Experiment completed successfully:")
            logger.info(
                f"  {summary['datasets']['total']} datasets, {summary['models']['total']} models, {summary['folds']['total']} folds - all successful!"
            )

    def __repr__(self) -> str:
        """String representation showing experiment status and failure info."""
        status = "completed" if self.end_time else "running"

        # Add failure info if there are any failures
        failure_info = ""
        if self.failed:
            status = "failed"

        elif any(dr.failed for dr in self.dataset_results):
            failed_count = sum(1 for dr in self.dataset_results if dr.failed)
            failure_info = f", {failed_count} datasets failed"

        return f"ExperimentResult({self.sampling_selection}, {len(self.dataset_results)} datasets, {status}{failure_info})"
