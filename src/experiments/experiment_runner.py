from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from src.data import load_data, process_data
from src.evaluation.results import (
    DatasetResult,
    ExperimentResult,
    FoldResult,
    ModelResult,
)
from src.mlflow import MLflowManager, launch_server
from src.models import get_classifier_dict
from src.sampling import sample_datasets
from src.utils import get_logger

from ..utils.exceptions import DatasetPreparationError, FatalExperimentError
from .experiment_dataset import ExperimentDataset


class ExperimentRunner:
    """
    Orchestrates ML experiments on time series data with MLflow tracking.

    Supports binary/multiclass classification with cross-validation and
    4-level hierarchical result tracking (Experiment -> Dataset -> Model -> Fold).
    Handles partial failures gracefully and provides comprehensive logging.
    """

    # Class variables to track MLflow server state
    _mlflow_server_running = False
    _MLFLOW_PORT = 5000

    def __init__(
        self,
        scenario_selection: str,
        sampling_selection: str,
        modeling_selection: str,
        # Technical configuration from config.yml (unpacked)
        target_length: int = 2000,
        screw_positions: str = "left",
        cv_folds: int = 5,
        stratify: bool = True,
        random_seed: int = 42,
        n_jobs: int = -1,
        log_level: str = "INFO",
        mlflow_port: int = 5000,
        # Accept any additional config parameters
        # TODO: holds a few not-implemented parameters at the moment
        **kwargs,
    ):
        """Initialize experiment runner with configuration parameters."""
        # Core experimental design choices (the 'crucial trio')
        self.scenario_selection = scenario_selection  # e.g. "s06"
        self.sampling_selection = sampling_selection  # e.g. "binary_for_extremes"
        self.modeling_selection = modeling_selection  # e.g. "paper"

        # Technical/modeling parameters (from config.yml)
        self.target_length = target_length
        self.screw_positions = screw_positions
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.stratify = stratify

        # Initialize logging
        self.logger = get_logger(__name__, log_level)

        # Initialize MLflow manager
        self.manager = MLflowManager(port=mlflow_port)

        # Instance variables for stateful design
        self.experiment_result: ExperimentResult = None
        self.datasets: List[ExperimentDataset] = []
        self.models: Dict[str, Any] = {}

        self._log_initialization()

    @classmethod
    def _ensure_mlflow_server(cls) -> None:
        """Start MLflow server if not already running."""
        if cls._mlflow_server_running:
            return

        try:
            launch_server(port=cls._MLFLOW_PORT)
            cls._mlflow_server_running = True
        except Exception as e:
            raise FatalExperimentError(
                f"Could not start MLflow server: {str(e)}. "
                "Please start the MLflow server manually or check the configuration."
            ) from e

    def _log_initialization(self) -> None:
        """Log experiment configuration for debugging."""
        self.logger.info(f"Initializing {self.sampling_selection} experiment")
        self.logger.info(
            f"  Dataset: {self.scenario_selection}, Models: {self.modeling_selection}"
        )
        self.logger.info(
            f"  CV folds: {self.cv_folds}, Random seed: {self.random_seed}"
        )
        self.logger.info(f"  Stratify: {self.stratify}, n_jobs: {self.n_jobs}")

    def _setup_experiment_tracking(self) -> None:
        """Set up MLflow server and tracking."""
        self.logger.info("Setting up experiment environment...")

        # Ensure MLflow server is running
        self._ensure_mlflow_server()

        # Setup MLflow tracking
        self.manager.setup_tracking(self.scenario_selection)

    def get_all_selections(self) -> Tuple[str, str, str]:
        """Helper function to return a tuple of the core trio."""
        return (
            self.scenario_selection,
            self.sampling_selection,
            self.modeling_selection,
        )

    def get_all_datasets(self) -> None:
        """Load data and apply preprocessing."""
        try:
            self.logger.info("Loading and preprocessing data...")

            # Load and preprocess time series data using PyScrew
            raw_data = load_data(
                scenario_selection=self.scenario_selection,
                target_length=self.target_length,
                screw_positions=self.screw_positions,
            )
            # Generate datasets based on sampling selection
            processed_data = process_data(raw_data, target_length=200)
            all_selections = self.get_all_selections()
            return sample_datasets(processed_data, *all_selections)

        except Exception as e:
            raise DatasetPreparationError(f"Failed to prepare data: {str(e)}") from e

    def get_all_models(self) -> None:
        """Initialize ML models with experiment configuration."""
        return get_classifier_dict(
            self.modeling_selection,
            self.random_seed,
            self.n_jobs,
        )

    def _update_split_method(self, dataset: ExperimentDataset):
        """Set up cross-validation with appropriate split strategy."""
        split_params = {
            "n_splits": self.cv_folds,
            "shuffle": True,
            "random_state": self.random_seed,
        }
        if self.stratify:
            cv = StratifiedKFold(**split_params)
            split_method = lambda: cv.split(dataset.x_values, dataset.y_values)
            self.logger.debug(f"Using StratifiedKFold with {self.cv_folds} splits")
        else:
            cv = KFold(**split_params)
            split_method = lambda: cv.split(dataset.x_values)
            self.logger.debug(f"Using regular KFold with {self.cv_folds} splits")

        self.split_method = split_method

    def _run_experiment(self) -> None:
        """Execute main experiment loop with clean start/update MLflow pattern."""

        # START: Initialize experiment result with inherited trio and start experiment run
        self.experiment_result = ExperimentResult(*self.get_all_selections())
        self.manager.start_experiment_run(self.experiment_result)

        try:
            # Iterate all datasets according to the sampled selection
            for dataset_idx, dataset in enumerate(self.datasets):
                # Logging (start)
                progress = f"{dataset_idx + 1}/{len(self.datasets)}"
                self.logger.info(f"Processing dataset {progress}: {dataset.name}")
                # 1. Run dataset
                dataset_result = self._run_dataset(dataset)
                # 2. Add result
                self.experiment_result.add_result(dataset_result)
                # 3. Update manager
                self.manager.update_experiment_run(self.experiment_result)
                # Logging (end)
                self.logger.info("..")

            # END: Mark experiment as complete
            self.experiment_result.end()
            self.manager.end_experiment_run(self.experiment_result)

        finally:
            self.manager.end_run()

    def _run_dataset(self, experiment_dataset: ExperimentDataset) -> DatasetResult:
        """Process all models on a single dataset.

        Takes a single DatasetResult, iteartes all models and adds a ModelResult to
        the Dataset Result class for each model, returns the Dataset Result.
        """

        # START: Initialize dataset result with inherited trio and start dataset run
        dataset_result = DatasetResult(experiment_dataset)
        self.manager.start_dataset_run(dataset_result)

        # Update the split method for the current dataset (random or stratified split)
        self._update_split_method(experiment_dataset)

        try:
            # Iterate all models for the current dataset
            for model_idx, model_name in enumerate(self.models):
                # Logging (start)
                progress = f"({model_idx + 1}/{len(self.models)})"  # e.g. "(2/5)"
                self.logger.info(f"- Applying model {progress}: {model_name}")
                # 1. Run model
                model_result = self._run_model(experiment_dataset, model_name)
                # 2. Add result
                dataset_result.add_result(model_result)
                # Update manager
                self.manager.update_dataset_run(dataset_result)
                # 3. Logging (end)
                f1_score = model_result.get_avg_metrics().get("avg_f1_score", 0.0)
                self.logger.info(f"{model_name}: f1_score (avg.) = {f1_score:.3f}")
        finally:
            self.manager.end_run()

        return dataset_result

    def _run_model(
        self, experiment_dataset: ExperimentDataset, model_name: str
    ) -> ModelResult:
        """Process all folds of a model for a given dataset."""

        # START: Initialize model result with inherited trio and start model run
        model_result = ModelResult(experiment_dataset, model_name)
        self.manager.start_model_run(model_result)

        try:
            # Generate splits once for all folds
            splits = list(self.split_method())

            # Iterate all folds for the selected number of cv_folds
            for fold_index in range(self.cv_folds):
                # 1. Run fold
                fold_result = self._run_fold(
                    experiment_dataset, model_name, fold_index, splits[fold_index]
                )
                # 2. Add result
                model_result.add_result(fold_result)
                # 3. Update manager
                self.manager.update_model_run(model_result)

        finally:
            self.manager.end_run()

        return model_result

    def _run_fold(
        self,
        experiment_dataset: ExperimentDataset,
        model_name: str,
        fold_index: int,
        split_indices: Tuple[np.ndarray, np.ndarray],  # (train_idx, test_idx)
    ) -> FoldResult:
        """Process a single cross-validation fold."""

        # START: Initialize fold run
        fold_result = FoldResult(experiment_dataset, model_name, fold_index)
        self.manager.start_fold_run(fold_result)

        try:
            # 1. Train and evaluate fold using the provided split indices
            train_idx, test_idx = split_indices
            fold_result.train_and_evaluate(self.models[model_name], train_idx, test_idx)

            # 2. Update manager
            self.manager.update_fold_run(fold_result)

            # 3. Log progress
            f1_score = fold_result.metrics.get("f1_score", 0)
            self.logger.info(f"      Fold {fold_index}: f1_score = {f1_score:.3f}")

        finally:
            self.manager.end_run()

        return fold_result

    def _evaluate_experiment(self) -> None:
        """Generate experiment summary and log best performers."""
        # Log experiment summary
        self.logger.info("Evaluating experiment results...")
        self.logger.info(f"Experiment completed:")
        self.logger.info(
            f"Total datasets: {len(self.experiment_result.dataset_results)}"
        )

    def run(self) -> ExperimentResult:
        """
        Execute complete experiment workflow.

        Returns hierarchical results with 4-level analysis:
        Experiment -> Dataset -> Model -> Fold
        """
        try:
            self.datasets = self.get_all_datasets()
            self.models = self.get_all_models()

            self._setup_experiment_tracking()

            # Execute experiment with real-time logging
            self._run_experiment()

            # Generate final summary
            self._evaluate_experiment()

            return self.experiment_result

        except FatalExperimentError:
            raise  # Re-raise fatal errors
        except Exception as e:

            raise FatalExperimentError(
                f"Unexpected error in experiment: {str(e)}"
            ) from e  # Wrap unexpected errors as fatal
