from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from sktime.classification.base import BaseClassifier

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

from ..utils.exceptions import (
    DatasetPreparationError,
    FatalExperimentError,
    ModelEvaluationError,
)
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
        # Core experimental design choices (the crucial trio)
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
        self.mlflow_manager = MLflowManager(port=mlflow_port)

        # Initialize experiment result class to track all results
        self.experiment_result = ExperimentResult(
            scenario_selection=self.scenario_selection,
            sampling_selection=self.sampling_selection,
            modeling_selection=self.modeling_selection,
        )

        # Instance variables for stateful design
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
        self.mlflow_manager.setup_tracking(self.sampling_selection)

    def _setup_datasets_and_models(self) -> None:
        """Load data, apply preprocessing, and initialize models."""
        try:
            self.logger.info("Loading and preprocessing data...")

            # Load and preprocess time series data using PyScrew
            raw_data = load_data(
                scenario_selection=self.scenario_selection,
                target_length=self.target_length,
                screw_positions=self.screw_positions,
            )
            processed_data = process_data(raw_data, target_length=200)

            # Generate datasets based on sampling selection
            self.datasets = sample_datasets(
                processed_data=processed_data,
                sampling_selection=self.sampling_selection,
                scenario_selection=self.scenario_selection,
            )

            # Initialize ML models with experiment configuration
            self.models = get_classifier_dict(
                self.modeling_selection,
                self.random_seed,
                self.n_jobs,
            )

            self.logger.info(
                f"Prepared {len(self.datasets)} datasets and {len(self.models)} models"
            )

        except Exception as e:
            raise DatasetPreparationError(f"Failed to prepare data: {str(e)}") from e

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
            cv = KFold()
            split_method = lambda: cv.split(dataset.x_values)
            self.logger.debug(f"Using regular KFold with {self.cv_folds} splits")

        self.split_method = split_method

    def _run_experiment(self) -> None:
        """Execute main experiment loop with clean start/update MLflow pattern."""

        # START: Initialize main experiment run
        self.mlflow_manager.start_experiment_run(self.experiment_result)

        try:
            # Process each dataset with all models
            for dataset_idx, dataset in enumerate(self.datasets):
                progress = f"{dataset_idx + 1}/{len(self.datasets)}"
                self.logger.info(f"Processing dataset {progress}: {dataset.name}")

                dataset_result = self._run_dataset(dataset)
                self.experiment_result.add_result(dataset_result)

                # UPDATE: Update experiment progress after each dataset
                self.mlflow_manager.update_experiment_run(self.experiment_result)

            # END: Mark experiment as complete
            self.experiment_result.end()
            self.mlflow_manager.end_experiment_run(self.experiment_result)

        finally:
            self.mlflow_manager.end_run()

    def _run_dataset(self, experiment_dataset: ExperimentDataset) -> DatasetResult:
        """Process all models on a single dataset."""

        # Initialize dataset result
        dataset_result = DatasetResult(
            dataset_name=experiment_dataset.name,
            dataset_tags={
                key: value
                for key, value in experiment_dataset.to_dict().items()
                if key not in ["name", "x_values", "y_values"]
            },
        )

        # START: Initialize dataset run
        self.mlflow_manager.start_dataset_run(dataset_result)

        # Update the split method for the current dataset
        self._update_split_method(experiment_dataset)

        try:
            # Process all models for this dataset
            for model_idx, model_name in enumerate(self.models):
                progress = f"{model_idx + 1}/{len(self.models)}"
                self.logger.info(f"  Applying model {progress}: {model_name}")

                try:
                    model_result = self._run_model(experiment_dataset, model_name)
                    dataset_result.add_result(model_result)

                    # Log success
                    f1_score = model_result.get_mean_metric("f1_score")
                    self.logger.info(
                        f"    {model_name}: f1_score (avg.) = {f1_score:.3f}"
                    )

                except ModelEvaluationError as e:
                    self.logger.warning(f"    {model_name}: FAILED - {str(e)}")
                    # Continue with next model

            # UPDATE: Update dataset aggregates after all models complete
            self.mlflow_manager.update_dataset_run(dataset_result)

        finally:
            self.mlflow_manager.end_run()

        return dataset_result

    def _run_model(self, dataset: ExperimentDataset, model_name: str) -> ModelResult:
        """Process all folds of a model for a given dataset."""

        # Initialize model result
        model_result = ModelResult(
            model_name=model_name,
            dataset_name=dataset.name,
        )

        # START: Initialize model run
        self.mlflow_manager.start_model_run(model_result)

        # Get data and model
        x_values = dataset.x_values
        y_values = dataset.y_values
        model = self.models[model_name]

        try:
            # Log fold processing start
            self.logger.info(f"    Running {self.cv_folds} folds")

            # Process all folds for this model
            fold_index = 0
            for train_idx, test_idx in self.split_method():
                x_values_split = x_values[train_idx], x_values[test_idx]
                y_values_split = y_values[train_idx], y_values[test_idx]

                # Process single fold
                fold_result = self._run_fold(
                    model=model,
                    x_values_split=x_values_split,
                    y_values_split=y_values_split,
                    fold_index=fold_index,
                )

                # Only add successful fold results (skip failed folds as requested)
                if "error" not in fold_result.metadata:
                    model_result.add_result(fold_result)

                    # UPDATE: Update model averages after each successful fold
                    model_result._compute_aggregated_metrics()
                    model_result._compute_aggregated_confusion_matrix()
                    self.mlflow_manager.update_model_run(model_result)

                fold_index += 1

            # Check if we have any successful folds
            if not model_result.fold_results:
                raise ModelEvaluationError("All folds failed for this model")

        except Exception as e:
            self.logger.error(f"Error running model {model_name}: {str(e)}")
            raise ModelEvaluationError(
                f"Failed to evaluate {model_name}: {str(e)}"
            ) from e

        finally:
            self.mlflow_manager.end_run()

        return model_result

    def _run_fold(
        self,
        model: Union[BaseEstimator, BaseClassifier],
        x_values_split: Tuple[np.ndarray, np.ndarray],
        y_values_split: Tuple[np.ndarray, np.ndarray],
        fold_index: int,
    ) -> FoldResult:
        """Process a single cross-validation fold."""

        # START: Initialize fold run
        self.mlflow_manager.start_fold_run(fold_index)

        try:
            # Delegate actual training to training module
            from .training import train_single_fold

            x_train, x_test = x_values_split
            y_train, y_test = y_values_split

            fold_result = train_single_fold(
                model=model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                fold_index=fold_index,
            )

            # UPDATE: Update fold run with results
            self.mlflow_manager.update_fold_run(fold_result)

            # Log progress with consistent indentation
            f1_score = fold_result.metrics.get("f1_score", 0)
            self.logger.info(f"      Fold {fold_index}: f1_score = {f1_score:.3f}")

            return fold_result

        finally:
            self.mlflow_manager.end_run()

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
            self._setup_datasets_and_models()
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
