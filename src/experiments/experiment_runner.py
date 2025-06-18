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
from src.utils.exceptions import (
    DataPreparationError,
    DataSamplingError,
    DatasetEvaluationError,
    ExperimentEvaluationError,
    FoldEvaluationError,
    ModelEvaluationError,
)

from .experiment_dataset import ExperimentDataset


class ExperimentRunner:
    """
    Orchestrates ML experiments on time series data with comprehensive failure handling.

    This class implements a robust 4-level hierarchical experiment framework:
    Experiment -> Dataset -> Model -> Fold

    Each level handles its own failure types gracefully, with configurable behavior
    for debugging (fail_fast=True) vs production (fail_fast=False) scenarios.

    Features:
    - Comprehensive exception handling with hierarchical failure tracking
    - MLflow integration with automatic experiment tracking
    - Cross-validation with stratified or regular splitting
    - Configurable failure behavior (fail fast vs continue)
    - Rich failure analysis and CSV export capabilities
    - Memory-efficient processing with automatic cleanup

    Exception Hierarchy:
    - ExperimentEvaluationError: Critical failures that stop everything
    - DatasetEvaluationError: Dataset processing failures
    - ModelEvaluationError: Model training/evaluation failures
    - FoldEvaluationError: Individual CV fold failures
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
        fail_fast: bool = False,
        # Accept any additional config parameters
        **kwargs,
    ):
        """
        Initialize experiment runner with configuration parameters.

        Parameters:
        -----------
        scenario_selection : str
            PyScrew dataset to use (e.g., "s04", "s05", "s06")
        sampling_selection : str
            Sampling strategy (e.g., "binary_vs_ref", "multiclass_with_groups")
        modeling_selection : str
            Model set to evaluate (e.g., "debug", "fast", "paper", "full")
        target_length : int, default=2000
            Target length for PAA dimensionality reduction
        screw_positions : str, default="left"
            Screw positions to use from PyScrew data
        cv_folds : int, default=5
            Number of cross-validation folds
        stratify : bool, default=True
            Whether to use stratified cross-validation
        random_seed : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = all cores)
        log_level : str, default="INFO"
            Logging level (DEBUG, INFO, WARNING, ERROR)
        mlflow_port : int, default=5000
            Port for MLflow tracking server
        fail_fast : bool, default=False
            If True, stop on first error. If False, continue and track failures.
        **kwargs
            Additional configuration parameters
        """
        # Core experimental design choices (the 'crucial trio')
        self.scenario_selection = scenario_selection
        self.sampling_selection = sampling_selection
        self.modeling_selection = modeling_selection

        # Technical/modeling parameters
        self.target_length = target_length
        self.screw_positions = screw_positions
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.stratify = stratify
        self.fail_fast = fail_fast

        # Initialize logging and MLflow
        self.logger = get_logger(__name__, log_level)
        self.manager = MLflowManager(port=mlflow_port)

        # Instance variables for stateful design
        self.experiment_result: ExperimentResult = None
        self.datasets: List[ExperimentDataset] = []
        self.models: Dict[str, Any] = {}
        self.split_method = None  # Set per dataset

        self._log_initialization()

    @classmethod
    def _ensure_mlflow_server(cls) -> None:
        """
        Start MLflow server if not already running.

        Raises:
        -------
        ExperimentEvaluationError
            If MLflow server cannot be started
        """
        if cls._mlflow_server_running:
            return

        try:
            launch_server(port=cls._MLFLOW_PORT)
            cls._mlflow_server_running = True
        except Exception as e:
            raise ExperimentEvaluationError(
                f"Could not start MLflow server: {str(e)}. "
                "Please start the MLflow server manually or check the configuration."
            ) from e

    def _log_initialization(self) -> None:
        """Log experiment configuration for debugging and transparency."""
        self.logger.info(f"Initializing {self.sampling_selection} experiment")
        self.logger.info(
            f"  Dataset: {self.scenario_selection}, Models: {self.modeling_selection}"
        )
        self.logger.info(
            f"  CV folds: {self.cv_folds}, Random seed: {self.random_seed}"
        )
        self.logger.info(
            f"  Stratify: {self.stratify}, n_jobs: {self.n_jobs}, fail_fast: {self.fail_fast}"
        )

    def _setup_experiment_tracking(self) -> None:
        """
        Set up MLflow server and tracking infrastructure.

        Raises:
        -------
        ExperimentEvaluationError
            If MLflow setup fails
        """
        self.logger.info("Setting up experiment environment...")

        # Ensure MLflow server is running
        self._ensure_mlflow_server()

        # Setup MLflow tracking
        self.manager.setup_tracking(self.scenario_selection)

    def get_all_selections(self) -> Tuple[str, str, str]:
        """Get the three core experimental design choices as a tuple."""
        return (
            self.scenario_selection,
            self.sampling_selection,
            self.modeling_selection,
        )

    def get_all_datasets(self) -> List[ExperimentDataset]:
        """
        Load data and generate experiment datasets.

        Returns:
        --------
        List[ExperimentDataset]
            List of datasets ready for model evaluation

        Raises:
        -------
        DataPreparationError
            If data loading or preprocessing fails
        DataSamplingError
            If dataset sampling fails
        """
        try:
            self.logger.info("Loading and preprocessing data...")

            # Load and preprocess time series data using PyScrew
            raw_data = load_data(
                scenario_selection=self.scenario_selection,
                target_length=self.target_length,
                screw_positions=self.screw_positions,
            )

            # Apply preprocessing pipeline (PAA + normalization)
            processed_data = process_data(raw_data, target_length=200)

            # Generate datasets based on sampling selection
            all_selections = self.get_all_selections()
            return sample_datasets(processed_data, *all_selections)

        except Exception as e:
            raise DataPreparationError(f"Failed to prepare data: {str(e)}") from e

    def get_all_models(self) -> Dict[str, Any]:
        """
        Initialize ML models with experiment configuration.

        Returns:
        --------
        Dict[str, Any]
            Dictionary mapping model names to initialized model instances

        Raises:
        -------
        ExperimentEvaluationError
            If model initialization fails
        """
        try:
            return get_classifier_dict(
                self.modeling_selection,
                self.random_seed,
                self.n_jobs,
            )
        except Exception as e:
            raise ExperimentEvaluationError(
                f"Failed to initialize models: {str(e)}"
            ) from e

    def _update_split_method(self, dataset: ExperimentDataset) -> None:
        """
        Set up cross-validation strategy for the current dataset.

        Parameters:
        -----------
        dataset : ExperimentDataset
            Dataset to configure CV splits for
        """
        split_params = {
            "n_splits": self.cv_folds,
            "shuffle": True,
            "random_state": self.random_seed,
        }

        if self.stratify:
            cv = StratifiedKFold(**split_params)
            self.split_method = lambda: cv.split(dataset.x_values, dataset.y_values)
            self.logger.debug(f"Using StratifiedKFold with {self.cv_folds} splits")
        else:
            cv = KFold(**split_params)
            self.split_method = lambda: cv.split(dataset.x_values)
            self.logger.debug(f"Using regular KFold with {self.cv_folds} splits")

    def _run_experiment(self) -> None:
        """
        Execute main experiment loop with comprehensive error handling.

        Processes all datasets in the experiment, handling failures according
        to the fail_fast configuration. Failed datasets are tracked but don't
        stop the experiment unless fail_fast=True.

        Raises:
        -------
        ExperimentEvaluationError
            If experiment setup fails or fail_fast=True and a dataset fails
        """
        self.experiment_result = ExperimentResult(*self.get_all_selections())

        try:
            # Start MLflow experiment tracking
            self.manager.start_experiment_run(self.experiment_result)

            # Process each dataset in the experiment
            for dataset_idx, dataset in enumerate(self.datasets):
                progress = f"{dataset_idx + 1}/{len(self.datasets)}"
                self.logger.info(f"Processing dataset {progress}: {dataset.name}")

                try:
                    # Process all models on this dataset
                    dataset_result = self._run_dataset(dataset)
                    self.experiment_result.add_result(dataset_result)
                    self.manager.update_experiment_run(self.experiment_result)
                    self.logger.info(f"Dataset {dataset.name} completed successfully")

                except DatasetEvaluationError as e:
                    if self.fail_fast:
                        # Stop immediately on first dataset failure
                        raise ExperimentEvaluationError(
                            f"Dataset {dataset.name} failed: {str(e)}"
                        ) from e
                    else:
                        # Log failure and continue with next dataset
                        self.logger.warning(
                            f"Dataset {dataset.name} failed: {str(e)} - continuing with next dataset"
                        )

            # Mark experiment as complete (triggers failure summary logging)
            self.experiment_result.end()

        except ExperimentEvaluationError:
            # Mark experiment as failed and re-raise
            self.experiment_result.mark_failed("Experiment evaluation failed")
            raise
        except Exception as e:
            # Handle unexpected errors (MLflow setup, configuration issues, etc.)
            self.experiment_result.mark_failed(f"Unexpected error: {str(e)}")
            raise ExperimentEvaluationError(f"Experiment failed: {str(e)}") from e
        finally:
            # Always cleanup MLflow, even if setup fails
            self.manager.end_run()

    def _run_dataset(self, experiment_dataset: ExperimentDataset) -> DatasetResult:
        """
        Process all models on a single dataset with comprehensive error handling.

        Parameters:
        -----------
        experiment_dataset : ExperimentDataset
            Dataset to process with all configured models

        Returns:
        --------
        DatasetResult
            Results container with all model evaluations

        Raises:
        -------
        DatasetEvaluationError
            If dataset setup fails or fail_fast=True and a model fails
        """
        dataset_result = DatasetResult(experiment_dataset)

        try:
            # Setup MLflow tracking and cross-validation for this dataset
            self.manager.start_dataset_run(dataset_result)
            self._update_split_method(experiment_dataset)

            # Process each model on this dataset
            for model_idx, model_name in enumerate(self.models):
                progress = f"({model_idx + 1}/{len(self.models)})"
                self.logger.info(f"- Applying model {progress}: {model_name}")

                try:
                    # Train and evaluate model across all CV folds
                    model_result = self._run_model(experiment_dataset, model_name)
                    dataset_result.add_result(model_result)
                    self.manager.update_dataset_run(dataset_result)

                    # Log model success with performance metric
                    f1_score = model_result.get_metrics().get("f1_score", 0.0)
                    self.logger.info(f"{model_name}: f1_score = {f1_score:.3f}")

                except ModelEvaluationError as e:
                    if self.fail_fast:
                        # Stop immediately on first model failure
                        raise DatasetEvaluationError(
                            f"Model {model_name} failed: {str(e)}"
                        ) from e
                    else:
                        # Log failure and continue with next model
                        self.logger.warning(
                            f"Model {model_name} failed: {str(e)} - continuing with next model"
                        )

        except DatasetEvaluationError:
            # Mark dataset as failed and re-raise
            dataset_result.mark_failed("Dataset evaluation failed")
            raise
        except Exception as e:
            # Handle unexpected errors (CV setup, data issues, etc.)
            dataset_result.mark_failed(f"Unexpected error: {str(e)}")
            raise DatasetEvaluationError(f"Dataset processing failed: {str(e)}") from e
        finally:
            # Always cleanup MLflow, even if setup fails
            self.manager.end_run()

        return dataset_result

    def _run_model(
        self, experiment_dataset: ExperimentDataset, model_name: str
    ) -> ModelResult:
        """
        Process all CV folds for a single model with comprehensive error handling.

        Parameters:
        -----------
        experiment_dataset : ExperimentDataset
            Dataset to train/evaluate the model on
        model_name : str
            Name of the model to evaluate

        Returns:
        --------
        ModelResult
            Results container with all fold evaluations

        Raises:
        -------
        ModelEvaluationError
            If model setup fails or fail_fast=True and a fold fails
        """
        model_result = ModelResult(experiment_dataset, model_name)

        try:
            # Setup MLflow tracking and generate CV splits for this model
            self.manager.start_model_run(model_result)
            splits = list(self.split_method())

            # Process each cross-validation fold
            for fold_index in range(self.cv_folds):
                try:
                    # Train and evaluate on this fold
                    fold_result = self._run_fold(
                        experiment_dataset, model_name, fold_index, splits[fold_index]
                    )
                    model_result.add_result(fold_result)
                    self.manager.update_model_run(model_result)

                except FoldEvaluationError as e:
                    if self.fail_fast:
                        # Stop immediately on first fold failure
                        raise ModelEvaluationError(
                            f"Fold {fold_index} failed: {str(e)}"
                        ) from e
                    else:
                        # Create failed fold result and continue
                        self.logger.warning(
                            f"Fold {fold_index} failed: {str(e)} - continuing with next fold"
                        )
                        fold_result = FoldResult(
                            experiment_dataset,
                            model_name,
                            fold_index,
                            self.models[model_name],
                        )
                        fold_result.mark_failed(str(e))
                        model_result.add_result(fold_result)

        except ModelEvaluationError:
            # Mark model as failed and re-raise
            model_result.mark_failed("Model evaluation failed")
            raise
        except Exception as e:
            # Handle unexpected errors (model initialization, split generation, etc.)
            model_result.mark_failed(f"Unexpected error: {str(e)}")
            raise ModelEvaluationError(f"Model processing failed: {str(e)}") from e
        finally:
            # Always cleanup MLflow, even if setup fails
            self.manager.end_run()

        return model_result

    def _run_fold(
        self,
        experiment_dataset: ExperimentDataset,
        model_name: str,
        fold_index: int,
        split_indices: Tuple[np.ndarray, np.ndarray],
    ) -> FoldResult:
        """
        Process a single cross-validation fold with comprehensive error handling.

        Parameters:
        -----------
        experiment_dataset : ExperimentDataset
            Dataset containing the data to split
        model_name : str
            Name of the model to train/evaluate
        fold_index : int
            Index of the current CV fold (0-based)
        split_indices : Tuple[np.ndarray, np.ndarray]
            Tuple of (train_indices, test_indices) for this fold

        Returns:
        --------
        FoldResult
            Results container with fold-level evaluation metrics

        Raises:
        -------
        FoldEvaluationError
            If fold training/evaluation fails
        """
        model = self.models[model_name]
        fold_result = FoldResult(experiment_dataset, model_name, fold_index, model)

        try:
            # Setup MLflow tracking for this fold
            self.manager.start_fold_run(fold_result)
            train_idx, test_idx = split_indices

            # Train model and evaluate on test set (can raise FoldEvaluationError)
            fold_result.train_and_evaluate(train_idx, test_idx)

            # Update MLflow with fold results
            self.manager.update_fold_run(fold_result)

            # Log fold success with performance metric
            f1_score = fold_result.metrics.get("f1_score", 0)
            self.logger.debug(f"Fold {fold_index}: f1_score = {f1_score:.3f}")

        except FoldEvaluationError:
            # Mark fold as failed and re-raise for model-level handling
            fold_result.mark_failed("Fold training/evaluation failed")
            raise
        except Exception as e:
            # Handle unexpected errors (MLflow issues, data access, etc.)
            fold_result.mark_failed(f"Unexpected error: {str(e)}")
            raise FoldEvaluationError(f"Fold processing failed: {str(e)}") from e
        finally:
            # Always cleanup MLflow, even if setup fails
            self.manager.end_run()

        return fold_result

    def _evaluate_experiment(self) -> None:
        """
        Generate experiment summary and log overall results.

        This method provides a high-level summary of experiment completion.
        Detailed failure analysis is handled automatically by ExperimentResult.end().
        """
        self.logger.info("Evaluating experiment results...")
        self.logger.info(
            f"Experiment completed with {len(self.experiment_result.dataset_results)} datasets"
        )

        # Additional summary metrics could be added here
        total_models = sum(
            len(dr.model_results) for dr in self.experiment_result.dataset_results
        )
        total_folds = sum(
            sum(len(mr.fold_results) for mr in dr.model_results)
            for dr in self.experiment_result.dataset_results
        )
        self.logger.info(
            f"Total evaluations: {total_models} models, {total_folds} folds"
        )

    def run(self) -> ExperimentResult:
        """
        Execute complete experiment workflow with robust error handling.

        This is the main entry point for running experiments. It orchestrates
        the entire pipeline from data loading to final result generation.

        Returns:
        --------
        ExperimentResult
            Hierarchical results with 4-level analysis structure:
            Experiment -> Dataset -> Model -> Fold

        Raises:
        -------
        DataPreparationError
            If initial data loading/preprocessing fails
        DataSamplingError
            If dataset generation fails
        ExperimentEvaluationError
            If critical experiment setup or execution fails

        Notes:
        ------
        The experiment result contains comprehensive failure tracking at all levels,
        enabling rich post-experiment analysis even when some components fail.
        Use result.to_dataframe() to get a flat CSV-ready view of all results.
        """
        try:
            # Phase 1: Data preparation and model initialization
            self.logger.info("Phase 1: Loading data and initializing models...")
            self.datasets = self.get_all_datasets()
            self.models = self.get_all_models()

            self.logger.info(
                f"Loaded {len(self.datasets)} datasets and {len(self.models)} models"
            )

            # Phase 2: MLflow setup
            self.logger.info("Phase 2: Setting up experiment tracking...")
            self._setup_experiment_tracking()

            # Phase 3: Main experiment execution
            self.logger.info("Phase 3: Running experiment...")
            self._run_experiment()

            # Phase 4: Final evaluation and summary
            self.logger.info("Phase 4: Generating experiment summary...")
            self._evaluate_experiment()

            return self.experiment_result

        except (DataPreparationError, DataSamplingError, ExperimentEvaluationError):
            # Re-raise known experiment errors without wrapping
            raise
        except Exception as e:
            # Wrap unexpected errors for consistent error handling
            raise ExperimentEvaluationError(
                f"Unexpected error in experiment pipeline: {str(e)}"
            ) from e
