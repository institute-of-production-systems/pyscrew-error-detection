import time
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sktime.classification.base import BaseClassifier

from src.experiments.experiment_dataset import ExperimentDataset

from .base_result import BaseResult


class FoldResult(BaseResult):
    """Results from evaluating a model on a single cross-validation fold."""

    def __init__(
        self,
        experiment_dataset: ExperimentDataset,
        model_name: str,
        fold_index: int,
    ):
        """Initialize fold result with inherited trio."""
        # Call parent constructor with the core configuration
        super().__init__(
            experiment_dataset.scenario_selection,
            experiment_dataset.sampling_selection,
            experiment_dataset.modeling_selection,
        )
        self.name = f"fold_{fold_index}"

        # Store references for training
        self.experiment_dataset = experiment_dataset
        self.model_name = model_name
        self.fold_index = fold_index

        # Initialize empty - will be populated by train_and_evaluate
        self.metrics = {}
        self.y_true = None
        self.y_pred = None
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.metadata = {}

    def train_and_evaluate(
        self,
        model: Union[BaseEstimator, BaseClassifier],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Train model and evaluate on test set."""
        try:
            # Split the data using indices
            x_train = self.experiment_dataset.x_values[train_idx]
            x_test = self.experiment_dataset.x_values[test_idx]
            y_train = self.experiment_dataset.y_values[train_idx]
            y_test = self.experiment_dataset.y_values[test_idx]

            # Train and evaluate
            self._train_model(model, x_train, y_train)
            self._evaluate_model(model, x_test, y_test)

            # Store metadata
            self.metadata = {
                "model_type": (
                    "sktime" if self._is_sktime_classifier(model) else "sklearn"
                ),
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
            }

        except Exception as e:
            # Handle errors gracefully
            self.metrics = {
                "error": 1.0,
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
            self.y_true = np.array([0])
            self.y_pred = np.array([0])
            self.training_time = 0.0
            self.prediction_time = 0.0
            self.metadata = {"error": str(e), "failed": True}

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for MLflow logging."""
        return {
            "fold_index": self.fold_index,
            "fold_type": "cross_validation",
            "model_name": self.model_name,
            **{
                k: v
                for k, v in self.metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
        }

    def _train_model(self, model, x_train, y_train) -> None:
        """Train the model with timing."""
        # Format data for model type
        is_sktime_model = self._is_sktime_classifier(model)
        x_train_formatted = self._format_data_for_modeling(x_train, is_sktime_model)

        # Train with timing
        train_start_time = time.time()
        model.fit(x_train_formatted, y_train)
        self.training_time = time.time() - train_start_time

    def _evaluate_model(self, model, x_test, y_test) -> None:
        """Evaluate the model with timing."""
        # Format data for model type
        is_sktime_model = self._is_sktime_classifier(model)
        x_test_formatted = self._format_data_for_modeling(x_test, is_sktime_model)

        # Predict with timing
        pred_start_time = time.time()
        y_pred = model.predict(x_test_formatted)
        self.prediction_time = time.time() - pred_start_time

        # Format predictions
        self.y_pred = self._format_pred_for_evaluation(y_pred)
        self.y_true = self._format_pred_for_evaluation(y_test)

        # Calculate metrics
        self.metrics = self._evaluate_fold_metrics(self.y_true, self.y_pred)

    def _is_sktime_classifier(self, model) -> bool:
        """Determine if a model is a sktime classifier."""
        if isinstance(model, BaseClassifier):
            return True
        if hasattr(model, "_is_sktime_classifier") or hasattr(
            model, "_is_sktime_estimator"
        ):
            return True
        if hasattr(model, "__module__") and any(
            mod in model.__module__ for mod in ["sktime", "sktime_dl", "sktime_forest"]
        ):
            return True
        return False

    def _format_data_for_modeling(self, x_values, is_sktime_model):
        """Format data to be compatible with the specific model type."""
        if not is_sktime_model:
            return x_values
        else:
            from sktime.datatypes._panel._convert import from_2d_array_to_nested

            return from_2d_array_to_nested(x_values)

    def _format_pred_for_evaluation(self, y_pred):
        """Format model predictions to a standard format for evaluation."""
        if hasattr(y_pred, "values"):
            return y_pred.values
        if not isinstance(y_pred, np.ndarray):
            return np.array(y_pred)
        return y_pred

    def _evaluate_fold_metrics(self, y_true, y_pred) -> dict:
        """Evaluate metrics for a single fold."""
        from src.evaluation.get_metrics import get_metrics

        return get_metrics(y_true, y_pred)

    def get_result_tags(self) -> Dict[str, str]:
        """Get tags for MLflow logging with inherited trio."""
        return {
            **super().get_result_tags(),
            "model_name": self.model_name,
            "fold_index": str(self.fold_index),
            "fold_type": "cross_validation",
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics for MLflow logging."""
        metrics = self.metrics.copy()
        metrics.update(
            {
                "training_time_seconds": self.training_time,
                "prediction_time_seconds": self.prediction_time,
            }
        )
        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for this fold."""
        from sklearn.metrics import confusion_matrix

        return confusion_matrix(self.y_true, self.y_pred)

    def __repr__(self) -> str:
        f1 = self.metrics.get("f1_score", 0)
        acc = self.metrics.get("accuracy", 0)
        return f"FoldResult(fold={self.fold_index}, f1={f1:.3f}, acc={acc:.3f})"
