import time
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sktime.classification.base import BaseClassifier

from src.evaluation.results import FoldResult


def train_single_fold(
    model: Union[BaseEstimator, BaseClassifier],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_index: int,
) -> FoldResult:
    """
    Train and evaluate a model on a single cross-validation fold.

    This function handles all the low-level ML work:
    - Model type detection and data formatting
    - Training with timing
    - Prediction with timing
    - Metric evaluation
    - FoldResult creation

    Parameters:
    -----------
    model : Union[BaseEstimator, BaseClassifier]
        The model to train (sklearn or sktime)
    x_train, x_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test labels
    fold_index : int
        Index of this fold (for tracking)

    Returns:
    --------
    FoldResult
        Complete results for this fold
    """
    try:
        # Detect model type and format data appropriately
        is_sktime_model = _is_sktime_classifier(model)
        x_train_formatted = _format_data_for_modeling(x_train, is_sktime_model)
        x_test_formatted = _format_data_for_modeling(x_test, is_sktime_model)

        # Train the model with timing
        train_start_time = time.time()
        model.fit(x_train_formatted, y_train)
        training_time = time.time() - train_start_time

        # Make predictions with timing
        pred_start_time = time.time()
        y_pred = model.predict(x_test_formatted)
        prediction_time = time.time() - pred_start_time

        # Standardize prediction and true label formats
        y_pred_formatted = _format_pred_for_evaluation(y_pred)
        y_true_formatted = _format_pred_for_evaluation(y_test)

        # Evaluate metrics for this fold
        fold_metrics = _evaluate_fold_metrics(y_true_formatted, y_pred_formatted)

        # Create and return fold result
        return FoldResult(
            fold_index=fold_index,
            metrics=fold_metrics,
            y_true=y_true_formatted,
            y_pred=y_pred_formatted,
            training_time=training_time,
            prediction_time=prediction_time,
            metadata={
                "model_type": "sktime" if is_sktime_model else "sklearn",
                "train_samples": len(x_train),
                "test_samples": len(x_test),
            },
        )

    except Exception as e:
        # Return error fold result instead of crashing
        return FoldResult(
            fold_index=fold_index,
            metrics={
                "error": 1.0,
                "accuracy": None,
                "f1_score": None,
                "precision": None,
                "recall": None,
            },
            y_true=np.array([0]),  # Minimal arrays to satisfy validation
            y_pred=np.array([0]),
            training_time=0.0,
            prediction_time=0.0,
            metadata={"error": str(e), "failed": True},
        )


def _is_sktime_classifier(model: Union[BaseEstimator, BaseClassifier]) -> bool:
    """Determine if a model is a sktime classifier."""
    # Check if it's an instance of sktime's BaseClassifier
    if isinstance(model, BaseClassifier):
        return True

    # Check for sktime-specific attributes
    if hasattr(model, "_is_sktime_classifier") or hasattr(
        model, "_is_sktime_estimator"
    ):
        return True

    # Check the module path
    if hasattr(model, "__module__") and any(
        mod in model.__module__ for mod in ["sktime", "sktime_dl", "sktime_forest"]
    ):
        return True

    return False


def _format_data_for_modeling(
    x_values: np.ndarray, is_sktime_model: bool
) -> Union[np.ndarray, pd.DataFrame]:
    """Format data to be compatible with the specific model type."""
    if not is_sktime_model:
        # For sklearn models: ensure numpy array format
        return x_values
    else:
        # Convert to nested format for sktime models
        from sktime.datatypes._panel._convert import from_2d_array_to_nested

        return from_2d_array_to_nested(x_values)


def _format_pred_for_evaluation(
    y_pred: Union[np.ndarray, pd.Series, pd.DataFrame],
) -> np.ndarray:
    """Format model predictions to a standard format for evaluation."""
    # For pandas objects
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        return y_pred.values

    # For any other case where we don't have a numpy array
    if not isinstance(y_pred, np.ndarray):
        return np.array(y_pred)

    # Already in the correct format
    return y_pred


def _evaluate_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate metrics for a single fold."""
    from src.evaluation.get_metrics import get_metrics

    return get_metrics(y_true, y_pred)
