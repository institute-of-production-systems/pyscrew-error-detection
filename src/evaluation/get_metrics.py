from typing import Dict, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    v_measure_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def get_metrics(
    y_true: Optional[Union[Sequence, np.ndarray]] = None,
    y_pred: Optional[Union[Sequence, np.ndarray]] = None,
    include_clustering_metrics: bool = True,
) -> Dict[str, float]:
    """
    Get dict to evaluate model performance for classification.

    Can be called in two modes:
    1. Without y_true/y_pred: Returns empty metrics dict for initialization
    2. With y_true/y_pred: Calculates and returns actual metrics

    Parameters:
    -----------
    y_true : array-like, optional
        Ground truth labels. If None, returns empty metrics.
    y_pred : array-like, optional
        Predicted labels. If None, returns empty metrics.
    include_clustering_metrics : bool, default=True
        Whether to include clustering evaluation metrics

    Returns:
    --------
    Dict[str, float]
        Dictionary containing evaluation metrics.
        Values are None if y_true/y_pred not provided, else actual scores.
    """
    # Define base metrics that are always included
    base_metrics = ["accuracy", "precision", "recall", "f1_score"]

    # Define clustering metrics (for future unsupervised experiments)
    clustering_metrics = [
        "adjusted_rand_score",
        "normalized_mutual_info_score",
        "homogeneity_score",
        "completeness_score",
        "v_measure_score",
        "fowlkes_mallows_score",
    ]

    # Return empty metrics for initialization
    if y_true is None or y_pred is None:
        init_value = 0.0
        metrics = {metric: init_value for metric in base_metrics}
        if include_clustering_metrics:
            metrics.update({metric: init_value for metric in clustering_metrics})
        return metrics

    # Determine averaging strategy based on number of classes
    n_unique_classes = len(np.unique(y_true))

    if n_unique_classes < 2:
        raise ValueError(
            f"Invalid number of classes: {n_unique_classes}. Need at least 2 classes."
        )
    elif n_unique_classes == 2:
        # Binary classification
        average = "binary"
        zero_division = 0
    else:
        # Multi-class classification
        average = "weighted"
        zero_division = 0

    # Calculate base metrics
    result_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        "f1_score": f1_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
    }

    # Add clustering metrics if requested
    if include_clustering_metrics:
        result_dict.update(
            {
                "adjusted_rand_score": adjusted_rand_score(y_true, y_pred),
                "normalized_mutual_info_score": normalized_mutual_info_score(
                    y_true, y_pred
                ),
                "homogeneity_score": homogeneity_score(y_true, y_pred),
                "completeness_score": completeness_score(y_true, y_pred),
                "v_measure_score": v_measure_score(y_true, y_pred),
                "fowlkes_mallows_score": fowlkes_mallows_score(y_true, y_pred),
            }
        )

    logger.debug(
        f"Calculated {len(result_dict)} metrics for {n_unique_classes}-class classification"
    )

    return result_dict
