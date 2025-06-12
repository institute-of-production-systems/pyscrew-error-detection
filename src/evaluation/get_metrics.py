from typing import Dict, Sequence, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import get_logger

logger = get_logger(__name__)


def get_metrics(
    y_true: Union[Sequence, np.ndarray],
    y_pred: Union[Sequence, np.ndarray],
    sampling_selection: str = None,
) -> Dict[str, float]:
    """
    Get dict to evaluate model performance for classification using direct metrics.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    sampling_selection : str, optional
        Type of experiment ('binary_vs_ref', 'binary_vs_all', 'multiclass_with_groups', 'multiclass_with_all')

    Returns:
    --------
    dict : Dictionary containing evaluation metrics (accuracy, precision, recall, f1_score)
    """
    # Base metric that (should) always works
    result_dict = {"accuracy": accuracy_score(y_true, y_pred)}

    # Determine metric parameters based on experiment type
    if sampling_selection is None:
        n_unique = len(np.unique(y_true))
        experiment_type = "bi" if n_unique == 2 else "mc"
        logger.debug(
            f"No experiment name provided for metric selection: \
            Infering experiment type from 'y_true' as '{experiment_type}"
        )
    else:
        experiment_type = sampling_selection.split("_")[0]

    if experiment_type in ["bi", "binary"]:  # Binary classification
        # the sampling defined that 1 = faulty
        metric_params = {"pos_label": 1, "average": "binary", "zero_division": 0}

    elif experiment_type in ["mc", "multiclass"]:  # Multiclass classification
        metric_params = {"average": "weighted", "zero_division": 0}

    else:  # Invalid
        raise ValueError(
            f"Invalid experiment name: '{sampling_selection}'. "
            f"Must start with 'binary' or 'multiclass'"
        )

    # Calculate remaining metrics with shared parameters
    result_dict.update(
        {
            "precision": precision_score(y_true, y_pred, **metric_params),
            "recall": recall_score(y_true, y_pred, **metric_params),
            "f1_score": f1_score(y_true, y_pred, **metric_params),
        }
    )

    if True:  # TODO: Add clustering metrics when implementing unsupervised experiments
        # NOTE: The downstream code (ModelResult, plotting, etc.) can handle
        # additional metrics as long as they return floats. The aggregation
        # will still automatically compute mean/std/min/max for any metric in the dict.
        #
        # Example clustering metrics for future implementation:
        from sklearn.metrics import (
            adjusted_rand_score,
            completeness_score,
            fowlkes_mallows_score,
            homogeneity_score,
            normalized_mutual_info_score,
            silhouette_score,
            v_measure_score,
        )

        result_dict.update(
            {
                # External evaluation metrics (when ground truth is available)
                "ari": adjusted_rand_score(y_true, y_pred),
                "nmi": normalized_mutual_info_score(y_true, y_pred),
                "homogeneity": homogeneity_score(y_true, y_pred),
                "completeness": completeness_score(y_true, y_pred),
                "v_measure": v_measure_score(y_true, y_pred),
                "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
                # TODO: Internal metrics would require X data
                # silhouette_score(X, y_pred)
            }
        )

    logger.debug(f"Fetching {len(result_dict)} metrics via result dict.")
    return result_dict
