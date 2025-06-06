from typing import Tuple

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def _apply_paa(torque_values: np.ndarray, target_length: int) -> np.ndarray:
    """
    Apply Piecewise Aggregate Approximation (PAA) using numpy vectorization.

    PAA is a dimensionality reduction technique for time series data that
    divides a time series into equal-sized segments and represents each
    segment by its mean value. This implementation handles both evenly
    and non-evenly divisible time series lengths.

    Parameters:
    -----------
    torque_values : np.ndarray
        Time series data with shape (n_samples, n_timestamps)
        Each row represents a separate time series sample
    target_length : int
        Target length after PAA transformation
        Must be > 0 and <= original_length

    Returns:
    --------
    np.ndarray
        PAA transformed time series with shape (n_samples, target_length)
        Each time series is reduced to the specified target length

    Notes:
    ------
    The algorithm uses two different approaches:
    1. For evenly divisible cases: reshapes the array and computes means
       using matrix operations for better performance
    2. For non-evenly divisible cases: computes segment boundaries and
       calculates means for each segment using vectorized operations
    """
    n_samples, original_length = torque_values.shape

    logger.debug(
        f"Applying PAA: {original_length} --> {target_length} time points for {n_samples} samples"
    )

    # Handle the case where segments divide evenly
    if original_length % target_length == 0:
        segment_size = original_length // target_length
        logger.debug(f"Using evenly divisible PAA with segment size {segment_size}")
        # Reshape and compute means along segments
        return np.mean(
            torque_values.reshape(n_samples, target_length, segment_size), axis=2
        )

    # For non-evenly divisible case, use a vectorized approach
    logger.debug("Using non-evenly divisible PAA with boundary computation")
    bounds = np.linspace(0, original_length, target_length + 1).astype(int)

    # Initialize result array
    result = np.zeros((n_samples, target_length))

    # For each segment, compute mean using vectorized operations
    for i in range(target_length):
        result[:, i] = np.mean(torque_values[:, bounds[i] : bounds[i + 1]], axis=1)

    return result


def _normalize_data(torque_values: np.ndarray) -> np.ndarray:
    """
    Normalize time series data to zero mean and unit variance using vectorized operations.

    This function performs z-score normalization (standardization) on each time series
    independently. The normalization helps improve the performance of many machine
    learning algorithms by ensuring all features are on a similar scale.

    Parameters:
    -----------
    torque_values : np.ndarray
        Time series data with shape (n_samples, n_timestamps)
        Each row represents a separate time series sample

    Returns:
    --------
    np.ndarray
        Normalized time series with same shape as input
        Each time series has zero mean and unit variance (where possible)

    Notes:
    ------
    - For time series with zero standard deviation, only mean centering is performed
      to avoid division by zero
    - The implementation uses numpy broadcasting for efficient computation
    - Normalization is performed independently for each sample (row)
    """
    logger.debug(
        f"Normalizing {torque_values.shape[0]} time series to zero mean and unit variance"
    )

    # Calculate mean and std along time dimension for each sample
    mean = np.mean(torque_values, axis=1, keepdims=True)
    std = np.std(torque_values, axis=1, keepdims=True)

    # Replace zero standard deviations with 1 to avoid division by zero
    # This effectively skips variance normalization for constant signals
    std = np.where(std == 0, 1, std)

    # Normalize using broadcasting
    return (torque_values - mean) / std


def process_data(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply preprocessing pipeline to time series data.

    This function applies a two-step preprocessing pipeline:
    1. Dimensionality reduction using Piecewise Aggregate Approximation (PAA)
    2. Normalization (z-score standardization)

    The preprocessing helps improve model performance by:
    - Reducing computational complexity through dimensionality reduction
    - Preserving important patterns while reducing noise
    - Standardizing the scale of features

    Parameters:
    -----------
    data : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing (torque_values, class_values, scenario_condition)
        - torque_values: Time series data with shape (n_samples, n_timestamps)
        - class_values: Class labels for each sample
        - scenario_condition: Additional categorical information for each sample
    target_length : int
        Target length for PAA dimensionality reduction

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Processed data tuple with the same structure as input:
        (processed_torque_values, class_values, scenario_condition)

    Notes:
    ------
    The function ensures data integrity by verifying that the number of samples
    remains consistent across all arrays after preprocessing.
    """
    torque_values, class_values, scenario_condition = data

    original_shape = torque_values.shape
    logger.info(
        f"Starting data preprocessing: {original_shape[0]} samples, {original_shape[1]} --> {target_length} time points"
    )

    # 1. apply PAA for dimensionality reduction
    torque_values = _apply_paa(torque_values, target_length)

    # 2. normalize data for better model performance
    torque_values = _normalize_data(torque_values)

    # 3. check output integrity
    assert len(torque_values) == len(class_values)
    assert len(torque_values) == len(scenario_condition)

    logger.info(
        f"Preprocessing completed: {torque_values.shape[0]} samples, {torque_values.shape[1]} time points"
    )

    # Return processed torque values and original labels/groups
    return torque_values, class_values, scenario_condition
