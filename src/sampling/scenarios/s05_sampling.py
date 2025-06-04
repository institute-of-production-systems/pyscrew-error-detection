from typing import Callable, Dict, List, Tuple

import numpy as np

from ...experiments.experiment_dataset import ExperimentDataset


def get_experiment_functions(
    experiment_names: List[str],
) -> Dict[
    str, Callable[[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[ExperimentDataset]]
]:
    """
    Returns dict of lazy functions for s05 experiments.
    TODO: Implement s05-specific sampling logic (uses 'is normal' tags instead of conditions)
    """
    functions = {}
    for exp_name in experiment_names:
        functions[exp_name] = lambda processed_data: _placeholder_implementation(
            exp_name
        )
    return functions


def _placeholder_implementation(experiment_name: str) -> List[ExperimentDataset]:
    """Placeholder - replace with actual s05 sampling logic"""
    raise NotImplementedError(
        f"s05 sampling for '{experiment_name}' not yet implemented"
    )
