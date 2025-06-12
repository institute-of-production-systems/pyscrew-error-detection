from typing import List, Tuple

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset
from src.utils.exceptions import SamplingError


def get_sampling_function(scenario_selection: str, sampling_selection: str):
    """Get the specific sampling function for the given scenario-sampling combination."""

    if scenario_selection == "s04":
        from .scenarios.s04_sampling import (
            sample_s04_binary_vs_all,
            sample_s04_binary_vs_ref,
            sample_s04_multiclass_with_all,
            sample_s04_multiclass_with_groups,
        )

        s04_functions = {
            "binary_vs_ref": sample_s04_binary_vs_ref,
            "binary_vs_all": sample_s04_binary_vs_all,
            "multiclass_with_groups": sample_s04_multiclass_with_groups,
            "multiclass_with_all": sample_s04_multiclass_with_all,
        }
        return s04_functions[sampling_selection]

    elif scenario_selection == "s05":
        from .scenarios.s05_sampling import (
            sample_s05_binary_for_extremes,
            sample_s05_multiclass_with_all,
            sample_s05_multiclass_within_groups,
        )

        s05_functions = {
            "binary_for_extremes": sample_s05_binary_for_extremes,
            "multiclass_with_all": sample_s05_multiclass_with_all,
            "multiclass_within_groups": sample_s05_multiclass_within_groups,
        }
        return s05_functions[sampling_selection]

    elif scenario_selection == "s06":
        from .scenarios.s06_sampling import (
            sample_s06_binary_for_extremes,
            sample_s06_multiclass_with_all,
            sample_s06_multiclass_within_groups,
        )

        s06_functions = {
            "binary_for_extremes": sample_s06_binary_for_extremes,
            "multiclass_with_all": sample_s06_multiclass_with_all,
            "multiclass_within_groups": sample_s06_multiclass_within_groups,
        }
        return s06_functions[sampling_selection]

    else:
        raise ValueError(f"Unknown scenario: {scenario_selection}")


def sample_datasets(
    processed_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    try:
        sampling_function = get_sampling_function(
            scenario_selection, sampling_selection
        )
        return sampling_function(
            *processed_data, scenario_selection, sampling_selection, modeling_selection
        )
    except KeyError:
        raise SamplingError(
            f"Unsupported sampling '{sampling_selection}' for scenario '{scenario_selection}'"
        )
