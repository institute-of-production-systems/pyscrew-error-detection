from typing import List, Tuple
import numpy as np
from src.experiments.experiment_dataset import ExperimentDataset
from src.utils.exceptions import DataSamplingError


def get_sampling_function(scenario_selection: str, sampling_selection: str):
    """
    Get the specific sampling function for the given scenario-sampling combination.

    Updated to support the new 2x2 sampling matrix for s06:
    - combined_all_classes: All groups combined, all parameter values
    - combined_only_extreme: All groups combined, only most extreme values
    - separate_all_classes: Each group separate, all parameter values
    - separate_only_extreme: Each group separate, only most extreme values
    """
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
        # Import new 2x2 matrix sampling functions
        from .scenarios.s06_sampling import (
            sample_s06_combined_all_classes,
            sample_s06_combined_only_extreme,
            sample_s06_separate_all_classes,
            sample_s06_separate_only_extreme,
        )

        s06_functions = {
            "combined_all_classes": sample_s06_combined_all_classes,
            "combined_only_extreme": sample_s06_combined_only_extreme,
            "separate_all_classes": sample_s06_separate_all_classes,
            "separate_only_extreme": sample_s06_separate_only_extreme,
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
    """Generate experiment datasets based on scenario and sampling strategy."""
    try:
        sampling_function = get_sampling_function(
            scenario_selection, sampling_selection
        )
        return sampling_function(
            *processed_data, scenario_selection, sampling_selection, modeling_selection
        )
    except KeyError:
        raise DataSamplingError(
            f"Unsupported sampling '{sampling_selection}' for scenario '{scenario_selection}'. "
            f"Check available strategies for {scenario_selection} in the documentation."
        )
