from .dispatcher import get_sampling_functions


def sample_datasets(processed_data, sampling_selection, scenario_selection):
    """Convenience wrapper that maintains the original interface"""

    sampling_functions = get_sampling_functions(
        scenario_selection, [sampling_selection]
    )
    return sampling_functions[sampling_selection](processed_data)
