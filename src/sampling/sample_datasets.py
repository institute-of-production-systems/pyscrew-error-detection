from .dispatcher import get_sampling_functions


def sample_datasets(processed_data, experiment_name, scenario_id):
    """Convenience wrapper that maintains the original interface"""

    sampling_functions = get_sampling_functions(scenario_id, [experiment_name])
    return sampling_functions[experiment_name](processed_data)
