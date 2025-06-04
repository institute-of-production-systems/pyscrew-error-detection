def get_sampling_functions(scenario_id, experiment_names):
    """Returns dict of functions that accept processed_data and return datasets"""

    if scenario_id == "s04":
        from .scenarios.s04_sampling import get_experiment_functions

        return get_experiment_functions(experiment_names)

    # elif etc. will follow in the next updates
