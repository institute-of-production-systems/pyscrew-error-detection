def get_sampling_functions(scenario_id, experiment_names):
    """Returns dict of functions that accept processed_data and return datasets"""

    if scenario_id == "s04":
        from .scenarios.s04_sampling import get_experiment_functions

        return get_experiment_functions(experiment_names)

    elif scenario_id == "s05":
        from .scenarios.s05_sampling import get_experiment_functions

        return get_experiment_functions(experiment_names)

    elif scenario_id == "s06":
        from .scenarios.s06_sampling import get_experiment_functions

        return get_experiment_functions(experiment_names)

    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")
