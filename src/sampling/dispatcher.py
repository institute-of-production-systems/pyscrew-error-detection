def get_sampling_functions(scenario_selection, sampling_selections):
    """Returns dict of functions that accept processed_data and return datasets"""

    if scenario_selection == "s04":
        from .scenarios.s04_sampling import get_experiment_functions

        return get_experiment_functions(sampling_selections)

    elif scenario_selection == "s05":
        from .scenarios.s05_sampling import get_experiment_functions

        return get_experiment_functions(sampling_selections)

    elif scenario_selection == "s06":
        from .scenarios.s06_sampling import get_experiment_functions

        return get_experiment_functions(sampling_selections)

    else:
        raise ValueError(f"Unknown scenario: {scenario_selection}")
