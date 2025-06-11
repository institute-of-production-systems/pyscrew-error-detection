from typing import Tuple

import numpy as np
import pyscrew


def load_data(
    scenario_selection: str = "s04",
    target_length: int = 2000,
    screw_positions: str = "left",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and return the pyscrew dataset.

    Parameters:
    -----------
    scenario_selection : str, optional
        Dataset ID to load (default: "s04")
    target_length : int, optional
        Length of time series (default: 2000)
    screw_positions : str, optional
        Screw positions to use (default: "left")

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]: (torque_values, class_values, scenario_condition)
    """
    # Load data using pyscrew
    data = pyscrew.get_data(
        scenario_selection,
        screw_positions=screw_positions,
        target_length=target_length,
    )

    # Extract and return relevant arrays
    torque_values = np.array(data["torque_values"])
    class_values = np.array(data["class_values"])
    scenario_condition = np.array(data["scenario_condition"])

    return torque_values, class_values, scenario_condition
