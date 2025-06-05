from typing import Callable, Dict, List, Tuple

import numpy as np

from src.utils.exceptions import SamplingError

from ...experiments.experiment_dataset import ExperimentDataset
from ..groups import load_groups


def get_experiment_functions(
    experiment_names: List[str],
) -> Dict[
    str, Callable[[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[ExperimentDataset]]
]:
    """
    Returns dict of lazy functions for s06 experiments.

    Note: The control group ("group_control") is intentionally skipped in all experiments
    to focus on parameter variation effects rather than baseline measurements.
    """
    functions = {}
    for exp_name in experiment_names:
        if exp_name == "multiclass_within_groups":
            functions[exp_name] = lambda processed_data: _get_multiclass_within_groups(
                *processed_data
            )
        elif exp_name == "multiclass_with_all":
            functions[exp_name] = lambda processed_data: _get_multiclass_with_all(
                *processed_data
            )
        elif exp_name == "binary_for_extremes":
            functions[exp_name] = lambda processed_data: _get_binary_for_extremes(
                *processed_data
            )
        else:
            functions[exp_name] = lambda processed_data: _placeholder_implementation(
                exp_name
            )
    return functions


def _placeholder_implementation(experiment_name: str) -> List[ExperimentDataset]:
    """Placeholder for any additional experiments not yet implemented"""
    raise NotImplementedError(
        f"s06 sampling for '{experiment_name}' not yet implemented."
    )


def _extract_parameter_value(class_name, group_name):
    """
    Apply simple static renaming for duplicate parameter values in s06.

    Explicit mappings for known duplicates:
    - "101_cooling-time-25-1" -> "100_cooling-time-25"
    - "107_cooling-time-25-2" -> "100_cooling-time-25" (same!)
    - "401_switching-point-22-1" -> "400_switching-point-22"
    - "409_switching-point-22-2" -> "400_switching-point-22" (same!)
    - "501_injection-velocity-030-1" -> "500_injection-velocity-030"
    - "509_injection-velocity-030-2" -> "500_injection-velocity-030" (same!)
    """

    # Static renaming for cooling time duplicates
    if class_name == "101_cooling-time-25-1" or class_name == "107_cooling-time-25-2":
        return "100_cooling-time-25"

    # Static renaming for switching point duplicates
    elif (
        class_name == "401_switching-point-22-1"
        or class_name == "409_switching-point-22-2"
    ):
        return "400_switching-point-22"

    # Static renaming for injection velocity duplicates
    elif (
        class_name == "501_injection-velocity-030-1"
        or class_name == "509_injection-velocity-030-2"
    ):
        return "500_injection-velocity-030"

    # For all other classes, return as-is
    else:
        return class_name


def _get_multiclass_within_groups(torque_values, class_values, scenario_condition):
    """
    Generate datasets for multi-class classification within s06 error groups.

    Note: Skips the control group to focus on parameter variations.
    """

    # Configuration: Group duplicate parameter values
    GROUP_DUPLICATE_PARAMETERS = True

    # Initialize a list to return the experiment datasets
    datasets: List[ExperimentDataset] = []

    # Load s06 groups from JSON
    groups = load_groups("s06")

    for group_name, group_classes in groups.items():

        # Skip the control group as requested
        if group_name == "group_control":
            continue

        if GROUP_DUPLICATE_PARAMETERS:
            # Extract parameter values and group duplicates
            parameter_values = [
                _extract_parameter_value(cls, group_name) for cls in group_classes
            ]
            unique_class_names = sorted(set(parameter_values))

            # Create mapping from original class name to parameter value
            class_to_param = {
                cls: _extract_parameter_value(cls, group_name) for cls in group_classes
            }
        else:
            # Use original class names without grouping
            unique_class_names = [
                str(class_name) for class_name in sorted(set(group_classes))
            ]
            class_to_param = {cls: cls for cls in group_classes}  # Identity mapping

        # Filter torque values by class_values
        class_group_mask = np.isin(class_values, group_classes)
        filtered_torque_values = torque_values[class_group_mask]
        filtered_class_values = class_values[class_group_mask]
        filtered_condition = scenario_condition[class_group_mask]

        # Skip if no data for this group
        if len(filtered_torque_values) == 0:
            continue

        # Get indices of normal and faulty samples and get counts
        normal_mask = filtered_condition == "normal"
        faulty_mask = filtered_condition != "normal"
        normal_counts = int(np.sum(normal_mask))
        faulty_counts = int(np.sum(faulty_mask))
        fault_ratio = round(faulty_counts / (faulty_counts + normal_counts), 4)

        # Build a mapping dict for the class names
        class_names = {k: v for k, v in enumerate(unique_class_names)}
        class_names_to_idx = {v: k for k, v in enumerate(unique_class_names)}

        # Make y values int instead of string using the mapping
        if GROUP_DUPLICATE_PARAMETERS:
            # Map original class names to parameter values, then to indices
            filtered_class_values = np.array(
                [
                    class_names_to_idx[class_to_param[fcv]]
                    for fcv in filtered_class_values
                ]
            )
        else:
            # Use original mapping
            filtered_class_values = np.array(
                [class_names_to_idx[fcv] for fcv in filtered_class_values]
            )

        # Create ExperimentDataset for this group
        dataset = ExperimentDataset(
            name=group_name,
            x_values=filtered_torque_values,
            y_values=filtered_class_values,
            experiment_name="multiclass_within_groups",
            class_count=len(unique_class_names),
            class_names=class_names,
            normal_counts=normal_counts,
            faulty_counts=faulty_counts,
            faulty_ratio=fault_ratio,
            description=f"Multi-class classification within s06 group '{group_name}'"
            + (f" (grouped duplicate parameters)" if GROUP_DUPLICATE_PARAMETERS else "")
            + " (control group excluded)",
        )
        datasets.append(dataset)

    # Check if we have any datasets
    if not datasets:
        raise SamplingError(
            f"No valid datasets could be created for multiclass_with_groups experiment (control group excluded)"
        )

    return datasets


def _get_binary_for_extremes(torque_values, class_values, scenario_condition):
    """
    Generate datasets for binary classification between parameter extremes in each s06 group.

    Note: Skips the control group to focus on parameter variations.
    """
    # Initialize a list to return the experiment datasets
    datasets: List[ExperimentDataset] = []

    # Load s06 groups from JSON
    groups = load_groups("s06")

    for group_name, group_classes in groups.items():

        # Skip the control group as requested
        if group_name == "group_control":
            continue

        # Apply parameter grouping to handle duplicates
        parameter_values = [
            _extract_parameter_value(cls, group_name) for cls in group_classes
        ]
        unique_parameter_values = sorted(set(parameter_values))

        # Take first (most normal) and last (most abnormal)
        most_normal_param = unique_parameter_values[0]
        most_abnormal_param = unique_parameter_values[-1]

        # Find original class names that map to these parameter values
        most_normal_classes = [
            cls
            for cls in group_classes
            if _extract_parameter_value(cls, group_name) == most_normal_param
        ]
        most_abnormal_classes = [
            cls
            for cls in group_classes
            if _extract_parameter_value(cls, group_name) == most_abnormal_param
        ]

        extreme_classes = most_normal_classes + most_abnormal_classes

        # Filter data for only these extreme classes
        extreme_mask = np.isin(class_values, extreme_classes)
        filtered_torque_values = torque_values[extreme_mask]
        filtered_class_values = class_values[extreme_mask]
        filtered_condition = scenario_condition[extreme_mask]

        # Skip if we don't have both extremes
        if (
            len(
                set(
                    [
                        _extract_parameter_value(cls, group_name)
                        for cls in filtered_class_values
                    ]
                )
            )
            < 2
        ):
            continue

        # Create binary labels: 0 = most normal, 1 = most abnormal
        y_values = np.zeros(len(filtered_class_values), dtype=int)
        for i, class_val in enumerate(filtered_class_values):
            if _extract_parameter_value(class_val, group_name) == most_abnormal_param:
                y_values[i] = 1

        # Count samples for each class
        normal_counts = int(np.sum(y_values == 0))
        abnormal_counts = int(np.sum(y_values == 1))
        fault_ratio = round(abnormal_counts / (normal_counts + abnormal_counts), 4)

        # Create class names mapping
        class_names = {0: most_normal_param, 1: most_abnormal_param}

        # Create ExperimentDataset for this group
        dataset = ExperimentDataset(
            name=f"{group_name}_extremes",
            x_values=filtered_torque_values,
            y_values=y_values,
            experiment_name="binary_for_extremes",
            class_count=2,
            class_names=class_names,
            normal_counts=normal_counts,
            faulty_counts=abnormal_counts,
            faulty_ratio=fault_ratio,
            description=f"Binary classification between parameter extremes in s06 group '{group_name}': {most_normal_param} vs {most_abnormal_param} (control group excluded)",
        )
        datasets.append(dataset)

    # Check if we have any datasets
    if not datasets:
        raise SamplingError(
            f"No valid datasets could be created for binary_for_extremes experiment (control group excluded)"
        )

    return datasets


def _get_multiclass_with_all(torque_values, class_values, scenario_condition):
    """
    Generate dataset for multi-class classification with all s06 classes.

    Note: Skips the control group to focus on parameter variations.
    """

    # Load s06 groups to identify control group
    groups = load_groups("s06")
    control_classes = groups.get("group_control", [])

    # Filter out control group classes
    non_control_mask = ~np.isin(class_values, control_classes)
    filtered_torque_values = torque_values[non_control_mask]
    filtered_class_values = class_values[non_control_mask]
    filtered_condition = scenario_condition[non_control_mask]

    # Get all unique classes (excluding control)
    unique_class_names = sorted(set(filtered_class_values))

    # Apply parameter grouping for duplicates
    parameter_values = [_extract_parameter_value(cls, "") for cls in unique_class_names]
    unique_parameter_values = sorted(set(parameter_values))

    # Build mapping from parameter values to indices
    param_names = {k: v for k, v in enumerate(unique_parameter_values)}
    param_names_to_idx = {v: k for k, v in enumerate(unique_parameter_values)}

    # Convert class names to parameter values, then to indices
    y_values = np.array(
        [
            param_names_to_idx[_extract_parameter_value(cv, "")]
            for cv in filtered_class_values
        ]
    )

    # Calculate normal vs faulty counts for metadata
    normal_mask = filtered_condition == "normal"
    faulty_mask = filtered_condition != "normal"
    normal_counts = int(np.sum(normal_mask))
    faulty_counts = int(np.sum(faulty_mask))
    fault_ratio = round(faulty_counts / (normal_counts + faulty_counts), 4)

    # Create dataset
    dataset = ExperimentDataset(
        name="all_errors",
        x_values=filtered_torque_values,
        y_values=y_values,
        experiment_name="multiclass_with_all",
        class_count=len(unique_parameter_values),
        class_names=param_names,
        normal_counts=normal_counts,
        faulty_counts=faulty_counts,
        faulty_ratio=fault_ratio,
        description=f"Multi-class classification with all {len(unique_parameter_values)} s06 parameter variations (control group excluded, duplicates grouped)",
    )

    return [dataset]
