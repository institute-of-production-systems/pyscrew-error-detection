from typing import List

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset
from src.sampling.groups import load_groups


def sample_s05_binary_for_extremes(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """Generate datasets for binary classification between parameter extremes in each s05 group."""
    datasets: List[ExperimentDataset] = []

    # Load groups from JSON using scenario parameter
    groups = load_groups(scenario_selection)

    for group_name, group_classes in groups.items():
        # Get sorted unique classes to find extremes
        unique_class_names = sorted(set(group_classes))

        # Take first (most normal) and last (most abnormal)
        most_normal_class = unique_class_names[0]
        most_abnormal_class = unique_class_names[-1]
        extreme_classes = [most_normal_class, most_abnormal_class]

        # Filter data for only these two extreme classes
        extreme_mask = np.isin(class_values, extreme_classes)
        filtered_torque_values = torque_values[extreme_mask]
        filtered_class_values = class_values[extreme_mask]
        filtered_condition = scenario_condition[extreme_mask]

        # Skipping if there are less then two classes
        if len(set(filtered_class_values)) < 2:
            continue

        # Create binary labels: 0 = most normal, 1 = most abnormal
        y_values = np.zeros(len(filtered_class_values), dtype=int)
        most_abnormal_mask = filtered_class_values == most_abnormal_class
        y_values[most_abnormal_mask] = 1

        # Count samples for each class
        normal_counts = int(np.sum(y_values == 0))
        abnormal_counts = int(np.sum(y_values == 1))
        fault_ratio = round(abnormal_counts / (normal_counts + abnormal_counts), 4)

        # Create class names mapping
        class_names = {0: most_normal_class, 1: most_abnormal_class}

        # Create ExperimentDataset for this group
        dataset = ExperimentDataset(
            name=f"{group_name}_extremes",
            x_values=filtered_torque_values,
            y_values=y_values,
            scenario_selection=scenario_selection,
            sampling_selection=sampling_selection,
            modeling_selection=modeling_selection,
            class_count=2,
            class_names=class_names,
            normal_counts=normal_counts,
            faulty_counts=abnormal_counts,
            faulty_ratio=fault_ratio,
            description=f"Binary classification between parameter extremes in {scenario_selection} group '{group_name}': {most_normal_class} vs {most_abnormal_class}",
        )
        datasets.append(dataset)

    return datasets


def sample_s05_multiclass_within_groups(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """Generate datasets for multi-class classification within s05 error groups."""

    # Configuration: Group duplicate parameter values (e.g., switching-point-15-1 and switching-point-15-2)
    GROUP_DUPLICATE_PARAMETERS = True  # TODO: discuss with Marco

    datasets: List[ExperimentDataset] = []

    # Load groups from JSON using scenario parameter
    groups = load_groups(scenario_selection)

    def _extract_parameter_value(class_name, group_name):
        """
        Apply simple static renaming for duplicate parameter values.

        Explicit mappings for known duplicates:
        - "301_switching-point-15-1" -> "300_switching-point-15"
        - "306_switching-point-15-2" -> "300_switching-point-15" (same!)
        - "401_injection-velocity-60-1" -> "400_injection-velocity-60"
        - "406_injection-velocity-60-2" -> "400_injection-velocity-60" (same!)
        """

        # Static renaming for switching point duplicates
        if (
            class_name == "301_switching-point-15-1"
            or class_name == "306_switching-point-15-2"
        ):
            return "300_switching-point-15"

        # Static renaming for injection velocity duplicates
        elif (
            class_name == "401_injection-velocity-60-1"
            or class_name == "406_injection-velocity-60-2"
        ):
            return "400_injection-velocity-60"

        # For all other classes, return as-is
        else:
            return class_name

    for group_name, group_classes in groups.items():

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

        # Get indices of normal and faulty samples and get counts
        normal_mask = filtered_condition == "normal"
        faulty_mask = filtered_condition != "normal"  # == "faulty"
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
            scenario_selection=scenario_selection,
            sampling_selection=sampling_selection,
            modeling_selection=modeling_selection,
            class_count=len(unique_class_names),
            class_names=class_names,
            normal_counts=normal_counts,
            faulty_counts=faulty_counts,
            faulty_ratio=fault_ratio,
            description=f"Multi-class classification within {scenario_selection} group '{group_name}'"
            + (
                f" (grouped duplicate parameters)" if GROUP_DUPLICATE_PARAMETERS else ""
            ),
        )
        datasets.append(dataset)

    return datasets


def sample_s05_multiclass_with_all(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    Generate dataset for multi-class classification with all s05 classes.

    TODO: this function does not yet account for "normal" and "faulty" observations,
    meaning the first classes (and sometimes some in-between classes) have the default
    configuration and should behave the same. To account for that, they should be made
    into the same class "000_control-group" or something similar.
    """

    # Get all unique classes across all groups
    unique_class_names = sorted(set(class_values))

    # Build mapping from class names to indices
    class_names = {k: v for k, v in enumerate(unique_class_names)}
    class_names_to_idx = {v: k for k, v in enumerate(unique_class_names)}

    # Convert class names to indices
    y_values = np.array([class_names_to_idx[cv] for cv in class_values])

    # Calculate normal vs faulty counts for metadata
    normal_mask = scenario_condition == "normal"
    faulty_mask = scenario_condition != "normal"
    normal_counts = int(np.sum(normal_mask))
    faulty_counts = int(np.sum(faulty_mask))
    fault_ratio = round(faulty_counts / (normal_counts + faulty_counts), 4)

    # Create dataset
    dataset = ExperimentDataset(
        name="all_errors",
        x_values=torque_values,
        y_values=y_values,
        scenario_selection=scenario_selection,
        sampling_selection=sampling_selection,
        modeling_selection=modeling_selection,
        class_count=len(unique_class_names),
        class_names=class_names,
        normal_counts=normal_counts,
        faulty_counts=faulty_counts,
        faulty_ratio=fault_ratio,
        description=f"Multi-class classification with all {len(unique_class_names)} {scenario_selection} parameter variations",
    )

    return [dataset]
