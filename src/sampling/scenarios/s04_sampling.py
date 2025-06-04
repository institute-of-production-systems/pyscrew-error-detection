import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.utils.exceptions import SamplingError

from ...experiments.experiment_dataset import ExperimentDataset


def get_experiment_functions(
    experiment_names: List[str],
) -> Dict[
    str, Callable[[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[ExperimentDataset]]
]:
    """
    Returns dict of lazy functions for s04 experiments.

    TODO: Prepared to be called with a list, even though the experiment runner
    currently just calls it with a single experiment name. Might expand or simplify
    this in the future.
    """

    functions = {}
    for exp_name in experiment_names:
        if exp_name == "binary_vs_ref":
            functions[exp_name] = lambda processed_data: _get_binary_vs_ref_data(
                *processed_data
            )
        elif exp_name == "binary_vs_all":
            functions[exp_name] = lambda processed_data: _get_binary_vs_all_data(
                *processed_data
            )
        elif exp_name == "multiclass_with_groups":
            functions[exp_name] = lambda processed_data: _get_multiclass_with_groups(
                *processed_data, "s04"
            )
        elif exp_name == "multiclass_with_all":
            functions[exp_name] = lambda processed_data: _get_multiclass_with_all(
                *processed_data
            )
        else:
            raise ValueError(f"Unknown experiment type: {exp_name}")

    return functions


def _get_binary_vs_ref_data(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[ExperimentDataset]:
    """Generate datasets for binary classification of errors vs reference in one class."""

    # Initialize a list to return the experiment datasets
    datasets: List[ExperimentDataset] = []

    # Get a sorted list of all unique class values in the experiment
    unique_class_names = [str(class_name) for class_name in sorted(set(class_values))]

    # Iterate the class names (which now become the dataset names)
    for dataset_name in unique_class_names:

        # Filter data for just this class
        class_mask = class_values == dataset_name
        x_values = torque_values[class_mask]
        filtered_condition = scenario_condition[class_mask]

        # Get indices of normal and faulty samples and get counts
        normal_mask = filtered_condition == "normal"
        faulty_mask = filtered_condition != "normal"  # == "faulty"
        normal_counts = int(np.sum(normal_mask))
        faulty_counts = int(np.sum(faulty_mask))

        # Create binary labels (0 for normal, 1 for faulty) for easier modeling
        y_values = np.zeros(len(filtered_condition), dtype=int)
        y_values[faulty_mask] = 1

        # Create dataset using standardized format
        dataset = ExperimentDataset(
            name=dataset_name,
            x_values=x_values,
            y_values=y_values,
            experiment_name="binary_vs_ref",
            class_count=2,  # Binary classification
            class_names={0: "normal", 1: "faulty"},
            normal_counts=normal_counts,
            faulty_counts=faulty_counts,
            faulty_ratio=round(faulty_counts / (faulty_counts + normal_counts), 4),
            description=f"Binary classification of {normal_counts} normal and {faulty_counts} faulty samples for class '{dataset_name}'",
        )

        datasets.append(dataset)

    # Check if we have any datasets
    if not datasets:
        raise SamplingError(
            f"No valid datasets could be created for binary_vs_ref experiment"
        )

    return datasets


def _get_binary_vs_all_data(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[ExperimentDataset]:
    """Generate datasets for binary classification comparing each class's faulty samples vs ALL normal samples."""
    datasets = []

    # Find all normal samples across all classes in the scenario
    normal_mask = scenario_condition == "normal"
    normal_indices = np.where(normal_mask)[0]
    normal_torque = torque_values[normal_indices]
    n_normal = len(normal_indices)

    # Process each class separately for its faulty samples
    for class_value in sorted(set(class_values)):
        # Filter to get only faulty samples for this class
        class_mask = class_values == class_value
        class_condition = scenario_condition[class_mask]
        faulty_mask = class_condition != "normal"
        n_faulty = np.sum(faulty_mask)

        # Check for classes with insufficient samples
        if n_normal == 0 or n_faulty == 0:
            raise SamplingError(
                f"Could not create vs_all dataset for class {class_value} due to missing samples "
                f"(n_normal={n_normal}, n_faulty={n_faulty})"
            )

        # Get faulty samples for this class
        class_torque = torque_values[class_mask]
        faulty_torque = class_torque[faulty_mask]

        # Combine normal samples from all classes with faulty samples from this class
        x_combined = np.vstack([normal_torque, faulty_torque])

        # Create binary labels (0 for normal, 1 for faulty)
        y_values = np.zeros(len(x_combined), dtype=int)
        y_values[n_normal:] = 1

        # Create dataset using standardized format
        dataset = ExperimentDataset(
            name=class_value,
            x_values=x_combined,
            y_values=y_values,
            experiment_name="binary_vs_all",
            class_count=2,  # Binary classification
            class_names={0: "normal", 1: "faulty"},
            normal_counts=int(n_normal),
            faulty_counts=int(n_faulty),
            faulty_ratio=round(n_faulty / (n_normal + n_faulty), 4),
            description=f"Binary classification of ALL normal samples vs faulty samples for class {class_value}",
        )

        datasets.append(dataset)

    return datasets


def _get_multiclass_with_groups(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_id: str,
) -> List[ExperimentDataset]:
    """Generate datasets for multi-class classification within error groups."""
    datasets = []

    # Load error groups from JSON file based on scenario ID
    groups_file = Path(f"src/sampling/groups/{scenario_id}.json")

    # Ensure directory exists
    os.makedirs(groups_file.parent, exist_ok=True)

    # Check if file exists, if not provide helpful error
    if not groups_file.exists():
        raise FileNotFoundError(
            f"Error groups configuration file not found at {groups_file}. "
            f"Please create the file with appropriate error groupings for scenario {scenario_id}."
        )

    # Load error groups from JSON file
    with open(groups_file, "r") as f:
        error_groups = json.load(f)

    # Process each group
    for group_name, group_errors in error_groups.items():
        # Filter data for this group's classes
        class_mask = np.isin(class_values, group_errors)
        filtered_torque_values = torque_values[class_mask]
        filtered_class_values = class_values[class_mask]
        filtered_condition = scenario_condition[class_mask]

        # Create mapping for class labels:
        # 0 = normal (regardless of class)
        # 1-5 = faulty samples from each class
        class_mapping = {}
        for i, class_val in enumerate(sorted(group_errors)):
            class_mapping[class_val] = i + 1  # Faulty samples get 1-5

        # Create reverse mapping for readability
        class_names = {0: "normal"}  # Start with normal class
        for class_val, idx in class_mapping.items():
            class_names[idx] = f"faulty_{class_val}"

        # Initialize y_values with zeros (normal)
        y_values = np.zeros(len(filtered_torque_values), dtype=int)

        # Set faulty samples to their respective class values
        for i in range(len(filtered_torque_values)):
            if filtered_condition[i] != "normal":
                y_values[i] = class_mapping[filtered_class_values[i]]

        # Calculate normal and faulty counts
        n_normal = int(np.sum(y_values == 0))
        n_faulty = len(y_values) - n_normal

        # Skip if we don't have both normal and faulty samples
        if n_normal == 0 or n_faulty == 0:
            continue

        # Create dataset using standardized format
        dataset = ExperimentDataset(
            name=group_name,
            x_values=filtered_torque_values,
            y_values=y_values,
            experiment_name="multiclass_with_groups",
            class_count=len(group_errors) + 1,  # Normal + all classes in group
            class_names=class_names,
            normal_counts=n_normal,
            faulty_counts=n_faulty,
            faulty_ratio=round(n_faulty / (n_normal + n_faulty), 4),
            description=f"Multi-class classification within error group '{group_name}' (0=normal, 1-{len(group_errors)}=faulty class)",
        )

        datasets.append(dataset)

    # Check if we have any datasets
    if not datasets:
        raise SamplingError(
            f"No valid datasets could be created for multiclass_group experiment"
        )

    return datasets


def _get_multiclass_with_all(
    torque_values: np.ndarray, class_values: np.ndarray, scenario_condition: np.ndarray
) -> List[ExperimentDataset]:
    """Generate dataset for multi-class classification with one class for normals and N classes for errors."""
    # Map class values to integers (1-25 or however many classes)
    unique_classes = sorted(set(class_values))
    class_mapping = {class_val: idx + 1 for idx, class_val in enumerate(unique_classes)}

    # Check that each class has samples
    for class_val in unique_classes:
        class_mask = class_values == class_val
        normal_mask = (class_values == class_val) & (scenario_condition == "normal")
        faulty_mask = (class_values == class_val) & (scenario_condition != "normal")
        n_normal = np.sum(normal_mask)
        n_faulty = np.sum(faulty_mask)

        # Check for classes with insufficient samples
        if n_normal == 0 or n_faulty == 0:
            raise SamplingError(
                f"Could not create multiclass_all dataset: class {class_val} has missing samples "
                f"(n_normal={n_normal}, n_faulty={n_faulty})"
            )

    # Create reverse mapping for readability
    class_names = {0: "normal"}  # Start with normal class
    for class_val, idx in class_mapping.items():
        class_names[idx] = class_val

    # Create multiclass labels (0 for normal, 1-N for error classes)
    y_values = np.zeros(len(torque_values), dtype=int)
    for i in range(len(torque_values)):
        if scenario_condition[i] != "normal":
            y_values[i] = class_mapping[class_values[i]]

    # Calculate normal vs faulty ratio
    n_normal = int(np.sum(y_values == 0))
    n_faulty = len(y_values) - n_normal

    # Final check for at least some samples in both categories
    if n_normal == 0 or n_faulty == 0:
        raise SamplingError(
            f"Could not create multiclass_all dataset: missing samples "
            f"(n_normal={n_normal}, n_faulty={n_faulty})"
        )

    # Create dataset using standardized format
    dataset = ExperimentDataset(
        name="all_errors",
        x_values=torque_values,
        y_values=y_values,
        experiment_name="multiclass_with_all",
        class_count=len(unique_classes) + 1,  # Normal + all error classes
        class_names=class_names,
        normal_counts=n_normal,
        faulty_counts=n_faulty,
        faulty_ratio=round(n_faulty / (n_normal + n_faulty), 4),
        description=f"Multi-class classification with class 0 for normal samples and {len(unique_classes)} classes for different error types",
    )

    return [dataset]  # Return as a list for consistent interface


"""
TODO: 
- In a future release, this should be moved to a separate module sampling/
- This way, we can start to implement s05 and s06 as well 
- The scenarios are currently not compatible because s04 has normal and faulty in a single class
- For all other scenarios, we can use the "is normal" tag in the class yml files (pyscrew)
"""
