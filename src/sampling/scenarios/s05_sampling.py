from typing import List, Tuple

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset
from src.sampling.groups import load_groups
from src.utils.exceptions import DataSamplingError


def _extract_parameter_value(class_name: str) -> str:
    """
    Apply static renaming for duplicate parameter values in s05.

    S05 has several parameter values that appear twice with different IDs
    (e.g., switching-point-15-1 and switching-point-15-2). This function groups
    them into single parameter values for cleaner classification.

    Parameters:
    -----------
    class_name : str
        Original class name from PyScrew dataset

    Returns:
    --------
    str
        Normalized parameter value name
    """
    # Handle known duplicate parameter values with explicit mappings

    # Switching point duplicates: both -1 and -2 variants map to base value
    if (
        class_name == "301_switching-point-15-1"
        or class_name == "306_switching-point-15-2"
    ):
        return "300_switching-point-15"

    # Injection velocity duplicates: both -1 and -2 variants map to base value
    elif (
        class_name == "401_injection-velocity-60-1"
        or class_name == "406_injection-velocity-60-2"
    ):
        return "400_injection-velocity-60"

    # All other classes: return without modification
    else:
        return class_name


def _find_extremes_in_group(group_classes: List[str]) -> Tuple[str, str]:
    """
    Find the most normal and most extreme parameter values in a group.

    Uses alphabetical sorting of normalized parameter values to automatically
    detect extremes. This approach works because parameter values are encoded
    with consistent naming (lower values = more normal, higher = more extreme).

    Parameters:
    -----------
    group_classes : List[str]
        List of original class names in the parameter group

    Returns:
    --------
    Tuple[str, str]
        (most_normal_original_class, most_extreme_original_class)
    """
    # Validate group has sufficient classes for extreme detection
    if len(group_classes) < 2:
        raise DataSamplingError(
            f"Group needs at least 2 classes for extreme detection, got {len(group_classes)}"
        )

    # Sort original classes by their normalized parameter values
    # This automatically finds min/max without hardcoding parameter ranges
    sorted_original_classes = sorted(
        group_classes, key=lambda cls: _extract_parameter_value(cls)
    )

    # Return original class names (not normalized values) for data filtering
    most_normal_original = sorted_original_classes[0]  # Minimum parameter value
    most_extreme_original = sorted_original_classes[-1]  # Maximum parameter value

    return most_normal_original, most_extreme_original


def _get_baseline_parameters() -> dict:
    """
    Define baseline parameter values for s05 (most normal manufacturing conditions).

    Since s05 has no explicit control group, baseline is inferred from
    most conservative/normal parameter values across all groups.

    Returns:
    --------
    dict
        Mapping of group_name to baseline class name
    """
    return {
        "group_1_glass_fiber_content": "107_glass-fiber-content-18",  # Highest fiber content
        "group_2_recyclate_content": "201_recyclate-content-000",  # No recyclate (purest)
        "group_3_switching_point": "302_switching-point-16",  # Middle/standard value
        "group_4_injection_velocity": "401_injection-velocity-60-1",  # Standard velocity
        "group_5_mold_temperature": "503_mold-temperature-40",  # Middle temperature
    }


def sample_s05_combined_all_classes(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Combined + All Classes

    Creates ONE comprehensive dataset with all s05 parameter variations.
    Unlike s06, s05 has no explicit control group, so all classes are treated as separate parameter variations.

    Strategy Details:
    -----------------
    - Scope: ALL parameter groups combined into one dataset
    - Intensity: ALL parameter values across all groups
    - Classes: ~33 classes (all parameter variations)
    - Logic: Comprehensive parameter classification

    What This Creates:
    ------------------
    Comprehensive multi-parameter classification dataset:
    • Class 0: 101_glass-fiber-content-30
    • Class 1: 102_glass-fiber-content-28
    • Class 2: 103_glass-fiber-content-26
    • ...
    • Class 32: 504_mold-temperature-45

    Example Usage:
    --------------
    Input: Torque signal from manufacturing process
    Output: "Glass fiber content 26%" OR "Injection velocity 90" OR "Mold temperature 45°C"
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

    # Create comprehensive multi-parameter dataset
    dataset = ExperimentDataset(
        name="combined_all_classes",
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
        description=f"Combined all classes: All {len(unique_class_names)} {scenario_selection} parameter variations",
    )

    return [dataset]


def sample_s05_combined_only_extreme(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Combined + Only Extreme

    Creates MULTIPLE binary datasets: Each parameter variation vs combined baseline.
    Uses inferred baseline parameters since s05 has no explicit control group.

    Strategy Details:
    -----------------
    - Scope: Each parameter variation vs ALL baseline parameters combined
    - Intensity: ONLY baseline vs individual parameter variations (binary)
    - Datasets: ~28 separate binary datasets (one per non-baseline parameter)
    - Baseline: Inferred most normal values across all groups

    What This Creates:
    ------------------
    Multiple binary anomaly detection datasets:
    • glass-fiber-content-30 vs baseline
    • glass-fiber-content-28 vs baseline
    • glass-fiber-content-26 vs baseline
    • recyclate-content-010 vs baseline
    • recyclate-content-020 vs baseline
    • ...

    Baseline Definition (Inferred):
    ------------------------------
    - glass-fiber-content-18 (highest fiber content)
    - recyclate-content-000 (no recyclate, purest)
    - switching-point-16 (standard setting)
    - injection-velocity-60-1 (standard speed)
    - mold-temperature-40 (middle temperature)

    Example Usage:
    --------------
    Input: Torque signal with unusual behavior
    Output: "Normal baseline operation" OR "Glass fiber content reduced to 26%"
    """
    datasets: List[ExperimentDataset] = []

    # Load groups and get baseline parameters
    groups = load_groups(scenario_selection)
    baseline_parameters = _get_baseline_parameters()

    # Collect all baseline samples across all groups
    baseline_classes = list(baseline_parameters.values())
    baseline_mask = np.isin(class_values, baseline_classes)
    baseline_torque = torque_values[baseline_mask]
    n_baseline = len(baseline_torque)

    # Process each parameter variation separately (excluding baselines)
    for group_name, group_classes in groups.items():
        baseline_class = baseline_parameters.get(group_name)

        for class_value in group_classes:
            # Skip if this is the baseline class
            if class_value == baseline_class:
                continue

            # Filter to get samples for this parameter variation
            class_mask = class_values == class_value
            class_torque = torque_values[class_mask]
            n_variation = len(class_torque)

            # Check for classes with insufficient samples
            if n_baseline == 0 or n_variation == 0:
                continue

            # Combine baseline samples with this parameter variation
            x_combined = np.vstack([baseline_torque, class_torque])

            # Create binary labels (0 for baseline, 1 for parameter variation)
            y_values = np.zeros(len(x_combined), dtype=int)
            y_values[n_baseline:] = 1

            # Create binary parameter-vs-baseline dataset
            dataset = ExperimentDataset(
                name=class_value,
                x_values=x_combined,
                y_values=y_values,
                scenario_selection=scenario_selection,
                sampling_selection=sampling_selection,
                modeling_selection=modeling_selection,
                class_count=2,  # Binary classification
                class_names={0: "baseline", 1: class_value},
                normal_counts=int(n_baseline),
                faulty_counts=int(n_variation),
                faulty_ratio=round(n_variation / (n_baseline + n_variation), 4),
                description=f"Combined only extreme: Baseline parameters vs {class_value}",
            )

            datasets.append(dataset)

    # Validate we created datasets
    if not datasets:
        raise DataSamplingError("No valid datasets created for combined only extreme")

    return datasets


def sample_s05_separate_all_classes(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Separate + All Classes

    Creates FIVE separate datasets, each focusing on all parameter values within one group.
    This enables group-specific parameter analysis and optimization.

    Strategy Details:
    -----------------
    - Scope: EACH parameter group gets its own separate dataset
    - Intensity: ALL parameter values within each group
    - Datasets: 5 separate multiclass datasets
    - Groups duplicate parameters for cleaner classification

    What This Creates:
    ------------------
    Five specialized datasets for group-specific analysis:

    1. **glass_fiber_content_all_classes**: 7 classes
       • glass-fiber-content-30, glass-fiber-content-28, ..., glass-fiber-content-18

    2. **recyclate_content_all_classes**: 11 classes
       • recyclate-content-000, recyclate-content-010, ..., recyclate-content-100

    3. **switching_point_all_classes**: 9 classes (duplicates grouped)
       • switching-point-11, switching-point-12, ..., switching-point-19

    4. **injection_velocity_all_classes**: 9 classes (duplicates grouped)
       • injection-velocity-20, injection-velocity-30, ..., injection-velocity-100

    5. **mold_temperature_all_classes**: 4 classes
       • mold-temperature-30, mold-temperature-35, ..., mold-temperature-45

    Example Usage:
    --------------
    Input: Torque signal from fiber content process
    Output (glass_fiber dataset): "This used glass-fiber-content-26" (vs 6 other fiber levels)
    """
    datasets: List[ExperimentDataset] = []
    groups = load_groups(scenario_selection)

    # Process each parameter group separately
    for group_name, group_classes in groups.items():
        # Apply parameter grouping to handle duplicate values
        # Maps "switching-point-15-1" and "switching-point-15-2" → "switching-point-15"
        parameter_values = [_extract_parameter_value(cls) for cls in group_classes]
        unique_class_names = sorted(set(parameter_values))

        # Create mapping from original class names to normalized parameter values
        class_to_param = {cls: _extract_parameter_value(cls) for cls in group_classes}

        # Filter data to include only this group's classes
        class_group_mask = np.isin(class_values, group_classes)
        filtered_torque_values = torque_values[class_group_mask]
        filtered_class_values = class_values[class_group_mask]
        filtered_condition = scenario_condition[class_group_mask]

        # Skip groups with no data
        if len(filtered_torque_values) == 0:
            continue

        # Calculate normal vs faulty distribution for this group
        normal_mask = filtered_condition == "normal"
        faulty_mask = filtered_condition != "normal"
        normal_counts = int(np.sum(normal_mask))
        faulty_counts = int(np.sum(faulty_mask))
        fault_ratio = round(faulty_counts / (faulty_counts + normal_counts), 4)

        # Build class index mapping: parameter_value → class_index
        class_names = {k: v for k, v in enumerate(unique_class_names)}
        class_names_to_idx = {v: k for k, v in enumerate(unique_class_names)}

        # Convert original class names to parameter values, then to class indices
        y_values = np.array(
            [class_names_to_idx[class_to_param[fcv]] for fcv in filtered_class_values]
        )

        # Create group-specific multiclass dataset
        dataset = ExperimentDataset(
            name=f"{group_name}_all_classes",
            x_values=filtered_torque_values,
            y_values=y_values,
            scenario_selection=scenario_selection,
            sampling_selection=sampling_selection,
            modeling_selection=modeling_selection,
            class_count=len(unique_class_names),
            class_names=class_names,
            normal_counts=normal_counts,
            faulty_counts=faulty_counts,
            faulty_ratio=fault_ratio,
            description=f"Separate all classes: All {len(unique_class_names)} parameter values within {group_name} (duplicates grouped)",
        )
        datasets.append(dataset)

    # Validate we created datasets
    if not datasets:
        raise DataSamplingError("No valid datasets created for separate all classes")

    return datasets


def sample_s05_separate_only_extreme(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Separate + Only Extreme

    Creates FIVE separate binary datasets: Most normal vs most extreme parameter within each group.
    This enables group-specific anomaly detection and go/no-go decisions.

    Strategy Details:
    -----------------
    - Scope: EACH parameter group gets its own separate dataset
    - Intensity: ONLY the most extreme parameter value within each group
    - Datasets: 5 separate binary datasets
    - Selection: Automatic extreme detection via sorting

    What This Creates:
    ------------------
    Five focused binary datasets for group-specific anomaly detection:

    1. **glass_fiber_content_only_extreme**: glass-fiber-content-30 vs glass-fiber-content-18
    2. **recyclate_content_only_extreme**: recyclate-content-000 vs recyclate-content-100
    3. **switching_point_only_extreme**: switching-point-11 vs switching-point-19
    4. **injection_velocity_only_extreme**: injection-velocity-20 vs injection-velocity-100
    5. **mold_temperature_only_extreme**: mold-temperature-30 vs mold-temperature-45

    Example Usage:
    --------------
    Input: Torque signal from recyclate process
    Output (recyclate dataset): "Low recyclate (0%)" OR "High recyclate (100%)"
    """
    datasets: List[ExperimentDataset] = []
    groups = load_groups(scenario_selection)

    # Process each parameter group separately for binary extreme detection
    for group_name, group_classes in groups.items():
        try:
            # Find most normal and most extreme parameters in this group using automatic sorting
            most_normal_original, most_extreme_original = _find_extremes_in_group(
                group_classes
            )
        except DataSamplingError:
            # Skip groups where extreme detection fails
            continue

        # Filter data for only these two extreme classes
        extreme_classes = [most_normal_original, most_extreme_original]
        extreme_mask = np.isin(class_values, extreme_classes)
        filtered_torque_values = torque_values[extreme_mask]
        filtered_class_values = class_values[extreme_mask]
        filtered_condition = scenario_condition[extreme_mask]

        # Skip if we don't have both extremes
        if len(set(filtered_class_values)) < 2:
            continue

        # Create binary labels: 0 = most normal, 1 = most extreme
        y_values = np.zeros(len(filtered_class_values), dtype=int)
        most_extreme_mask = filtered_class_values == most_extreme_original
        y_values[most_extreme_mask] = 1

        # Calculate binary distribution
        normal_counts = int(np.sum(y_values == 0))
        extreme_counts = int(np.sum(y_values == 1))
        fault_ratio = round(extreme_counts / (normal_counts + extreme_counts), 4)

        # Create class names with normalized parameter names
        most_normal_param = _extract_parameter_value(most_normal_original)
        most_extreme_param = _extract_parameter_value(most_extreme_original)
        class_names = {0: most_normal_param, 1: most_extreme_param}

        # Create group-specific binary extreme detection dataset
        dataset = ExperimentDataset(
            name=f"{group_name}_only_extreme",
            x_values=filtered_torque_values,
            y_values=y_values,
            scenario_selection=scenario_selection,
            sampling_selection=sampling_selection,
            modeling_selection=modeling_selection,
            class_count=2,
            class_names=class_names,
            normal_counts=normal_counts,
            faulty_counts=extreme_counts,
            faulty_ratio=fault_ratio,
            description=f"Separate only extreme: {most_normal_param} vs {most_extreme_param} in {group_name}",
        )
        datasets.append(dataset)

    # Validate we created datasets
    if not datasets:
        raise DataSamplingError("No valid datasets created for separate only extreme")

    return datasets
