from typing import List, Tuple

import numpy as np

from src.experiments.experiment_dataset import ExperimentDataset
from src.sampling.groups import load_groups
from src.utils.exceptions import DataSamplingError


def _extract_parameter_value(class_name: str) -> str:
    """
    Apply static renaming for duplicate parameter values in s06.

    S06 has several parameter values that appear twice with different IDs
    (e.g., cooling-time-25-1 and cooling-time-25-2). This function groups
    them into single parameter values for cleaner classification.

    Parameters:
    -----------
    class_name : str
        Original class name from PyScrew dataset
    group_name : str
        Parameter group name (for context, currently unused)

    Returns:
    --------
    str
        Normalized parameter value name
    """
    # Handle known duplicate parameter values with explicit mappings

    # Cooling time duplicates: both -1 and -2 variants map to base value
    if class_name == "101_cooling-time-25-1" or class_name == "107_cooling-time-25-2":
        return "100_cooling-time-25"

    # Switching point duplicates: both -1 and -2 variants map to base value
    elif (
        class_name == "401_switching-point-22-1"
        or class_name == "409_switching-point-22-2"
    ):
        return "400_switching-point-22"

    # Injection velocity duplicates: both -1 and -2 variants map to base value
    elif (
        class_name == "501_injection-velocity-030-1"
        or class_name == "509_injection-velocity-030-2"
    ):
        return "500_injection-velocity-030"

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


def _normalize_class_name(class_name: str) -> str:
    """
    Direct normalization: Handle duplicates AND group defaults in one step.

    Maps original PyScrew class names to normalized values with two key transformations:
    1. Groups all default parameter values into single "000_control_group" class
    2. Handles duplicate parameter values (e.g., -1 and -2 variants)

    Parameters:
    -----------
    class_name : str
        Original class name from PyScrew dataset

    Returns:
    --------
    str
        Normalized class name
    """
    # DEFAULT PARAMETERS: Group all baseline manufacturing conditions
    # These represent standard/normal operation across all parameter groups
    if class_name in [
        # Default cooling time (25s) - both duplicate variants
        "101_cooling-time-25-1",
        "107_cooling-time-25-2",
        # Default mold temperature (30°C)
        "201_mold-temperature-30",
        # Default glass fiber content (30%)
        "301_glass-fiber-content-30",
        # Default switching point (22) - both duplicate variants
        "401_switching-point-22-1",
        "409_switching-point-22-2",
        # Default injection velocity (030) - both duplicate variants
        "501_injection-velocity-030-1",
        "509_injection-velocity-030-2",
    ]:
        return "000_control_group"

    # ALL OTHER CLASSES: Keep original name (these are actual parameter variations)
    else:
        return class_name


def sample_s06_combined_all_classes(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Combined + All Classes

    Creates ONE comprehensive dataset distinguishing between normal operation and parameter variations.
    Groups all default parameter values as "normal" and treats parameter deviations as separate classes.

    Strategy Details:
    -----------------
    - Scope: ALL parameter groups combined into one dataset
    - Intensity: ALL parameter values, with defaults grouped as "normal"
    - Classes: 1 normal class + ~20 parameter variation classes
    - Logic: Normal operation vs specific parameter deviations

    What This Creates:
    ------------------
    Comprehensive anomaly classification dataset:
    • Class 0: 000_control_group (ALL default parameter combinations)
    • Class 1: 102_cooling-time-26 (cooling variation)
    • Class 2: 103_cooling-time-27 (cooling variation)
    • Class 3: 202_mold-temperature-35 (temperature variation)
    • Class 4: 203_mold-temperature-45 (temperature variation)
    • ...
    • Class 20: 512_injection-velocity-170 (velocity variation)

    Default Parameters Grouped as "Normal":
    ---------------------------------------
    - cooling-time-25 (baseline cooling)
    - mold-temperature-30 (baseline temperature)
    - glass-fiber-content-30 (baseline fiber content)
    - switching-point-22 (baseline switching)
    - injection-velocity-030 (baseline injection speed)

    Example Usage:
    --------------
    Input: Torque signal from manufacturing process
    Output: "Normal operation" OR "Temperature increased to 45°C" OR "Fiber content reduced to 20%"
    """
    # Load parameter groups (we'll process all classes, not excluding control)
    groups = load_groups(scenario_selection)

    # Apply direct normalization: defaults→control_group, duplicates→normalized, others→as-is
    normalized_class_values = np.array(
        [_normalize_class_name(class_val) for class_val in class_values]
    )

    # Get all unique normalized classes
    unique_normalized_classes = sorted(set(normalized_class_values))

    # Build class index mapping: normalized_class → class_index
    # Ensure control group gets index 0 for interpretability
    class_names = {}
    class_names_to_idx = {}

    if "000_control_group" in unique_normalized_classes:
        # Put control group first (index 0)
        class_names[0] = "000_control_group"
        class_names_to_idx["000_control_group"] = 0

        # Add all other classes starting from index 1
        other_classes = [
            cls for cls in unique_normalized_classes if cls != "000_control_group"
        ]
        for i, cls in enumerate(other_classes, start=1):
            class_names[i] = cls
            class_names_to_idx[cls] = i
    else:
        # Fallback: no control group found, use standard indexing
        for i, cls in enumerate(unique_normalized_classes):
            class_names[i] = cls
            class_names_to_idx[cls] = i

    # Convert normalized class names to class indices
    y_values = np.array(
        [
            class_names_to_idx[normalized_class]
            for normalized_class in normalized_class_values
        ]
    )

    # Calculate normal vs parameter-variation distribution
    # Control group samples represent normal operation
    control_mask = normalized_class_values == "000_control_group"
    variation_mask = ~control_mask
    normal_counts = int(np.sum(control_mask))
    variation_counts = int(np.sum(variation_mask))

    # Calculate fault ratio based on scenario_condition for metadata consistency
    scenario_normal_mask = scenario_condition == "normal"
    scenario_faulty_mask = scenario_condition != "normal"
    scenario_normal_counts = int(np.sum(scenario_normal_mask))
    scenario_faulty_counts = int(np.sum(scenario_faulty_mask))
    fault_ratio = round(
        scenario_faulty_counts / (scenario_normal_counts + scenario_faulty_counts), 4
    )

    # Create comprehensive normal-vs-variations dataset
    dataset = ExperimentDataset(
        name="combined_all_classes",
        x_values=torque_values,
        y_values=y_values,
        scenario_selection=scenario_selection,
        sampling_selection=sampling_selection,
        modeling_selection=modeling_selection,
        class_count=len(unique_normalized_classes),
        class_names=class_names,
        normal_counts=normal_counts,  # Based on control group classification
        faulty_counts=variation_counts,  # Based on parameter variations
        faulty_ratio=fault_ratio,  # Based on original scenario conditions
        description=f"Combined all classes: 1 control group + {len(unique_normalized_classes)-1} parameter variations (defaults grouped as normal)",
    )

    return [dataset]


def sample_s06_combined_only_extreme(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Combined + Only Extreme

    Creates ONE strategic dataset with 6 classes: Normal + most extreme parameter from each group.
    This focuses on the most critical parameter deviations across the manufacturing process.

    Strategy Details:
    -----------------
    - Scope: ALL parameter groups combined into one dataset
    - Intensity: ONLY the most extreme parameter value from each group
    - Classes: Exactly 6 (normal + 5 extreme parameters)
    - Selection: Automatic detection via sorting (highest value = most extreme)

    What This Creates:
    ------------------
    Strategic 6-class dataset focusing on critical deviations:
    • Class 0: normal (baseline manufacturing conditions)
    • Class 1: cooling-time-30 (longest cooling time)
    • Class 2: mold-temperature-65 (highest temperature)
    • Class 3: glass-fiber-content-10 (lowest fiber content)
    • Class 4: switching-point-32 (highest switching point)
    • Class 5: injection-velocity-170 (fastest injection speed)

    Example Usage:
    --------------
    Input: Torque signal showing unusual behavior
    Output: "Normal operation" OR "Injection velocity reached extreme (170)"
    """
    # Load parameter groups and find most extreme parameter from each
    groups = load_groups(scenario_selection)

    # Identify the most extreme original class from each parameter group
    strongest_classes = {}  # Maps original_class_name → group_index
    group_index = 1  # Start from 1, reserve 0 for normal samples

    for group_name, group_classes in groups.items():
        # Skip control group - focus on parameter variations only
        if group_name == "group_control":
            continue

        try:
            # Find most extreme parameter in this group using sorting
            _, most_extreme_original = _find_extremes_in_group(group_classes)
            strongest_classes[most_extreme_original] = group_index
            group_index += 1
        except DataSamplingError:
            # Skip groups where extreme detection fails
            continue

    # Validate we found extreme parameters
    if not strongest_classes:
        raise DataSamplingError("No extreme parameters found across any groups")

    # Filter data: keep normal samples + most extreme from each group
    normal_mask = scenario_condition == "normal"
    strongest_mask = np.isin(class_values, list(strongest_classes.keys()))
    combined_mask = normal_mask | strongest_mask

    filtered_torque_values = torque_values[combined_mask]
    filtered_class_values = class_values[combined_mask]
    filtered_condition = scenario_condition[combined_mask]

    # Create labels: 0=normal, 1-5=most extreme from each group
    y_values = np.zeros(len(filtered_condition), dtype=int)
    for i, (class_val, condition) in enumerate(
        zip(filtered_class_values, filtered_condition)
    ):
        if condition == "normal":
            y_values[i] = 0  # All normal samples get class 0
        else:
            y_values[i] = strongest_classes[
                class_val
            ]  # Group's extreme gets group index

    # Calculate normal vs extreme distribution
    normal_counts = int(np.sum(y_values == 0))
    extreme_counts = len(y_values) - normal_counts
    fault_ratio = round(extreme_counts / len(y_values), 4)

    # Build class names mapping with normalized parameter names
    class_names = {0: "normal"}
    for original_class, group_idx in strongest_classes.items():
        normalized_param = _extract_parameter_value(original_class)
        class_names[group_idx] = normalized_param

    # Create focused extreme-parameter dataset
    dataset = ExperimentDataset(
        name="combined_only_extreme",
        x_values=filtered_torque_values,
        y_values=y_values,
        scenario_selection=scenario_selection,
        sampling_selection=sampling_selection,
        modeling_selection=modeling_selection,
        class_count=len(class_names),
        class_names=class_names,
        normal_counts=normal_counts,
        faulty_counts=extreme_counts,
        faulty_ratio=fault_ratio,
        description=f"Combined only extreme: Normal + most extreme from each group ({len(strongest_classes)} extreme parameters)",
    )

    return [dataset]


def sample_s06_separate_all_classes(
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

    1. **cooling_time_all_classes**: 6 classes
       • cooling-time-25, cooling-time-26, ..., cooling-time-30

    2. **mold_temperature_all_classes**: 5 classes
       • mold-temperature-30, mold-temperature-35, ..., mold-temperature-65

    3. **glass_fiber_content_all_classes**: 5 classes
       • glass-fiber-content-10, glass-fiber-content-15, ..., glass-fiber-content-30

    4. **switching_point_all_classes**: 13 classes
       • switching-point-13, switching-point-14, ..., switching-point-32

    5. **injection_velocity_all_classes**: 10 classes
       • injection-velocity-015, injection-velocity-020, ..., injection-velocity-170

    Example Usage:
    --------------
    Input: Torque signal from cooling process
    Output (cooling_time dataset): "This used cooling-time-28" (vs 5 other cooling times)
    """
    datasets: List[ExperimentDataset] = []
    groups = load_groups(scenario_selection)

    # Process each parameter group separately
    for group_name, group_classes in groups.items():
        # Skip control group - focus on parameter variations only
        if group_name == "group_control":
            continue

        # Apply parameter grouping to handle duplicate values
        # Maps "cooling-time-25-1" and "cooling-time-25-2" → "cooling-time-25"
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


def sample_s06_separate_only_extreme(
    torque_values: np.ndarray,
    class_values: np.ndarray,
    scenario_condition: np.ndarray,
    scenario_selection: str,
    sampling_selection: str,
    modeling_selection: str,
) -> List[ExperimentDataset]:
    """
    SAMPLING MATRIX POSITION: Separate + Only Extreme

    Creates FIVE separate binary datasets: Normal vs most extreme parameter within each group.
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

    1. **cooling_time_only_extreme**: Normal vs cooling-time-30
    2. **mold_temperature_only_extreme**: Normal vs mold-temperature-65
    3. **glass_fiber_content_only_extreme**: Normal vs glass-fiber-content-10
    4. **switching_point_only_extreme**: Normal vs switching-point-32
    5. **injection_velocity_only_extreme**: Normal vs injection-velocity-170

    Example Usage:
    --------------
    Input: Torque signal from temperature control process
    Output (mold_temperature dataset): "Normal" OR "Extreme temperature (65°C detected)"
    """
    datasets: List[ExperimentDataset] = []
    groups = load_groups(scenario_selection)

    # Process each parameter group separately for binary extreme detection
    for group_name, group_classes in groups.items():
        # Skip control group - focus on parameter variations only
        if group_name == "group_control":
            continue

        try:
            # Find most extreme parameter in this group using automatic sorting
            _, most_extreme_original = _find_extremes_in_group(group_classes)
        except DataSamplingError:
            # Skip groups where extreme detection fails
            continue

        # Create binary dataset: normal samples vs most extreme parameter in group
        normal_mask = scenario_condition == "normal"
        extreme_mask = class_values == most_extreme_original

        # Combine normal and extreme samples for binary classification
        combined_mask = normal_mask | extreme_mask
        filtered_torque_values = torque_values[combined_mask]
        filtered_condition = scenario_condition[combined_mask]

        # Create binary labels: 0=normal, 1=most extreme in this group
        y_values = np.zeros(len(filtered_condition), dtype=int)
        extreme_indices = filtered_condition != "normal"
        y_values[extreme_indices] = 1

        # Calculate binary distribution
        normal_counts = int(np.sum(y_values == 0))
        extreme_counts = int(np.sum(y_values == 1))
        fault_ratio = round(extreme_counts / (normal_counts + extreme_counts), 4)

        # Create class names with normalized parameter name
        most_extreme_param = _extract_parameter_value(most_extreme_original)
        class_names = {0: "normal", 1: most_extreme_param}

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
            description=f"Separate only extreme: Normal vs most extreme in {group_name} ({most_extreme_param})",
        )
        datasets.append(dataset)

    # Validate we created datasets
    if not datasets:
        raise DataSamplingError("No valid datasets created for separate only extreme")

    return datasets
