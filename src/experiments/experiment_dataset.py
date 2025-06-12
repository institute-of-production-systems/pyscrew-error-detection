from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ExperimentDataset:
    """Standardized dataset representation for all experiment types."""

    name: str
    x_values: np.ndarray
    y_values: np.ndarray

    # The three crucial experimental design choices (static configuration)
    scenario_selection: str  # Which dataset (s04/s05/s06)
    sampling_selection: str  # Which sampling strategy
    modeling_selection: str  # Which model set (debug/fast/paper/full)

    # Dataset-specific metadata
    class_count: int
    class_names: dict[int:str]
    normal_counts: int
    faulty_counts: int
    faulty_ratio: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, including None values."""
        return asdict(self)

    def get_tags(self) -> Dict[str, Any]:
        """Returns a dict of dataset tags for logging."""
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in ["name", "x_values", "y_values"]
        }
