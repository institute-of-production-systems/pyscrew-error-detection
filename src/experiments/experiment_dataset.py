from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ExperimentDataset:
    """Standardized dataset representation for all experiment types."""

    name: str

    x_values: np.ndarray
    y_values: np.ndarray

    sampling_selection: str
    class_count: int
    class_names: dict[int:str]
    normal_counts: int
    faulty_counts: int
    faulty_ratio: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, inlcuding None values."""
        return asdict(self)
