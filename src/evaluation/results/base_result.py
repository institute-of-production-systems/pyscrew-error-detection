from abc import ABC
from datetime import datetime
from typing import Any, Dict, Optional, Union


class BaseResult(ABC):
    """
    Base class for all result types with static configuration inheritance.

    Provides the three core configuration tags (scenario, sampling, modeling) that
    inherit down the 4-level MLflow hierarchy, enabling cross-experiment queries.
    Also tracks end-to-end pipeline runtime to complement MLflow's per-run timing.

    The inheritance hierarchy:
    BaseResult -> ExperimentResult -> DatasetResult -> ModelResult -> FoldResult

    All subclasses inherit the core configuration and timing tags.
    """

    def __init__(
        self, scenario_selection: str, sampling_selection: str, modeling_selection: str
    ):
        """
        Initialize with the three core experimental design choices.

        Parameters:
        -----------
        scenario_selection : str
            PyScrew dataset to use (e.g., "s04", "s05", "s06")
        sampling_selection : str
            Sampling strategy (e.g., "binary_vs_ref", "multiclass_with_groups")
        modeling_selection : str
            Model set to evaluate (e.g., "debug", "fast", "paper", "full")
        """
        # Simple validation
        if not all([scenario_selection, sampling_selection, modeling_selection]):
            raise ValueError("All selection parameters must be non-empty strings")

        # Core experimental configuration - inherited by all child runs
        self.scenario_selection = scenario_selection
        self.sampling_selection = sampling_selection
        self.modeling_selection = modeling_selection

        # SQL-friendly run name for MLflow (not used in tags)
        self.run_name = f"run_{self.scenario_selection}_{self.sampling_selection}_{self.modeling_selection}"

        # Pipeline timing (supplements MLflow's automatic timing)
        self.start_time: Optional[str] = None  # Set on first access
        self.end_time: Optional[str] = None  # Set when end() called
        self.run_time: Optional[str] = None  # Human-readable duration

        # Failure tracking (inherited by all result classes)
        self.failed: bool = False
        self.failure_reason: Optional[str] = None

    def get_start_time(self) -> str:
        """Get start time, setting it on first access (lazy evaluation)."""
        if not self.start_time:
            self.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        return self.start_time

    def get_end_time(self) -> Optional[str]:
        """Get end time, or None if not ended yet."""
        return self.end_time

    def get_run_time(self) -> Optional[str]:
        """Get human-readable runtime (e.g. "2h 15m 30s"), or None if not ended yet."""
        return self.run_time

    def mark_failed(self, reason: Union[str, Exception]) -> None:
        """Mark this result as failed with a reason."""
        self.failed = True

        # Handle both string reasons and exception objects
        if isinstance(reason, Exception):
            self.failure_reason = f"{type(reason).__name__}: {str(reason)}"
        else:
            self.failure_reason = reason

        # Also mark as ended if not already
        if not self.end_time:
            self.end()

    def end(self) -> None:
        """Mark pipeline complete and calculate human-readable runtime."""
        self.end_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        if self.start_time and self.end_time:
            start_dt = datetime.strptime(self.start_time, "%d-%m-%Y %H:%M:%S")
            end_dt = datetime.strptime(self.end_time, "%d-%m-%Y %H:%M:%S")
            duration = end_dt - start_dt

            total_seconds = int(duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                self.run_time = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                self.run_time = f"{minutes}m {seconds}s"
            else:
                self.run_time = f"{seconds}s"

    def format_tags(self, tags: Dict[str, Any]) -> Dict[str, str]:
        """Format tags for MLflow compatibility: convert to strings and filter None values."""
        return {k: str(v) for k, v in tags.items() if v is not None}

    def get_tags(self) -> Dict[str, str]:
        """
        Get static tags that inherit down MLflow hierarchy for cross-experiment queries.

        Returns:
        --------
        Dict[str, str]
            Core configuration and timing tags:
            - scenario: Dataset selectio
            - sampling: Sampling strategy
            - modeling: Model set
            - start_time: Pipeline start timestamp
            - end_time: Pipeline end timestamp (None if not ended)
            - run_time: Human-readable duration (None if not ended)

        Note:
            The run_name attribute is used for MLflow run naming but not included
            in tags to avoid redundancy.
        """
        tags = {
            # Core configuration with shortened names for cleaner MLflow UI
            "scenario": self.scenario_selection,
            "sampling": self.sampling_selection,
            "modeling": self.modeling_selection,
            # Timing information
            "start_time": self.get_start_time(),
            "end_time": self.get_end_time(),
            "run_time": self.get_run_time(),
            # Failure tracking
            "failed": self.failed,
            "failure_reason": self.failure_reason,
        }

        return self.format_tags(tags)
