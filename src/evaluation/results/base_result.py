from abc import ABC
from datetime import datetime
from typing import Dict, Optional


class BaseResult(ABC):
    """
    Base class for all result types with static configuration inheritance.

    Provides the three crucial static tags (scenario, sampling, modeling) that
    inherit down the 4-level MLflow hierarchy, enabling cross-experiment queries.

    Also tracks end-to-end pipeline runtime to complement MLflow's per-run timing.
    """

    def __init__(
        self, scenario_selection: str, sampling_selection: str, modeling_selection: str
    ):
        """Initialize with the three crucial experimental design choices."""
        # Core experimental configuration - inherited by all child runs
        self.scenario_selection = scenario_selection  # e.g. "s06"
        self.sampling_selection = sampling_selection  # e.g. "binary_for_extremes"
        self.modeling_selection = modeling_selection  # e.g. "paper"

        # SQL-friendly identifier for queries: "run_s06_binary_for_extremes_paper"
        self.run_name = f"run_{self.scenario_selection}_{self.sampling_selection}_{self.modeling_selection}"

        # Pipeline timing (business time vs MLflow's technical time)
        self.start_time: Optional[str] = None  # Set on first access
        self.end_time: Optional[str] = None  # Set when end() called
        self.run_time: Optional[str] = None  # Calculated when end() called

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

    def get_result_tags(self) -> Dict[str, str]:
        """
        Get static tags that inherit down MLflow hierarchy for cross-experiment queries.

        Enables SQL queries like: WHERE tags.run_name LIKE 'run_s06_%'
        """
        return {
            "run_name": self.run_name,
            "scenario_selection": self.scenario_selection,
            "sampling_selection": self.sampling_selection,
            "modeling_selection": self.modeling_selection,
            "start_time": self.get_start_time(),
            "end_time": self.get_end_time(),
            "run_time": self.get_run_time(),
        }
