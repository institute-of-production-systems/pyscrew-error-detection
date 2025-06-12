#!/usr/bin/env python3
"""
Main entry point for PyScrew Error Detection Experiments.

This script orchestrates comprehensive machine learning experiments on industrial screw
driving data from the PyScrew dataset collection. The framework supports supervised error
detection across multiple scenarios (currently, for s04-s06) with various sampling
strategies and model selections.

The architecture centers around three crucial experimental design choices that determine
the entire experimental pipeline:
1. Scenario Selection: Which PyScrew dataset to use (s04/s05/s06)
2. Sampling Selection: How to structure the ML problem (binary/multiclass approaches)
3. Modeling Selection: Which set of ML models to evaluate (debug/fast/paper/full)

Technical parameters (CV folds, data processing, etc.) are configured via config.yml
to keep this main script focused on experimental design decisions.

Example Usage:
    python main.py --scenario s04 --sampling binary_vs_ref --modeling paper
    python main.py --scenario s05 --sampling all --modeling fast

For detailed configuration options, see config.yml in the project root.
"""

import argparse
import sys
import textwrap
import time
from typing import List, Protocol

import yaml

from src.experiments import ExperimentRunner
from src.utils import get_logger
from src.utils.exceptions import FatalExperimentError

# =============================================================================
# EXPERIMENTAL DESIGN CONFIGURATION
# =============================================================================
# The three crucial choices that define the experimental design and determine
# the entire pipeline execution. These selections control:
# - Which dataset is loaded and processed
# - How the ML problem is structured (binary vs multiclass)
# - Which models are trained and evaluated
# - The scope and duration of the experiment

EXPERIMENTAL_DEFAULTS = {
    "scenario_selection": "s06",  # Which PyScrew dataset (s04/s05/s06)
    "sampling_selection": "binary_for_extremes",  # Which sampling strategy
    "modeling_selection": "paper",  # Which model set (debug/fast/paper/full)
}

# Scenario-Sampling Compatibility Matrix
# =====================================
# Not all sampling strategies are available for all scenarios due to differences
# in dataset structure and error types:
# - s04: Assembly errors with normal/faulty conditions per class
# - s05: Parameter variations in upper workpiece manufacturing
# - s06: Parameter variations in lower workpiece manufacturing
SCENARIO_SAMPLING_COMPATIBILITY = {
    "s04": [
        "binary_vs_ref",
        "binary_vs_all",
        "multiclass_with_groups",
        "multiclass_with_all",
    ],
    "s05": ["binary_for_extremes", "multiclass_with_all", "multiclass_within_groups"],
    "s06": ["binary_for_extremes", "multiclass_with_all", "multiclass_within_groups"],
}

# Flatten all sampling options for CLI choices validation
# This ensures the CLI accepts any valid sampling strategy regardless of scenario,
# with validation happening in get_sampling_strategies_to_run()
ALL_SAMPLING_OPTIONS = sorted(
    set(
        option
        for options in SCENARIO_SAMPLING_COMPATIBILITY.values()
        for option in options
    )
) + ["all"]

# Available modeling and scenario options (no compatibility restrictions)
MODELING_OPTIONS = ["debug", "fast", "paper", "full", "sklearn", "sktime"]
SCENARIO_OPTIONS = ["s04", "s05", "s06"]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


class ParsedArgs(Protocol):
    """Type hint for parsed command line arguments."""

    scenario: str
    sampling: str
    modeling: str
    quiet: bool
    verbose: bool


def parse_arguments() -> ParsedArgs:
    """
    Parse command line arguments for the three core experimental choices.

    This function defines the CLI interface focused exclusively on experimental design.
    Technical parameters (CV folds, random seeds, etc.) are handled via config.yml
    to maintain clean separation between experimental choices and implementation details.

    The CLI validates scenario-sampling compatibility and provides helpful error messages
    for invalid combinations, along with comprehensive help text showing which strategies
    are compatible with each scenario.

    Returns:
        ParsedArgs: Parsed arguments containing scenario_selection,
                   sampling_selection, modeling_selection, and logging flags.

    Raises:
        SystemExit: If invalid arguments are provided or --help is requested.
    """
    parser = argparse.ArgumentParser(
        description="Run supervised error detection experiments on industrial screw driving data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
            \t%(prog)s                                                         # Run default experiment  
            \t%(prog)s --scenario s04 --sampling binary_vs_ref --modeling fast # Run binary vs reference with fast models
            \t%(prog)s --scenario s05 --sampling all --modeling paper          # Run all s05-compatible strategies
            \t%(prog)s --scenario s06 --sampling binary_for_extremes           # Valid s06 strategy
            
            Scenario-Sampling Compatibility:
            \ts04: binary_vs_ref, binary_vs_all, multiclass_with_groups, multiclass_with_all
            \ts05: binary_for_extremes, multiclass_with_all, multiclass_within_groups  
            \ts06: binary_for_extremes, multiclass_with_all, multiclass_within_groups  
            """
        ).strip(),
    )

    # The three core experimental design choices (scenario, sampling, modeling)
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_OPTIONS,
        default=EXPERIMENTAL_DEFAULTS["scenario_selection"],
        help=f"Dataset scenario to use (default: {EXPERIMENTAL_DEFAULTS['scenario_selection']})",
    )

    parser.add_argument(
        "--sampling",
        choices=ALL_SAMPLING_OPTIONS,
        default=EXPERIMENTAL_DEFAULTS["sampling_selection"],
        help=f"Sampling strategy (default: {EXPERIMENTAL_DEFAULTS['sampling_selection']}). "
        f"Note: Not all strategies are compatible with all scenarios. "
        f"Use 'all' to run all compatible strategies for the selected scenario.",
    )

    parser.add_argument(
        "--modeling",
        choices=MODELING_OPTIONS,
        default=EXPERIMENTAL_DEFAULTS["modeling_selection"],
        help=f"Model selection strategy (default: {EXPERIMENTAL_DEFAULTS['modeling_selection']})",
    )

    # Logging control (overrides config.yml)
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output (WARNING level)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output (DEBUG level)"
    )

    return parser.parse_args()


def get_sampling_strategies_to_run(args: ParsedArgs) -> List[str]:
    """
    Determine which sampling strategies to run based on CLI arguments with validation.

    This function handles two modes:
    1. Single strategy: Validates that the specified strategy is compatible with the scenario
    2. All strategies: Returns all strategies compatible with the selected scenario

    The validation prevents runtime errors by catching incompatible scenario-sampling
    combinations early with clear error messages showing valid alternatives.

    Args:
        args: Parsed command line arguments containing scenario and sampling selections

    Returns:
        List[str]: List of sampling strategy names to execute

    Raises:
        ValueError: If the specified sampling strategy is incompatible with the scenario,
                   includes helpful message showing compatible alternatives
    """
    if args.sampling == "all":
        # Return all sampling strategies compatible with the selected scenario
        return SCENARIO_SAMPLING_COMPATIBILITY[args.scenario]
    else:
        # Validate that the selected sampling strategy is compatible with the scenario
        compatible_strategies = SCENARIO_SAMPLING_COMPATIBILITY[args.scenario]
        if args.sampling not in compatible_strategies:
            raise ValueError(
                f"Sampling strategy '{args.sampling}' is not compatible with scenario '{args.scenario}'. "
                f"Compatible options for {args.scenario}: {', '.join(compatible_strategies)}"
            )
        return [args.sampling]


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> int:
    """
    Main entry point for PyScrew error detection experiments.

    This function orchestrates the complete experimental pipeline from CLI parsing
    to result generation. It implements a clean separation between experimental
    design choices (handled here) and technical configuration (loaded from config.yml).

    Execution Flow:
    1. Parse experimental design choices from command line arguments
    2. Load technical configuration (CV settings, data processing, etc.) from config.yml
    3. Override logging configuration with CLI flags if provided
    4. Validate scenario-sampling compatibility with early error reporting
    5. Execute ExperimentRunner.run() for each sampling strategy
    6. Provide comprehensive execution summary and MLflow UI access information

    The core of the entire project is the ExperimentRunner.run() call, which:
    - Loads and preprocesses PyScrew data according to scenario selection
    - Generates datasets according to sampling selection
    - Trains and evaluates models according to modeling selection
    - Logs all results to MLflow with 4-level hierarchy (Experiment→Dataset→Model→Fold)

    Error Handling:
    - Graceful handling of user interruption (Ctrl+C)
    - Clear error messages for invalid scenario-sampling combinations
    - Individual experiment failure tracking without stopping the entire run
    - Comprehensive summary of successes and failures

    Returns:
        int: Exit code (0 for success, 1 for errors, 130 for user interruption)

    Raises:
        SystemExit: Via return codes for shell integration
    """
    try:
        # Parse experimental design choices
        args = parse_arguments()

        # Load technical configuration straight from the yaml file
        with open("config.yml") as config_file:
            config: dict = yaml.load(config_file, Loader=yaml.SafeLoader)

        # Override logging config with CLI flags if provided
        if args.verbose:
            config["log_level"] = "DEBUG"
        elif args.quiet:
            config["log_level"] = "WARNING"

        # Setup logging
        logger = get_logger(__name__, config["log_level"])

        # Determine and validate sampling strategies to run
        try:
            sampling_strategies = get_sampling_strategies_to_run(args)
        except ValueError as e:
            logger.error(str(e))
            return 1

        logger.info(f"Starting PyScrew error detection experiments")
        logger.info(f"  Scenario: {args.scenario}")
        logger.info(f"  Sampling: {len(sampling_strategies)} strategies")
        logger.info(f"  Modeling: {args.modeling}")

        # Execute experiments
        successful = 0
        failed = 0

        for i, sampling_strategy in enumerate(sampling_strategies, 1):
            logger.info(
                f"Experiment {i}/{len(sampling_strategies)}: {sampling_strategy}"
            )

            try:
                # Loop Step 1: Create ExperimentRunner with complete configuration
                # The three crucial experimental design choices determine the entire pipeline:
                # - scenario_selection: Controls data loading and sampling module selection
                # - sampling_selection: Controls dataset generation strategy
                # - modeling_selection: Controls which ML models are trained and evaluated
                # Technical config from config.yml provides implementation parameters
                runner = ExperimentRunner(
                    # The three crucial experimental design choices
                    scenario_selection=args.scenario,
                    sampling_selection=sampling_strategy,
                    modeling_selection=args.modeling,
                    # Technical configuration (unpacked from config.yml)
                    **config,
                )

                logger.info(f"Starting {sampling_strategy} experiment...")

                # Loop Step 2: CORE EXECUTION - This is the heart of the entire project
                # ExperimentRunner.run() orchestrates the complete ML pipeline:
                # 1. Loads PyScrew data according to scenario_selection
                # 2. Applies preprocessing (PAA reduction, normalization)
                # 3. Generates datasets according to sampling_selection
                # 4. Initializes models according to modeling_selection
                # 5. Executes cross-validation training and evaluation
                # 6. Logs comprehensive results to MLflow with 4-level hierarchy
                # 7. Returns hierarchical results for analysis
                start_time = time.time()
                results = runner.run()  # ← THE ENTIRE PROJECT CENTERS ON THIS CALL
                duration = time.time() - start_time

                # Loop Step 3: Log execution summary with key metrics
                total_datasets = len(results.dataset_results)
                total_models = sum(
                    len(dr.model_results) for dr in results.dataset_results
                )

                logger.info(
                    f"Completed {sampling_strategy} in {duration:.1f}s "
                    f"({total_datasets} datasets, {total_models} models)"
                )

                successful += 1

            except FatalExperimentError as e:
                # Handle expected experiment failures (MLflow issues, data problems, etc.)
                logger.error(f"Fatal error in {sampling_strategy}: {str(e)}")
                failed += 1
            except Exception as e:
                # Handle unexpected failures with full error information
                logger.error(f"Unexpected error in {sampling_strategy}: {str(e)}")
                failed += 1

        # Provide comprehensive execution summary
        logger.info(f"Experiment Summary: {successful} successful, {failed} failed")
        if successful > 0:
            logger.info(
                f"View results at: http://localhost:{config.get('mlflow_port', 5000)} (MLflow UI)"
            )

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C) gracefully
        logger = get_logger(__name__)
        logger.warning("Interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        # Handle any unexpected errors at the top level
        logger = get_logger(__name__)
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    # Entry point when script is run directly (not imported)
    # Exits with appropriate code for shell integration
    sys.exit(main())
