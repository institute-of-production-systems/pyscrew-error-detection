import argparse
import sys
import time
from typing import List

from src.experiments import ExperimentRunner
from src.utils import get_logger
from src.utils.exceptions import FatalExperimentError

# Default configuration
DEFAULT_SETTINGS = {
    "scenario_id": "s01",
    "target_length": 2000,
    "screw_positions": "left",
    "cv_folds": 5,
    "random_seed": 42,
    "n_jobs": -1,
    "stratify": True,
    "log_level": "INFO",
}

EXPERIMENT_OPTIONS = [
    "binary_vs_ref",
    "binary_vs_all",
    "multiclass_with_groups",
    "multiclass_with_all",
]

MODEL_SELECTIONS = ["debug", "fast", "paper", "full", "sklearn", "sktime"]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments - keep it simple."""
    parser = argparse.ArgumentParser(
        description="Run supervised error detection experiments on screw driving data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run all experiments (paper models)
  %(prog)s --experiment binary_vs_ref   # Run specific experiment  
  %(prog)s --models fast                # Use fast models for testing
        """,
    )

    # Core arguments
    parser.add_argument(
        "--experiment",
        "-e",
        choices=EXPERIMENT_OPTIONS + ["all"],
        default="all",
        help="Experiment type to run (default: all)",
    )

    parser.add_argument(
        "--models",
        "-m",
        choices=MODEL_SELECTIONS,
        default="paper",
        help="Model selection strategy (default: paper)",
    )

    # Optional tuning
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_SETTINGS["cv_folds"],
        help=f"Number of CV folds (default: {DEFAULT_SETTINGS['cv_folds']})",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_SETTINGS["random_seed"],
        help=f"Random seed (default: {DEFAULT_SETTINGS['random_seed']})",
    )

    # Logging control
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output (WARNING level)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output (DEBUG level)"
    )

    return parser.parse_args()


def get_experiments_to_run(args: argparse.Namespace) -> List[str]:
    """Determine which experiments to run."""
    if args.experiment == "all":
        return EXPERIMENT_OPTIONS
    else:
        return [args.experiment]


def run_single_experiment(
    experiment_name: str, args: argparse.Namespace, logger
) -> bool:
    """Run a single experiment and return success status."""
    try:
        # Create experiment runner
        runner = ExperimentRunner(
            experiment_name=experiment_name,
            model_selection=args.models,
            scenario_id=DEFAULT_SETTINGS["scenario_id"],
            target_length=DEFAULT_SETTINGS["target_length"],
            screw_positions=DEFAULT_SETTINGS["screw_positions"],
            cv_folds=args.cv_folds,
            random_seed=args.random_seed,
            n_jobs=DEFAULT_SETTINGS["n_jobs"],
            stratify=DEFAULT_SETTINGS["stratify"],
            log_level="DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO",
        )

        logger.info(
            f"Starting experiment: {experiment_name} (models: {args.models}, cv: {args.cv_folds})"
        )

        start_time = time.time()
        results = runner.run()
        duration = time.time() - start_time

        # Log completion
        total_datasets = len(results.dataset_results)
        total_models = sum(len(dr.model_results) for dr in results.dataset_results)

        logger.info(
            f"Completed {experiment_name} in {duration:.1f}s ({total_datasets} datasets, {total_models} models)"
        )

        return True

    except FatalExperimentError as e:
        logger.error(f"Fatal error in {experiment_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in {experiment_name}: {str(e)}")
        return False


def main() -> int:
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        log_level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
        logger = get_logger(__name__, log_level)

        # Determine experiments to run
        experiments_to_run = get_experiments_to_run(args)

        logger.info(
            f"Running {len(experiments_to_run)} experiment(s) with '{args.models}' models"
        )

        # Run experiments
        successful = 0
        failed = 0

        for i, experiment_name in enumerate(experiments_to_run, 1):
            logger.info(f"Experiment {i}/{len(experiments_to_run)}: {experiment_name}")

            if run_single_experiment(experiment_name, args, logger):
                successful += 1
            else:
                failed += 1

        # Final summary
        logger.info(f"Summary: {successful} successful, {failed} failed")
        if successful > 0:
            logger.info("View results at: http://localhost:5000 (MLflow UI)")

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
