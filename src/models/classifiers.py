import importlib
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import yaml

from src.utils import get_logger

# Set up logger
logger = get_logger(__name__)

# Define type variables for models
ModelType = TypeVar("ModelType")


# Custom exceptions
class ModelInitializationError(Exception):
    """Exception raised when a model fails to initialize."""

    pass


class ConfigurationError(Exception):
    """Exception raised when there's an issue with the configuration."""

    pass


def load_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Load model configurations from YAML files.

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Combined dictionary of model configurations from sklearn and sktime
    """
    # Load sklearn models
    with open("src/models/sklearn_models.yml", "r") as file:
        sklearn_configs = yaml.safe_load(file)
    # Load sktime models
    with open("src/models/sktime_models.yml", "r") as file:
        sktime_configs = yaml.safe_load(file)
    return {**sklearn_configs, **sktime_configs}


def get_classifier_dict(
    modeling_selection: str, random_seed: int, n_jobs: int
) -> Dict[str, Any]:
    """
    Return a dictionary of initialized models based on selection.

    Parameters:
    -----------
    modeling_selection : str
        One of 'debug', 'fast', 'paper', 'full', 'sklearn', 'sktime'
    random_seed : int
        Random seed to set for all models that support it
    n_jobs : int
        Number of parallel jobs for models that support it
        -1 means using all processors

    Returns:
    --------
    Dict[str, Any]
        Dictionary with model names as keys and initialized models as values

    Raises:
    -------
    ConfigurationError
        If modeling_selection is not valid or config files cannot be loaded
    """
    try:
        # Load model configurations
        model_configs = load_model_configs()

        logger.debug(
            f"Initializing models for '{modeling_selection}' selection with random_seed={random_seed}, n_jobs={n_jobs}"
        )

        # Validate modeling_selection
        valid_selections = ["debug", "fast", "paper", "full", "sklearn", "sktime"]
        if modeling_selection not in valid_selections:
            raise ConfigurationError(
                f"Invalid modeling_selection: {modeling_selection}. Must be one of {valid_selections}"
            )

        # Initialize empty result dictionary
        classifiers: Dict[str, Any] = {}

        # Filter models based on modeling_selection
        for name, config in model_configs.items():
            if modeling_selection in config.get("in_sets", []):
                try:
                    classifiers[name] = _initialize_model(
                        name, config, random_seed, n_jobs
                    )
                    logger.debug(f"Successfully initialized model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize model {name}: {str(e)}")
                    raise ModelInitializationError(
                        f"Failed to initialize {name}: {str(e)}"
                    ) from e

        logger.info(
            f"Initialized {len(classifiers)} models for '{modeling_selection}' selection"
        )
        return classifiers

    except Exception as e:
        logger.error(f"Error in get_classifier_dict: {str(e)}")
        raise


def _initialize_model(
    name: str, config: Dict[str, Any], random_seed: int, n_jobs: int
) -> Any:
    """
    Initialize a model from its configuration.

    Handles special cases for models that need additional setup,
    applies random seed and n_jobs settings to all models that support them.

    Parameters:
    -----------
    name : str
        Name of the model to initialize
    config : Dict[str, Any]
        Configuration dictionary for the model
    random_seed : int
        Random seed to set for models that support it
    n_jobs : int
        Number of parallel jobs for models that support it

    Returns:
    --------
    Any
        Initialized model instance

    Raises:
    -------
    ModelInitializationError
        If the model fails to initialize
    """
    try:
        # Get the parameters from config
        params = config.get("params", {}).copy()

        logger.debug(f"Initializing {name} with params: {params}")

        # Handle special cases
        if name == "BaggingClassifier" and params.get("estimator") is None:
            # Set default base estimator for BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier

            params["estimator"] = DecisionTreeClassifier(
                max_depth=5, random_state=random_seed
            )
            logger.debug(f"Setting default DecisionTreeClassifier estimator for {name}")

        if name == "ShapeletTransformClassifier" and params.get("estimator") is None:
            # Set default classifier for transformed data
            from sklearn.ensemble import RandomForestClassifier

            params["estimator"] = RandomForestClassifier(
                n_estimators=100, n_jobs=n_jobs, random_state=random_seed
            )
            logger.debug(f"Setting default RandomForestClassifier estimator for {name}")

        # Get parameter names for the model
        module_path = config["import"]["from"]
        class_name = config["import"]["class"]
        param_names = _get_param_names(module_path, class_name)

        # Set random_state parameter if the model supports it
        if "random_state" in param_names:
            params["random_state"] = random_seed
            logger.debug(f"Setting random_state={random_seed} for {name}")

        # Set n_jobs parameter if the model supports it
        if "n_jobs" in param_names:
            # Special case for WEASEL which has thread limitations (max 12)
            if name == "WEASEL":
                # If n_jobs is -1 or greater than 12, limit to 8 as a safe value
                if n_jobs == -1 or n_jobs > 12:
                    params["n_jobs"] = 8
                    logger.debug(
                        f"Limiting {name} to n_jobs=8 due to thread limitations"
                    )
                else:
                    params["n_jobs"] = n_jobs
                    logger.debug(f"Setting n_jobs={n_jobs} for {name}")
            else:
                params["n_jobs"] = n_jobs
                logger.debug(f"Setting n_jobs={n_jobs} for {name}")

        # Dynamically import the module and class
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Initialize the model with parameters
            model = model_class(**params)
            return model

        except ImportError:
            logger.error(f"Could not import module {module_path}")
            raise ModelInitializationError(f"Module not found: {module_path}")
        except AttributeError:
            logger.error(f"Class {class_name} not found in module {module_path}")
            raise ModelInitializationError(
                f"Class not found: {class_name} in {module_path}"
            )

    except Exception as e:
        logger.error(f"Error initializing model {name}: {str(e)}")
        raise ModelInitializationError(f"Failed to initialize {name}: {str(e)}") from e


def _get_param_names(module_name: str, class_name: str) -> List[str]:
    """
    Get parameter names for a given class.

    Parameters:
    -----------
    module_name : str
        Name of the module
    class_name : str
        Name of the class

    Returns:
    --------
    List[str]
        List of parameter names for the class
    """
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        logger.debug(f"Getting parameter names for {class_name} from {module_name}")

        # Try to get parameter names from class init
        # This works for scikit-learn and most sktime models
        if hasattr(model_class, "__init__"):
            init_signature = getattr(model_class.__init__, "__signature__", None)
            if init_signature:
                params = list(init_signature.parameters.keys())
                logger.debug(f"Found {len(params)} parameters from signature: {params}")
                return params

        # Fallback: try to get parameter names from get_params method
        if hasattr(model_class, "get_params"):
            # Create a temporary instance with default parameters
            try:
                temp_instance = model_class()
                params = list(temp_instance.get_params().keys())
                logger.debug(
                    f"Found {len(params)} parameters from get_params(): {params}"
                )
                return params
            except Exception as e:
                logger.warning(f"Failed to create instance of {class_name}: {str(e)}")

        # If all else fails, return empty list
        logger.warning(
            f"Could not determine parameters for {class_name}, returning empty list"
        )
        return []
    except ImportError:
        logger.error(f"Could not import module {module_name}")
        return []
    except AttributeError:
        logger.error(f"Class {class_name} not found in module {module_name}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting parameters for {class_name}: {str(e)}")
        return []
