# =============================================================================
# PIPELINE PREPARATION ERRORS (Early stages)
# =============================================================================


class DataPreparationError(Exception):
    """
    Errors during data loading and preprocessing.

    Examples: PyScrew data loading failures, file not found,
    PAA transformation errors, normalization issues.
    """

    pass


class DataSamplingError(Exception):
    """
    Errors during dataset sampling and generation.

    Examples: insufficient samples for sampling strategy,
    missing classes in data, invalid sampling configuration,
    scenario-sampling compatibility issues.
    """

    pass


# =============================================================================
# EVALUATION EXECUTION ERRORS (ML pipeline stages)
# =============================================================================


class ExperimentEvaluationError(Exception):
    """
    Critical errors that stop the entire experiment.

    Examples: MLflow server unreachable, invalid experiment configuration,
    too many dataset failures, fundamental setup issues.
    """

    pass


class DatasetEvaluationError(Exception):
    """
    Errors during dataset processing in ML pipeline.

    Examples: cross-validation setup failures, incompatible data formats,
    too many model failures for a dataset.
    """

    pass


class ModelEvaluationError(Exception):
    """
    Errors during individual model evaluation.

    Examples: model initialization failures, training errors,
    incompatible data formats, too many fold failures.
    """

    pass


class FoldEvaluationError(Exception):
    """
    Errors during individual fold evaluation.

    Examples: training failures, prediction errors, memory issues,
    data format problems in specific CV folds.
    """

    pass
