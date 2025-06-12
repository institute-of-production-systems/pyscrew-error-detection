"""
Utility functions for plotting and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ensure_directory(path):
    """
    Ensure a directory exists, creating it if necessary.

    Parameters:
    -----------
    path : str
        Directory path to ensure exists.
    """
    os.makedirs(path, exist_ok=True)


def save_results_with_plots(results_df, base_dir, sampling_selection):
    """
    Save results DataFrame and generate standard visualizations.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing model results with metrics.
    base_dir : str
        Base directory for saving results.
    sampling_selection : str
        Name of the experiment (used for subdirectory).

    Returns:
    --------
    str
        Path to the saved results directory.
    """
    from .confusion_matrix import plot_confusion_matrix
    from .model_metrics import plot_metrics_comparison
    from .class_performance import plot_class_performance

    if results_df.empty:
        print("Warning: Empty results DataFrame.")
        return None

    # Create directory structure
    results_path = os.path.join(base_dir, sampling_selection)
    images_path = os.path.join(results_path, "images")
    ensure_directory(results_path)
    ensure_directory(images_path)

    # Save results DataFrame
    results_df.to_csv(os.path.join(results_path, "results.csv"), index=False)

    # Generate model metrics visualization
    plot_metrics_comparison(
        results_df,
        save_path=os.path.join(images_path, "overall_metrics_comparison.png"),
    )

    # Generate per-model visualizations
    for model_name in results_df["model"].unique():
        plot_class_performance(
            results_df,
            model_name=model_name,
            save_path=os.path.join(images_path, f"f1_by_class_{model_name}.png"),
        )

    print(f"Results and visualizations saved to {results_path}/")
    return results_path
