import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_classification_results(csv_path, output_dir=None, figsize=(12, 10), dpi=100):
    """
    Generate visualizations for time series classification results.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing classification results.
    output_dir : str, optional
        Directory to save the generated plots. If None, plots will only be displayed.
    figsize : tuple, optional
        Figure size for plots (width, height) in inches.
    dpi : int, optional
        Resolution of the plots.

    Returns:
    --------
    dict
        Dictionary containing the figure objects for further customization.
    """
    # Create output directory if specified
    if output_dir:
        # Create the full directory path
        os.makedirs(output_dir, exist_ok=True)

        # Extract experiment name from the path for figure naming
        csv_filename = os.path.basename(csv_path)
        sampling_selection = os.path.splitext(csv_filename)[0]

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract unique classes and models
    classes = df["class"].unique()
    models = df["model"].unique()

    # Calculate average performance metrics for each model
    model_summary = (
        df.groupby("model")
        .agg(
            {
                "accuracy": "mean",
                "precision": "mean",
                "recall": "mean",
                "f1-score": "mean",
            }
        )
        .reset_index()
    )

    # Sort by accuracy
    model_summary = model_summary.sort_values("accuracy", ascending=False)

    # Find the best performing model for each class
    best_model_by_class = df.loc[df.groupby("class")["accuracy"].idxmax()]

    # Count how many times each model is the best
    model_win_counts = best_model_by_class["model"].value_counts().reset_index()
    model_win_counts.columns = ["model", "win_count"]

    # Calculate average performance by class prefix
    df["prefix"] = df["class"].str.split("_").str[0]
    prefix_performance = (
        df.groupby(["prefix", "model"])
        .agg(
            {
                "accuracy": "mean",
                "precision": "mean",
                "recall": "mean",
                "f1-score": "mean",
            }
        )
        .reset_index()
    )

    # Dictionary to store all figures
    figures = {}

    # Set up the style
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # 1. Plot average accuracy by model
    fig1, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    # Fix seaborn warning by using hue and legend=False
    sns.barplot(
        x="model",
        y="accuracy",
        data=model_summary,
        ax=ax1,
        hue="model",
        palette="viridis",
        legend=False,
    )
    ax1.set_title("Average Accuracy by Model", fontsize=16)
    ax1.set_xlabel("Model", fontsize=14)
    ax1.set_ylabel("Average Accuracy", fontsize=14)
    ax1.set_ylim(0.7, max(model_summary["accuracy"]) + 0.05)
    ax1.tick_params(axis="x", rotation=45)

    for i, v in enumerate(model_summary["accuracy"]):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    figures["accuracy_by_model"] = fig1

    if output_dir:
        fig1.savefig(Path(output_dir) / f"{sampling_selection}_accuracy_by_model.png")

    # 2. Plot win count by model
    fig2, ax2 = plt.subplots(figsize=figsize, dpi=dpi)

    # Sort by win count
    model_win_counts = model_win_counts.sort_values("win_count", ascending=False)

    # Fix seaborn warning by using hue and legend=False
    sns.barplot(
        x="model",
        y="win_count",
        data=model_win_counts,
        ax=ax2,
        hue="model",
        palette="plasma",
        legend=False,
    )
    ax2.set_title("Number of Classes Where Model Performs Best", fontsize=16)
    ax2.set_xlabel("Model", fontsize=14)
    ax2.set_ylabel("Win Count", fontsize=14)
    ax2.tick_params(axis="x", rotation=45)

    for i, v in enumerate(model_win_counts["win_count"]):
        ax2.text(i, v + 0.3, str(v), ha="center", fontsize=12)

    plt.tight_layout()
    figures["win_count_by_model"] = fig2

    if output_dir:
        fig2.savefig(Path(output_dir) / f"{sampling_selection}_win_count_by_model.png")

    # 3. Plot performance metrics comparison
    fig3, ax3 = plt.subplots(figsize=figsize, dpi=dpi)

    metrics = ["accuracy", "precision", "recall", "f1-score"]
    model_metrics = model_summary.melt(
        id_vars="model", value_vars=metrics, var_name="metric", value_name="value"
    )

    sns.barplot(
        x="model", y="value", hue="metric", data=model_metrics, ax=ax3, palette="Set2"
    )
    ax3.set_title("Performance Metrics by Model", fontsize=16)
    ax3.set_xlabel("Model", fontsize=14)
    ax3.set_ylabel("Value", fontsize=14)
    ax3.set_ylim(0.7, max(model_metrics["value"]) + 0.05)
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend(title="Metric", fontsize=12)

    plt.tight_layout()
    figures["metrics_by_model"] = fig3

    if output_dir:
        fig3.savefig(Path(output_dir) / f"{sampling_selection}_metrics_by_model.png")

    # 4. Heatmap of model performance by class category
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(index="prefix", columns="model", values="accuracy")

    fig4, ax4 = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(
        heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax4
    )
    ax4.set_title("Model Accuracy by Category", fontsize=16)
    ax4.set_xlabel("Model", fontsize=14)
    ax4.set_ylabel("Category", fontsize=14)

    plt.tight_layout()
    figures["category_heatmap"] = fig4

    if output_dir:
        fig4.savefig(Path(output_dir) / f"{sampling_selection}_category_heatmap.png")

    # 5. Best model performance by class
    fig5, ax5 = plt.subplots(figsize=(figsize[0], figsize[1] * 1.5), dpi=dpi)

    best_model_by_class = best_model_by_class.sort_values("accuracy", ascending=False)

    # Create color mapping for models
    model_colors = {model: color for model, color in zip(models, colors)}
    bar_colors = [model_colors[model] for model in best_model_by_class["model"]]

    bars = ax5.barh(
        best_model_by_class["class"], best_model_by_class["accuracy"], color=bar_colors
    )
    ax5.set_title("Best Model Performance by Class", fontsize=16)
    ax5.set_xlabel("Accuracy", fontsize=14)
    ax5.set_ylabel("Class", fontsize=14)
    ax5.set_xlim(0.5, 1.05)

    # Add model names to the bars
    for i, bar in enumerate(bars):
        ax5.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            best_model_by_class.iloc[i]["model"],
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    figures["best_model_by_class"] = fig5

    if output_dir:
        fig5.savefig(Path(output_dir) / f"{sampling_selection}_best_model_by_class.png")

    # 6. Radar chart of model performance
    fig6, ax6 = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=dpi)

    # Prepare data for radar chart
    metrics = ["accuracy", "precision", "recall", "f1-score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Set up radar chart
    ax6.set_theta_offset(np.pi / 2)
    ax6.set_theta_direction(-1)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)

    for i, model in enumerate(model_summary["model"]):
        values = (
            model_summary.loc[model_summary["model"] == model, metrics]
            .values.flatten()
            .tolist()
        )
        values += values[:1]  # Close the polygon
        ax6.plot(angles, values, linewidth=2, label=model, color=colors[i])
        ax6.fill(angles, values, alpha=0.1, color=colors[i])

    ax6.set_title("Model Performance Across Metrics", fontsize=16)
    ax6.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    figures["radar_chart"] = fig6

    if output_dir:
        fig6.savefig(Path(output_dir) / f"{sampling_selection}_radar_chart.png")

    # 7. Box plot of accuracy distribution by model
    fig7, ax7 = plt.subplots(figsize=figsize, dpi=dpi)

    # Sort models by median accuracy
    model_order = (
        df.groupby("model")["accuracy"].median().sort_values(ascending=False).index
    )

    # Fix seaborn warning by using hue and legend=False
    sns.boxplot(
        x="model",
        y="accuracy",
        data=df,
        ax=ax7,
        order=model_order,
        hue="model",
        palette="viridis",
        legend=False,
    )
    ax7.set_title("Accuracy Distribution by Model", fontsize=16)
    ax7.set_xlabel("Model", fontsize=14)
    ax7.set_ylabel("Accuracy", fontsize=14)
    ax7.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    figures["accuracy_distribution"] = fig7

    if output_dir:
        fig7.savefig(
            Path(output_dir) / f"{sampling_selection}_accuracy_distribution.png"
        )

    # 8. Average performance by class
    avg_by_class = df.groupby("class")["accuracy"].mean().reset_index()
    avg_by_class = avg_by_class.sort_values("accuracy", ascending=False)

    fig8, ax8 = plt.subplots(figsize=(figsize[0], figsize[1] * 1.5), dpi=dpi)

    # Fix seaborn warning by using hue and legend=False
    sns.barplot(
        x="accuracy",
        y="class",
        data=avg_by_class,
        ax=ax8,
        hue="class",
        palette="cividis",
        legend=False,
    )
    ax8.set_title("Average Accuracy by Class", fontsize=16)
    ax8.set_xlabel("Average Accuracy", fontsize=14)
    ax8.set_ylabel("Class", fontsize=14)
    ax8.set_xlim(0.4, 1.05)

    plt.tight_layout()
    figures["avg_by_class"] = fig8

    if output_dir:
        fig8.savefig(Path(output_dir) / f"{sampling_selection}_avg_by_class.png")

    # Save a summary text file if output_dir is specified
    if output_dir:
        # Save a summary text file
        summary_file = Path(output_dir) / f"{sampling_selection}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(
                f"Model performance summary for {sampling_selection} (sorted by accuracy):\n"
            )
            f.write(
                model_summary[["model", "accuracy", "f1-score"]].to_string(index=False)
                + "\n\n"
            )
            f.write("Number of classes where each model performs best:\n")
            f.write(model_win_counts.to_string(index=False) + "\n")

        print(f"\nAll plots and summary saved to: {output_dir}")
        print(f"Summary file: {summary_file}")

    # Display a summary in the console
    print(f"Model performance summary (sorted by accuracy):")
    print(model_summary[["model", "accuracy", "f1-score"]].to_string(index=False))

    if len(model_win_counts) > 0:
        print("\nNumber of classes where each model performs best:")
        print(model_win_counts.to_string(index=False))
    else:
        print("\nNo model win counts available.")

    return figures
