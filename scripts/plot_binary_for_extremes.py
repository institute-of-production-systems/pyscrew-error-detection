# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run(scenario):

    # Daten laden
    result_path = f"../results/results_s06/{scenario}.csv"
    df_result = pd.read_csv(result_path, index_col=0)

    # Pivot-Tabelle: Mittelwert des F1-Scores je Modell & Dataset
    heatmap_data = (
        df_result.groupby(["model", "dataset"])["f1_score"]
        .mean()
        .unstack()  # Modelle als Zeilen, Datasets als Spalten
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "F1-Score"},
    )
    plt.title(f"'{scenario}': F1-Score Heatmap je Modell und Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Modell")
    plt.tight_layout()
    plt.show()

    # Gruppieren nach Modell & Dataset, dann Mittelwert & Varianz berechnen
    grouped = df_result.groupby(["model", "dataset"])["f1_score"].agg(["mean", "std"])

    # Formatieren als "Mittelwert ± Varianz"
    def format_row(row):
        return f"{row['mean']*100:.2f} ± {row['std']*100:.2f}"

    # Tabelle mit formatierten Strings
    table = grouped.apply(format_row, axis=1).unstack()

    # Anzeige
    print("F1-Score (Mittelwert ± Varianz) je Modell und Dataset")
    print(scenario)
    print(table)


for scenario in [
    "combined_all_classes",
    "combined_only_extreme",
    "separate_all_classes",
    "separate_only_extreme",
]:
    run(scenario)

    # %%

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run(scenario):
    # Daten laden
    result_path = f"../results/results_s06/{scenario}.csv"
    df_result = pd.read_csv(result_path, index_col=0)

    # Define metrics to analyze
    metrics = ["f1_score", "accuracy", "precision", "recall"]

    # Create subplots for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        # Pivot-Tabelle: Mittelwert je Modell & Dataset
        heatmap_data = (
            df_result.groupby(["model", "dataset"])[metric]
            .mean()
            .unstack()  # Modelle als Zeilen, Datasets als Spalten
        )

        # Plot
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": metric.replace("_", " ").title()},
            ax=axes[idx],
        )
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}")
        axes[idx].set_xlabel("Dataset")
        axes[idx].set_ylabel("Model")

    fig.suptitle(f"'{scenario}': Performance Metrics Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Create comprehensive table with all metrics
    print(f"\nPerformance Metrics for scenario: {scenario}")
    print("=" * 80)

    for metric in metrics:
        # Gruppieren nach Modell & Dataset, dann Mittelwert & Varianz berechnen
        grouped = df_result.groupby(["model", "dataset"])[metric].agg(["mean", "std"])

        # Formatieren als "Mittelwert ± Varianz"
        def format_row(row):
            return f"{row['mean']*100:.2f} ± {row['std']*100:.2f}"

        # Tabelle mit formatierten Strings
        table = grouped.apply(format_row, axis=1).unstack()

        # Anzeige
        print(f"\n{metric.upper().replace('_', ' ')} (Mean ± Std)")
        print(table)

    print("\n" + "=" * 80 + "\n")


# Run for all scenarios
for scenario in [
    "combined_all_classes",
    "combined_only_extreme",
    "separate_all_classes",
    "separate_only_extreme",
]:
    run(scenario)

# %%
