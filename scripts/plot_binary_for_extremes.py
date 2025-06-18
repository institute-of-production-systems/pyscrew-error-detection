# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Daten laden
result_path = "../results/result_s06/separate_only_extreme"
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
    heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "F1-Score"}
)
plt.title("F1-Score Heatmap je Modell und Dataset")
plt.xlabel("Dataset")
plt.ylabel("Modell")
plt.tight_layout()
plt.show()


# Gruppieren nach Modell & Dataset, dann Mittelwert & Varianz berechnen
grouped = df_result.groupby(["model", "dataset"])["f1_score"].agg(["mean", "var"])


# Formatieren als "Mittelwert ± Varianz"
def format_row(row):
    return f"{row['mean']*100:.2f} ± {row['var']*100:.2f}"


# Tabelle mit formatierten Strings
table = grouped.apply(format_row, axis=1).unstack()

# Anzeige
print("F1-Score (Mittelwert ± Varianz) je Modell und Dataset")
table
