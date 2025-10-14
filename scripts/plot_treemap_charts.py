# %%
import matplotlib.pyplot as plt

# Daten
ok_count = 1200
nok_count = 133  # Summe aller NOK aus den nicht entfernten Klassen

# Balkendiagramm
fig, ax = plt.subplots(figsize=(12, 4))

# Horizontale Balken
bars = ax.barh(
    ["NOK", "OK"],
    [nok_count, ok_count],
    color=["firebrick", "forestgreen"],
    alpha=0.95,
    height=0.75,
)

# Beschriftung
ax.set_xlabel("Anzahl", fontsize=18)
ax.set_ylabel("")
# ax.set_title('Gesamtverteilung: OK vs. NOK', fontsize=16, weight='bold', pad=20)

# Zahlen rechts von den Balken
ax.text(
    nok_count + 30,
    0,
    f"{f'{nok_count:,}'.replace(',', '.')}",
    va="center",
    ha="left",
    fontsize=18,
    weight="bold",
)
ax.text(
    ok_count + 30,
    1,
    f"{f'{ok_count:,}'.replace(',', '.')}",
    va="center",
    ha="left",
    fontsize=18,
    weight="bold",
)

# Grid für bessere Lesbarkeit
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)

# X-Achse bis zur größten Zahl + etwas Platz
ax.set_xlim(0, max(ok_count, nok_count) * 1.15)

plt.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# Zusätzliche Info ausgeben
print(f"OK: {ok_count}")
print(f"NOK: {nok_count}")
print(f"Gesamt: {ok_count + nok_count}")
print(f"Erfolgsrate: {ok_count/(ok_count + nok_count)*100:.1f}%")
