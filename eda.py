import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def save(name):
    plt.savefig(FIG_DIR / name)
    plt.close()
    print(f"  [FIG] figures/{name}")


df = pd.read_csv("flights_clean.csv", parse_dates=["date", "departure", "arrival"])
print(f"{df.shape[0]} lignes × {df.shape[1]} colonnes")

print(df[["price", "duration_min", "stops", "days_until_dep"]].describe().round(2))

# distribution des prix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["price"], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Prix (€)")
axes[0].set_ylabel("Nombre de vols")
axes[0].set_title("Distribution des prix")
axes[1].boxplot(df["price"], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.7))
axes[1].set_ylabel("Prix (€)")
axes[1].set_title("Boxplot des prix")
save("01_distribution_prix.png")

# évolution sur 30 jours
prix_date = df.groupby("date")["price"].agg(["min", "mean", "max"]).reset_index()
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(prix_date["date"], prix_date["min"], prix_date["max"],
                alpha=0.2, color="#4C72B0", label="min–max")
ax.plot(prix_date["date"], prix_date["mean"], color="#4C72B0", linewidth=2, label="Prix moyen")
ax.plot(prix_date["date"], prix_date["min"],  color="#55A868", linewidth=1,
        linestyle="--", label="Prix min")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=45)
ax.set_xlabel("Date de départ")
ax.set_ylabel("Prix (€)")
ax.set_title("Évolution des prix CMN→MAD sur 30 jours")
ax.legend()
save("02_evolution_prix.png")

# top 8 compagnies triées par médiane
top_airlines = df.groupby("airline")["price"].count().nlargest(8).index
df_top = df[df["airline"].isin(top_airlines)]
order = df_top.groupby("airline")["price"].median().sort_values().index
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df_top, x="airline", y="price", order=order,
            hue="airline", palette="muted", legend=False, ax=ax)
ax.set_xlabel("Compagnie aérienne")
ax.set_ylabel("Prix (€)")
ax.set_title("Distribution des prix par compagnie")
plt.xticks(rotation=30, ha="right")
save("03_prix_par_compagnie.png")

order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=df, x="day_of_week", y="price", order=order_days,
            hue="day_of_week", palette="coolwarm", legend=False,
            estimator=np.mean, ax=ax)
ax.set_xlabel("Jour de la semaine")
ax.set_ylabel("Prix moyen (€)")
ax.set_title("Prix moyen par jour de la semaine")
plt.xticks(rotation=30, ha="right")
save("04_prix_par_jour.png")

df["type_vol"] = df["stops"].apply(lambda x: "Direct" if x == 0 else f"{x} escale(s)")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="type_vol", y="price",
            hue="type_vol", palette="Set2", legend=False, ax=ax)
ax.set_xlabel("Type de vol")
ax.set_ylabel("Prix (€)")
ax.set_title("Prix selon le nombre d'escales")
save("05_prix_escales.png")

num_cols = ["price", "stops", "duration_min", "days_until_dep"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Matrice de corrélation")
save("06_correlation.png")

prix_min = df.groupby("date")["price"].min().reset_index()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(prix_min["date"], prix_min["price"], marker="o", markersize=4,
        color="#C44E52", linewidth=1.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
ax.set_xlabel("Date de départ")
ax.set_ylabel("Prix minimum (€)")
ax.set_title("Prix minimum disponible par date")
ax.axhline(prix_min["price"].mean(), color="gray", linestyle="--",
           linewidth=1, label=f"Moyenne : {prix_min['price'].mean():.0f}€")
ax.legend()
save("07_serie_temporelle_min.png")

n_vols = df.groupby("date").size().reset_index(name="n_vols")
fig, ax = plt.subplots(figsize=(14, 3))
ax.bar(n_vols["date"], n_vols["n_vols"], color="#8172B2", width=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=45)
ax.set_xlabel("Date")
ax.set_ylabel("Nombre de vols")
ax.set_title("Nombre de vols disponibles par date")
save("08_nb_vols_par_date.png")

# résumé
print(f"\npériode  : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"vols     : {len(df)}")
print(f"compagnies : {df['airline'].nunique()} — {list(df['airline'].unique())}")
print(f"prix     : min {df['price'].min():.0f}€  moy {df['price'].mean():.0f}€  max {df['price'].max():.0f}€")
print(f"directs  : {df['is_direct'].sum()} ({df['is_direct'].mean()*100:.1f}%)")
print(f"corr stops↔prix   : {corr.loc['stops','price']:.3f}")
print(f"corr durée↔prix   : {corr.loc['duration_min','price']:.3f}")