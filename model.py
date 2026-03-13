import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

FIG_DIR   = Path("figures");  FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models");   MODEL_DIR.mkdir(exist_ok=True)

def save(name):
    plt.savefig(FIG_DIR / name)
    plt.close()
    print(f"  [FIG] figures/{name}")


df = pd.read_csv("flights_clean.csv", parse_dates=["date", "departure", "arrival"])
print(f"{df.shape[0]} lignes\n")

le_airline    = LabelEncoder()
df["airline_enc"] = le_airline.fit_transform(df["airline"])

le_dow        = LabelEncoder()
df["dow_enc"] = le_dow.fit_transform(df["day_of_week"])

FEATURES = ["airline_enc", "stops", "duration_min",
            "days_until_dep", "is_direct", "dow_enc", "month", "week"]


# ── régression ──

X = df[FEATURES]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"train : {len(X_train)} | test : {len(X_test)}")

# la régression linéaire a besoin du scaler, les arbres non
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models_reg = {
    "Régression linéaire": LinearRegression(),
    "Arbre de décision":   DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest":       RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
}

results_reg = {}
for name, model in models_reg.items():
    if name == "Régression linéaire":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results_reg[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "pred": y_pred}
    print(f"  {name:25s} → MAE={mae:.2f}€  RMSE={rmse:.2f}€  R²={r2:.3f}")

best_reg_name = max(results_reg, key=lambda k: results_reg[k]["R2"])
best_reg      = results_reg[best_reg_name]
print(f"\nmeilleur : {best_reg_name} (R²={best_reg['R2']:.3f})")

joblib.dump(results_reg["Random Forest"]["model"], MODEL_DIR / "rf_regressor.pkl")
joblib.dump(scaler,     MODEL_DIR / "scaler.pkl")
joblib.dump(le_airline, MODEL_DIR / "le_airline.pkl")

# comparaison des 4 modèles — rouge = meilleur
names  = list(results_reg.keys())
colors = ["#C44E52" if n == best_reg_name else "#4C72B0" for n in names]
short  = ["Lin. Reg.", "Arbre", "Rnd Forest", "Grad. Boost"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(short, [results_reg[n]["MAE"]  for n in names], color=colors)
axes[0].set_title("MAE (€) — plus bas = mieux"); axes[0].set_ylabel("€")
axes[1].bar(short, [results_reg[n]["RMSE"] for n in names], color=colors)
axes[1].set_title("RMSE (€) — plus bas = mieux")
axes[2].bar(short, [results_reg[n]["R2"]   for n in names], color=colors)
axes[2].set_title("R² — plus haut = mieux"); axes[2].set_ylim(0, 1)
for ax in axes:
    plt.sca(ax); plt.xticks(rotation=20, ha="right", fontsize=9)
fig.suptitle("Comparaison des modèles de régression", fontsize=13, y=1.02)
plt.tight_layout()
save("09_comparaison_regression.png")

y_pred_best = best_reg["pred"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_best, alpha=0.5, color="#4C72B0", s=20)
lims = [min(y_test.min(), y_pred_best.min()) - 10,
        max(y_test.max(), y_pred_best.max()) + 10]
ax.plot(lims, lims, "r--", linewidth=1, label="Prédiction parfaite")
ax.set_xlabel("Prix réel (€)"); ax.set_ylabel("Prix prédit (€)")
ax.set_title(f"Prédit vs Réel — {best_reg_name}")
ax.legend()
save("10_predit_vs_reel.png")

importances = pd.Series(
    results_reg["Random Forest"]["model"].feature_importances_, index=FEATURES
).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", color="#4C72B0", ax=ax)
ax.set_xlabel("Importance")
ax.set_title("Importance des features — Random Forest Régression")
plt.tight_layout()
save("11_feature_importance_reg.png")


# ── classification ──

# cible : est-ce que le prix minimum va baisser dans les 3 prochains jours ?
df_sorted     = df.sort_values("date").copy()
prix_par_date = df_sorted.groupby("date")["price"].min().reset_index()
prix_par_date.columns = ["date", "prix_min"]
prix_par_date["prix_min_j3"] = prix_par_date["prix_min"].shift(-3)
prix_par_date["baisse"]      = (prix_par_date["prix_min_j3"] < prix_par_date["prix_min"]).astype(int)
prix_par_date = prix_par_date.dropna()

df2 = df_sorted.merge(prix_par_date[["date", "baisse"]], on="date", how="inner")
print(f"\nclassification : {len(df2)} lignes")
print(f"baisse=1 : {df2['baisse'].sum()} ({df2['baisse'].mean()*100:.1f}%)  "
      f"hausse=0 : {(1-df2['baisse']).sum()} ({(1-df2['baisse']).mean()*100:.1f}%)")

X2 = df2[FEATURES]
y2 = df2["baisse"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

scaler2      = StandardScaler()
X2_train_sc  = scaler2.fit_transform(X2_train)
X2_test_sc   = scaler2.transform(X2_test)

models_clf = {
    "Régression logistique": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":         RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
}

results_clf = {}
for name, model in models_clf.items():
    if name == "Régression logistique":
        model.fit(X2_train_sc, y2_train)
        y_pred = model.predict(X2_test_sc)
    else:
        model.fit(X2_train, y2_train)
        y_pred = model.predict(X2_test)

    acc = accuracy_score(y2_test, y_pred)
    results_clf[name] = {"acc": acc, "model": model, "pred": y_pred}
    print(f"\n  {name}  accuracy={acc:.3f}")
    print(classification_report(y2_test, y_pred,
                                 target_names=["Hausse/stable", "Baisse"],
                                 zero_division=0))

best_clf_name = max(results_clf, key=lambda k: results_clf[k]["acc"])
print(f"meilleur : {best_clf_name}")

joblib.dump(results_clf["Random Forest"]["model"], MODEL_DIR / "rf_classifier.pkl")

cm = confusion_matrix(y2_test, results_clf["Random Forest"]["pred"])
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Hausse/stable", "Baisse"]).plot(
    ax=ax, colorbar=False, cmap="Blues"
)
ax.set_title("Matrice de confusion — Random Forest")
save("12_confusion_matrix.png")

imp_clf = pd.Series(
    results_clf["Random Forest"]["model"].feature_importances_, index=FEATURES
).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
imp_clf.plot(kind="barh", color="#55A868", ax=ax)
ax.set_xlabel("Importance")
ax.set_title("Importance des features — Random Forest Classification")
plt.tight_layout()
save("13_feature_importance_clf.png")

# résumé
print("\nrégression :")
for name, r in results_reg.items():
    print(f"  {name:25s} MAE={r['MAE']:.1f}€  R²={r['R2']:.3f}")

print("\nclassification :")
for name, r in results_clf.items():
    print(f"  {name:25s} accuracy={r['acc']:.3f}")