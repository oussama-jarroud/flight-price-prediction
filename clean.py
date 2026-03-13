"""
Projet 11 - Phase 3 : Nettoyage & structuration
Input  : flights_raw.csv
Outputs: flights_clean.csv + flights_clean.parquet
"""

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════
df = pd.read_csv("flights_raw.csv")
print(f"Dimensions brutes : {df.shape}")
print(f"Colonnes : {list(df.columns)}")

# ══════════════════════════════════════════════
# 3.1 CONVERSION DES TYPES
# ══════════════════════════════════════════════
df["date_str"]  = df["date"].astype(str).str.zfill(6)
df["date"]      = pd.to_datetime(df["date_str"], format="%y%m%d", errors="coerce")
df["departure"] = pd.to_datetime(df["departure"], errors="coerce")
df["arrival"]   = pd.to_datetime(df["arrival"],   errors="coerce")
df["scraped_at"]= pd.to_datetime(df["scraped_at"], errors="coerce")

df["price"]        = pd.to_numeric(df["price"],        errors="coerce")
df["stops"]        = pd.to_numeric(df["stops"],        errors="coerce").astype("Int64")
df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")

print("\n--- Types après conversion ---")
print(df.dtypes)

# ══════════════════════════════════════════════
# 3.2 VALEURS MANQUANTES
# ══════════════════════════════════════════════
print("\n--- Valeurs manquantes avant nettoyage ---")
print(df.isnull().sum())

# Prix manquant → ligne inutilisable pour ML
before = len(df)
df = df.dropna(subset=["price"])
print(f"\nLignes supprimées (prix manquant) : {before - len(df)}")

# Compagnie vide → "Unknown"
df["airline"] = df["airline"].fillna("Unknown").replace("", "Unknown")

# Duration manquante → médiane par compagnie
df["duration_min"] = df.groupby("airline")["duration_min"] \
                       .transform(lambda x: x.fillna(x.median()))
df["duration_min"] = df["duration_min"].fillna(df["duration_min"].median())

# Stops manquants → 0
df["stops"] = df["stops"].fillna(0).astype(int)

print("\n--- Valeurs manquantes après traitement ---")
print(df.isnull().sum())

# ══════════════════════════════════════════════
# 3.3 DOUBLONS
# ══════════════════════════════════════════════
before = len(df)
df = df.drop_duplicates(subset=["date", "price", "airline", "departure"])
print(f"\nDoublons supprimés : {before - len(df)}")

# ══════════════════════════════════════════════
# 3.4 OUTLIERS PRIX
# ══════════════════════════════════════════════
q_low  = df["price"].quantile(0.01)
q_high = df["price"].quantile(0.99)
before = len(df)
df = df[(df["price"] >= q_low) & (df["price"] <= q_high)]
print(f"Outliers supprimés (< {q_low:.0f}€ ou > {q_high:.0f}€) : {before - len(df)}")

# ══════════════════════════════════════════════
# 3.5 FEATURES DÉRIVÉES
# ══════════════════════════════════════════════
df["day_of_week"]    = df["date"].dt.day_name()
df["days_until_dep"] = (df["date"] - df["scraped_at"].dt.normalize()).dt.days
df["is_direct"]      = (df["stops"] == 0).astype(int)
df["duration_h"]     = (df["duration_min"] / 60).round(2)
df["month"]          = df["date"].dt.month
df["week"]           = df["date"].dt.isocalendar().week.astype(int)

# ══════════════════════════════════════════════
# 3.6 EXPORT
# ══════════════════════════════════════════════
df.to_csv("flights_clean.csv", index=False, encoding="utf-8")
df.to_parquet("flights_clean.parquet", index=False)

print(f"\nDataset nettoyé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"Sauvegardé : flights_clean.csv + flights_clean.parquet")
print("\nAperçu :")
print(df[["date","price","airline","stops","duration_h","is_direct","day_of_week"]].head(10).to_string())