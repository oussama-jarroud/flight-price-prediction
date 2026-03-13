import pandas as pd
import numpy as np

df = pd.read_csv("flights_raw.csv")
print(f"brut : {df.shape}")

# la date vient du scraper au format YYMMDD (ex: 260313)
df["date_str"]  = df["date"].astype(str).str.zfill(6)
df["date"]      = pd.to_datetime(df["date_str"], format="%y%m%d", errors="coerce")
df["departure"] = pd.to_datetime(df["departure"], errors="coerce")
df["arrival"]   = pd.to_datetime(df["arrival"],   errors="coerce")
df["scraped_at"]= pd.to_datetime(df["scraped_at"], errors="coerce")

df["price"]        = pd.to_numeric(df["price"],        errors="coerce")
df["stops"]        = pd.to_numeric(df["stops"],        errors="coerce").astype("Int64")
df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")

print(df.dtypes)

# valeurs manquantes
print(df.isnull().sum())

before = len(df)
df = df.dropna(subset=["price"])
print(f"lignes sans prix supprimées : {before - len(df)}")

df["airline"] = df["airline"].fillna("Unknown").replace("", "Unknown")

# médiane par compagnie plutôt que médiane globale — plus cohérent
df["duration_min"] = df.groupby("airline")["duration_min"] \
                       .transform(lambda x: x.fillna(x.median()))
df["duration_min"] = df["duration_min"].fillna(df["duration_min"].median())

df["stops"] = df["stops"].fillna(0).astype(int)

# doublons
before = len(df)
df = df.drop_duplicates(subset=["date", "price", "airline", "departure"])
print(f"doublons supprimés : {before - len(df)}")

# outliers prix — percentiles 1% et 99% pour garder les cas extrêmes légitimes
q_low  = df["price"].quantile(0.01)
q_high = df["price"].quantile(0.99)
before = len(df)
df = df[(df["price"] >= q_low) & (df["price"] <= q_high)]
print(f"outliers supprimés (< {q_low:.0f}€ ou > {q_high:.0f}€) : {before - len(df)}")

# features pour le modèle ML
df["day_of_week"]    = df["date"].dt.day_name()
df["days_until_dep"] = (df["date"] - df["scraped_at"].dt.normalize()).dt.days
df["is_direct"]      = (df["stops"] == 0).astype(int)
df["duration_h"]     = (df["duration_min"] / 60).round(2)
df["month"]          = df["date"].dt.month
df["week"]           = df["date"].dt.isocalendar().week.astype(int)

df.to_csv("flights_clean.csv", index=False, encoding="utf-8")
df.to_parquet("flights_clean.parquet", index=False)

print(f"\n{df.shape[0]} lignes × {df.shape[1]} colonnes")
print(df[["date", "price", "airline", "stops", "duration_h", "is_direct", "day_of_week"]].head(10).to_string())