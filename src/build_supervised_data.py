"""
Build a supervised dataset: predict next-hour kWh from household + time features.

Reuses:
- data/raw/malawi_households_hourly.csv (hourly consumption)
- data/processed/household_clusters_named.csv (unsupervised labels)

Output:
- data/processed/supervised_data.parquet  (or .csv)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "malawi_households_hourly.csv"
CLUSTERS = REPO_ROOT / "data" / "processed" / "household_clusters_named.csv"
OUT = REPO_ROOT / "data" / "processed" / "supervised_data.parquet"


def build():
    # --- Load hourly data ---
    print("Loading hourly consumption...")
    df = pd.read_csv(RAW, parse_dates=["timestamp"])
    df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)

    # --- Load cluster labels ---
    print("Loading cluster labels...")
    clusters = pd.read_csv(CLUSTERS)

    # Only keep the label columns we actually need.
    # ⚠️ Do NOT include 'area' — the hourly df already has 'area'.
    label_cols = [c for c in ["household_id", "cluster", "cluster_name"]
                  if c in clusters.columns]
    labels = clusters[label_cols].copy()

    df = df.merge(labels, on="household_id", how="left")
    # --- Build features ---
    print("Engineering features...")
    g = df.groupby("household_id")["consumption_kwh"]

    # Lag & rolling features (per household!)
    df["lag_1h"]  = g.shift(1)
    df["lag_24h"] = g.shift(24)
    df["rolling_24h_mean"] = g.transform(lambda s: s.rolling(24, min_periods=1).mean())

    # Cyclical time features
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Target: NEXT hour's consumption ---
    df["target"] = g.shift(-1)

    # --- Drop rows with missing values (first/last of each household) ---
    df = df.dropna(subset=["lag_1h", "lag_24h", "target"]).reset_index(drop=True)

    # --- Encode categoricals ---
    df["cluster_name"] = df["cluster_name"].fillna("unknown")
    df["area_is_urban"] = (df["area"] == "urban").astype(int)

    print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Target mean:   {df['target'].mean():.4f} kWh")
    print(f"Target std:    {df['target'].std():.4f} kWh")

    df.to_parquet(OUT, index=False)
    print(f"Saved: {OUT}")
    return df


if __name__ == "__main__":
    build()