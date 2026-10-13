"""
Preprocessing for household clustering.

Takes the EDA output (household_features.csv) and produces a scaled,
outlier-handled feature matrix ready for K-Means / DBSCAN / Hierarchical.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# --- Design decisions (from Phase 2 EDA) ---
CLUSTER_FEATURES = [
    "mean_daily_kwh",     # level (Cohen's d = 1.47)
    "max_hourly_kwh",     # peak (d = 1.68, r = 0.74 with above)
    "midday_ratio",       # shape: daytime activity (d = 0.79)
    "volatility_cv",      # shape: erraticness
    "evening_ratio",      # shape: evening peak strength
]

# --- Path handling: always resolve relative to the repo root ---
# __file__ = /path/to/repo/src/preprocess.py
# parents[0] = src/, parents[1] = repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"

def load_household_features() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "household_features.csv")


def cap_outliers(df: pd.DataFrame, cols: list[str],
                 lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Cap extreme values at the 1st/99th percentile.

    Why: a handful of urban households use 15+ kWh/day, and K-Means
    (distance-based) will let those dominate. Capping preserves the
    ranking while taming the influence.
    """
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)
    return df


def build_cluster_matrix(save: bool = True):
    """
    Returns (X_scaled, df_with_labels_kept, scaler, feature_names).
    """
    df = load_household_features()

    # 1. Keep the metadata we want for later interpretation
    meta_cols = ["household_id", "area", "region", "household_size", "is_urban"]
    meta = df[meta_cols].copy()

    # 2. Select clustering features
    X_raw = df[CLUSTER_FEATURES].copy()

    # 3. Cap outliers
    X_capped = cap_outliers(X_raw, CLUSTER_FEATURES)

    # 4. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_capped)

    # 5. Save artifacts
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"X_scaled": X_scaled,
             "meta": meta,
             "scaler": scaler,
             "feature_cols": CLUSTER_FEATURES},
            DATA_DIR / "cluster_input.joblib"
        )
        # Also save a CSV view for easy inspection
        X_view = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)
        X_view = pd.concat([meta, X_view], axis=1)
        X_view.to_csv(DATA_DIR / "cluster_input_scaled.csv", index=False)

    return X_scaled, meta, scaler, CLUSTER_FEATURES


if __name__ == "__main__":
    X, meta, scaler, cols = build_cluster_matrix()
    print(f"X shape: {X.shape}")
    print(f"Features: {cols}")
    print(f"\nScaled means (should be ~0): {X.mean(axis=0).round(3)}")
    print(f"Scaled stds  (should be ~1): {X.std(axis=0).round(3)}")
    print(f"\nMetadata columns: {list(meta.columns)}")
    print(f"Rows: {len(meta)}")
    print(f"\nSaved to data/processed/cluster_input.joblib")