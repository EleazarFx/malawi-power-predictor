"""
Train two RandomForest models:
  Baseline: time + lag features ONLY (no cluster info)
  Ours:     time + lag + cluster label

Compares MAE and R² — the delta IS the value of unsupervised learning.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed" / "supervised_data.parquet"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


TIME_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rainy_season",
    "lag_1h", "lag_24h", "rolling_24h_mean",
]

CLUSTER_FEATURES = ["cluster", "area_is_urban"]   # numeric-encoded


def load_data():
    print("Loading supervised data...")
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} rows")
    return df


def train_and_evaluate(df):
    y = df["target"].values

    # --- Split by TIME, not randomly ---
    # Random split would leak future data. Use time-ordered split.
    split = int(len(df) * 0.8)
    X_train_time = df.iloc[:split]
    X_test_time  = df.iloc[split:]

    # ---- Baseline model (no clusters) ----
    print("\n[1/2] Training BASELINE (no cluster features)...")
    Xb_train = X_train_time[TIME_FEATURES]
    Xb_test  = X_test_time[TIME_FEATURES]
    yb_train = y[:split]
    yb_test  = y[split:]

    baseline = RandomForestRegressor(
        n_estimators=60, max_depth=12, n_jobs=-1, random_state=42
    )
    baseline.fit(Xb_train, yb_train)
    pred_b = baseline.predict(Xb_test)

    mae_b = mean_absolute_error(yb_test, pred_b)
    r2_b  = r2_score(yb_test, pred_b)
    print(f"  MAE: {mae_b:.5f} kWh   R²: {r2_b:.4f}")

    # ---- Our model (with clusters) ----
    print("\n[2/2] Training OURS (time + cluster features)...")
    feats_full = TIME_FEATURES + CLUSTER_FEATURES
    X_train = X_train_time[feats_full]
    X_test  = X_test_time[feats_full]

    ours = RandomForestRegressor(
        n_estimators=60, max_depth=12, n_jobs=-1, random_state=42
    )
    ours.fit(X_train, y[:split])
    pred_o = ours.predict(X_test)

    mae_o = mean_absolute_error(yb_test, pred_o)
    r2_o  = r2_score(yb_test, pred_o)
    print(f"  MAE: {mae_o:.5f} kWh   R²: {r2_o:.4f}")

    # ---- Improvement ----
    mae_gain = (mae_b - mae_o) / mae_b * 100
    r2_gain  = (r2_o - r2_b) * 100
    print(f"\n>>> Improvement from cluster features:")
    print(f"    MAE: {mae_gain:+.1f}%")
    print(f"    R²:  {r2_gain:+.2f} percentage points")

    # ---- Save models ----
    joblib.dump({"model": ours, "features": feats_full}, MODEL_DIR / "predictor_ours.joblib")
    joblib.dump({"model": baseline, "features": TIME_FEATURES}, MODEL_DIR / "predictor_baseline.joblib")
    print(f"\nSaved models to {MODEL_DIR}")

    return {
        "baseline": {"mae": mae_b, "r2": r2_b},
        "ours":     {"mae": mae_o, "r2": r2_o},
        "mae_gain_pct": mae_gain,
        "r2_gain_pp": r2_gain,
    }


if __name__ == "__main__":
    df = load_data()
    results = train_and_evaluate(df)