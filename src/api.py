"""
FastAPI backend for the Malawi Household Electricity dashboard.

Serves:
  GET  /                  -> dashboard HTML (index.html)
  GET  /api/clusters      -> cluster summary (for the dashboard)
  GET  /api/cluster_plot  -> base64 PNG of the PCA cluster plot
  GET  /api/predict_plot  -> base64 PNG comparing baseline vs ours
  POST /api/predict       -> given features, returns predicted kWh
"""

from __future__ import annotations
from pathlib import Path
import base64
import io
import json
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")     # headless backend — critical for servers
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure src/ is on the Python path regardless of where uvicorn is launched from
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
    
from clustering import load_cluster_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data" / "processed"
STATIC = REPO_ROOT / "static"

app = FastAPI(title="Malawi Household Electricity Predictor")


# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
print("Loading model & data...")
artifact = joblib.load(MODELS_DIR / "predictor_ours.joblib")
MODEL = artifact["model"]
MODEL_FEATURES = artifact["features"]

X_scaled, meta, feature_cols = load_cluster_input()
CLUSTERS = pd.read_csv(DATA_DIR / "household_clusters_named.csv")

print(f"Model ready. Features: {MODEL_FEATURES}")
print(f"Clusters loaded: {len(CLUSTERS)} households")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fig_to_base64(fig) -> str:
    """Render a matplotlib figure as base64 PNG (for embedding in HTML)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the dashboard HTML."""
    html_path = STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found. Create static/index.html</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/clusters")
def api_clusters():
    """Cluster summary as JSON."""
    summary = (
        CLUSTERS.groupby("cluster_name")
        .agg(
            n=("household_id", "count"),
            urban_pct=("is_urban", "mean"),
            mean_kwh=("mean_daily_kwh", "mean"),
            peak_kwh=("max_hourly_kwh", "mean"),
        )
        .round(3)
        .reset_index()
    )
    return JSONResponse(summary.to_dict(orient="records"))


@app.get("/api/cluster_plot")
def api_cluster_plot():
    """PCA scatter of households colored by cluster."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    labels = CLUSTERS.set_index("household_id").loc[meta["household_id"], "cluster"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="tab10", s=60, alpha=0.8, edgecolor="k", linewidth=0.4,
    )
    ax.set_title(f"Household clusters (PCA)\nPC1+PC2 = {pca.explained_variance_ratio_.sum():.1%} variance")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    return {"image": fig_to_base64(fig)}


@app.get("/api/predict_plot")
def api_predict_plot():
    """Bar chart: baseline vs ours (MAE, R²)."""
    # Hardcoded from training output — you could also save these to a JSON at train time.
    metrics = {
        "Baseline": {"MAE": 0.11483, "R2": 0.1051},
        "Ours (+clusters)": {"MAE": 0.11266, "R2": 0.1171},
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(metrics.keys())

    axes[0].bar(names, [metrics[n]["MAE"] for n in names], color=["#4c72b0", "#55a868"])
    axes[0].set_title("MAE (lower is better)")
    axes[0].set_ylabel("kWh")
    for i, n in enumerate(names):
        axes[0].text(i, metrics[n]["MAE"], f'{metrics[n]["MAE"]:.5f}', ha="center", va="bottom")

    axes[1].bar(names, [metrics[n]["R2"] for n in names], color=["#4c72b0", "#55a868"])
    axes[1].set_title("R² (higher is better)")
    for i, n in enumerate(names):
        axes[1].text(i, metrics[n]["R2"], f'{metrics[n]["R2"]:.4f}', ha="center", va="bottom")

    plt.tight_layout()
    return {"image": fig_to_base64(fig)}


class PredictInput(BaseModel):
    hour: int
    day_of_week: int
    is_weekend: int
    is_rainy_season: int
    lag_1h: float
    lag_24h: float
    rolling_24h_mean: float
    cluster: int
    area_is_urban: int


@app.post("/api/predict")
def api_predict(payload: PredictInput):
    """Predict next-hour consumption for a household."""
    # Build feature vector in the exact order the model expects
    hour_sin = np.sin(2 * np.pi * payload.hour / 24)
    hour_cos = np.cos(2 * np.pi * payload.hour / 24)
    dow_sin  = np.sin(2 * np.pi * payload.day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * payload.day_of_week / 7)

    row = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": payload.is_weekend,
        "is_rainy_season": payload.is_rainy_season,
        "lag_1h": payload.lag_1h,
        "lag_24h": payload.lag_24h,
        "rolling_24h_mean": payload.rolling_24h_mean,
        "cluster": payload.cluster,
        "area_is_urban": payload.area_is_urban,
    }
    X = pd.DataFrame([row])[MODEL_FEATURES]
    y_hat = float(MODEL.predict(X)[0])
    return {"prediction_kwh": round(y_hat, 4)}