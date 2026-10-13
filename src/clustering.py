"""
Clustering algorithms + evaluation metrics for household segmentation.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# --- Path handling: always resolve relative to the repo root ---
# __file__ = /path/to/repo/src/preprocess.py
# parents[0] = src/, parents[1] = repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"

def load_cluster_input():
    data = joblib.load(DATA_DIR / "cluster_input.joblib")
    return data["X_scaled"], data["meta"], data["feature_cols"]


def evaluate_k_range(X, k_range=range(2, 11)):
    """Run K-Means for each k and compute elbow + silhouette + DB."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return rows


def plot_k_selection(rows):
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(ks, [r["inertia"] for r in rows], "o-")
    ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")

    ax[1].plot(ks, [r["silhouette"] for r in rows], "o-", color="green")
    ax[1].set_title("Silhouette (higher = better)"); ax[1].set_xlabel("k")

    ax[2].plot(ks, [r["davies_bouldin"] for r in rows], "o-", color="red")
    ax[2].set_title("Davies-Bouldin (lower = better)"); ax[2].set_xlabel("k")

    plt.tight_layout()
    return fig


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def fit_dbscan(X, eps=0.8, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db.labels_, db


def fit_hierarchical(X, k, linkage="ward"):
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = hc.fit_predict(X)
    return labels, hc


def plot_clusters_pca(X, labels, title="Clusters (PCA projection)"):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=labels, cmap="tab10", s=60,
                         alpha=0.75, edgecolor="k", linewidth=0.5)
    ax.set_title(f"{title}\nPC1+PC2 = {pca.explained_variance_ratio_.sum():.1%} variance")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return fig, X_2d