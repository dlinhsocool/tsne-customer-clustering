"""
models/clustering.py
====================
Người phụ trách: Duy Lĩnh (mô hình hóa & thực nghiệm)

NHIỆM VỤ FILE: chạy K-Means, GMM, DBSCAN trên một không gian đặc trưng
("original" | "pca" | "umap" | "tsne") và trả về định dạng ClusteringResult thống nhất.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from utils.constants import (
    DEFAULT_PARAMS,
    FEATURE_SPACES,
    NOISE_LABEL,
    RANDOM_STATE,
    SUPPORTS_PREDICT,
)


def estimate_dbscan_eps(
    X: np.ndarray,
    min_samples: int = 5,
    quantile: float = 90.0,
) -> float:
    """Ước lượng eps phù hợp cho từng không gian dựa trên k-distance graph."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or len(X_arr) < 2:
        return 0.5
    n_neighbors = max(1, min(int(min_samples) - 1, len(X_arr) - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X_arr)
    dists = nn.kneighbors(return_distance=True)[0][:, -1]
    eps = float(np.percentile(dists, quantile))
    if not np.isfinite(eps) or eps <= 0:
        pos = dists[dists > 0]
        eps = float(np.median(pos)) if len(pos) > 0 else 0.5
    return round(eps, 4)


def make_clustering_result(
    algorithm: str,
    model: Any,
    labels: np.ndarray,
    params: dict,
    runtime_seconds: float | None = None,
    cluster_centers: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    extra: dict | None = None,
) -> dict:
    """Đóng gói kết quả phân cụm theo đúng hợp đồng."""
    if algorithm not in SUPPORTS_PREDICT:
        raise ValueError(f"algorithm phải thuộc {list(SUPPORTS_PREDICT)}, nhận '{algorithm}'")
    labels = np.asarray(labels).astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {int(u): int(c) for u, c in zip(unique, counts)}
    n_noise = int(sizes.get(NOISE_LABEL, 0))
    n_clusters = int(len([u for u in unique if u != NOISE_LABEL]))
    full_extra = {"inertia": None, "bic": None, "aic": None, "core_sample_indices": None}
    full_extra.update(extra or {})
    return {
        "algorithm": algorithm,
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": sizes,
        "cluster_centers": cluster_centers,
        "probabilities": probabilities,
        "params": dict(params),
        "feature_space": None,
        "supports_predict": SUPPORTS_PREDICT[algorithm],
        "runtime_seconds": runtime_seconds,
        "extra": full_extra,
    }


def run_clustering(
    X: np.ndarray,
    algorithm: str,
    params: dict | None = None,
    feature_space: str = "original",
    random_state: int = RANDOM_STATE,
) -> dict:
    """Hàm điều phối phân cụm trên không gian 'original' | 'pca' | 'umap' | 'tsne'."""
    if algorithm not in DEFAULT_PARAMS:
        raise ValueError(f"algorithm phải thuộc {list(DEFAULT_PARAMS)}, nhận '{algorithm}'")

    normalized_space = "original" if feature_space == "scaled" else feature_space
    if normalized_space not in FEATURE_SPACES:
        raise ValueError(
            f"feature_space phải thuộc {list(FEATURE_SPACES)}, nhận '{feature_space}'"
        )

    merged = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    if algorithm == "kmeans":
        result = run_kmeans(X, random_state=random_state, **merged)
    elif algorithm == "gmm":
        result = run_gmm(X, random_state=random_state, **merged)
    else:
        if merged.get("eps") is None:
            merged["eps"] = estimate_dbscan_eps(X, min_samples=int(merged.get("min_samples", 5)))
        result = run_dbscan(X, **merged)

    result["feature_space"] = normalized_space
    return result


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> dict:
    """K-Means."""
    t0 = time.perf_counter()
    model = KMeans(
        n_clusters=int(n_clusters),
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(random_state),
    ).fit(X)
    runtime = time.perf_counter() - t0

    params = {
        "n_clusters": int(n_clusters),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    return make_clustering_result(
        "kmeans",
        model,
        model.labels_,
        params,
        runtime,
        cluster_centers=model.cluster_centers_,
        probabilities=None,
        extra={"inertia": float(model.inertia_)},
    )


def run_gmm(
    X: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "full",
    max_iter: int = 100,
    random_state: int = 42,
) -> dict:
    """Gaussian Mixture Model."""
    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=int(n_components),
        covariance_type=covariance_type,
        max_iter=int(max_iter),
        random_state=int(random_state),
    ).fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)
    runtime = time.perf_counter() - t0

    params = {
        "n_components": int(n_components),
        "covariance_type": covariance_type,
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    return make_clustering_result(
        "gmm",
        model,
        labels,
        params,
        runtime,
        cluster_centers=model.means_,
        probabilities=probs,
        extra={"bic": float(model.bic(X)), "aic": float(model.aic(X))},
    )


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict:
    """DBSCAN. Nhãn noise = -1; không có predict (supports_predict=False)."""
    t0 = time.perf_counter()
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X)
    runtime = time.perf_counter() - t0

    params = {"eps": float(eps), "min_samples": int(min_samples)}
    return make_clustering_result(
        "dbscan",
        model,
        model.labels_,
        params,
        runtime,
        cluster_centers=None,
        probabilities=None,
        extra={"core_sample_indices": model.core_sample_indices_},
    )