"""
models/clustering.py
====================
Người phụ trách: Duy Lĩnh (mô hình hóa & thực nghiệm)

NHIỆM VỤ FILE: chạy K-Means và trả về MỘT định dạng kết quả chung.

QUY TẮC QUAN TRỌNG:
  * Phân cụm chạy trên X_scaled hoặc X_pca (đa chiều). KHÔNG chạy trên tọa độ t-SNE.
  * K-Means không có noise.
  * labels là np.ndarray int (n,), CÙNG THỨ TỰ DÒNG với X. Nhãn cụm thường là 0..k-1.
  * KHÔNG giả định mô hình nào cũng có predict().
  * Cách lưu model: để nguyên đối tượng trong result["model"] (không pickle ở đây).
  * Cả 3 hàm run_* PHẢI trả kết quả qua make_clustering_result() bên dưới để đúng key.

QUY ĐỊNH CHUNG: xem đầu file models/data_processing.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import time
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from utils.constants import DEFAULT_PARAMS, NOISE_LABEL, RANDOM_STATE, SUPPORTS_PREDICT


# ---------------------------------------------------------------------------
# ĐÃ VIẾT SẴN - DÙNG, KHÔNG SỬA
# ---------------------------------------------------------------------------
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
    """Đóng gói kết quả theo đúng hợp đồng. Mọi run_* gọi hàm này ở dòng return.

    Args:
        algorithm (str): "kmeans" | "gmm" | "dbscan".
        model: đối tượng model đã fit.
        labels (np.ndarray): nhãn từng dòng, shape (n,).
        params (dict): tham số đã dùng.
        runtime_seconds (float | None): thời gian chạy (time.perf_counter).
        cluster_centers (np.ndarray | None): (k, d). K-Means: cluster_centers_; GMM: means_;
            DBSCAN: None.
        probabilities (np.ndarray | None): (n, k) xác suất thuộc cụm. Chỉ GMM, còn lại None.
        extra (dict | None): có thể chứa "inertia", "bic", "aic", "core_sample_indices".

    Returns:
        dict ClusteringResult (key cố định, xem CONTRACT_KEYS["run_clustering"]).
    """
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
        "feature_space": None,  # run_clustering() điền "scaled" hoặc "pca"
        "supports_predict": SUPPORTS_PREDICT[algorithm],
        "runtime_seconds": runtime_seconds,
        "extra": full_extra,
    }


def run_clustering(
    X: np.ndarray,
    algorithm: str,
    params: dict | None = None,
    feature_space: str = "scaled",
    random_state: int = RANDOM_STATE,
) -> dict:
    """Hàm điều phối: giao diện chỉ gọi hàm này. ĐÃ VIẾT SẴN, không cần sửa.

    Args:
        X (np.ndarray): (n, d) X_scaled hoặc X_pca.
        algorithm (str): "kmeans" | "gmm" | "dbscan".
        params (dict | None): ghi đè tham số mặc định (xem utils.constants.DEFAULT_PARAMS).
        feature_space (str): "scaled" | "pca" - X thuộc không gian nào (để dự đoán khách mới).
        random_state (int): mặc định 42.

    Returns:
        dict ClusteringResult (xem make_clustering_result).

    Ví dụ gọi:
        res = interface.run_clustering(X_scaled, "kmeans", {"n_clusters": 4}, feature_space="scaled")
    """
    if algorithm not in DEFAULT_PARAMS:
        raise ValueError(f"algorithm phải thuộc {list(DEFAULT_PARAMS)}, nhận '{algorithm}'")
    merged = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    if algorithm == "kmeans":
        result = run_kmeans(X, random_state=random_state, **merged)
    elif algorithm == "gmm":
        result = run_gmm(X, random_state=random_state, **merged)
    else:
        result = run_dbscan(X, **merged)
    result["feature_space"] = feature_space
    return result


# ---------------------------------------------------------------------------
# DUY LĨNH VIẾT THÂN HÀM (giữ nguyên chữ ký và cách return)
# ---------------------------------------------------------------------------
def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> dict:
    """K-Means.

    Args:
        X (np.ndarray): (n, d).
        n_clusters (int): số cụm K. Mặc định 5.
        n_init (int): số lần khởi tạo. Mặc định 10.
        max_iter (int): số vòng lặp tối đa. Mặc định 300.
        random_state (int): mặc định 42.

    Returns:
        dict ClusteringResult qua make_clustering_result(...):
            algorithm="kmeans", cluster_centers=model.cluster_centers_ (k, d),
            probabilities=None, extra={"inertia": model.inertia_}.

    Gợi ý thân hàm:
        t0 = time.perf_counter(); model = KMeans(...).fit(X); labels = model.labels_
        return make_clustering_result("kmeans", model, labels, {...}, time.perf_counter() - t0, ...)
    """
    t0 = time.perf_counter()
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(X)
    runtime = time.perf_counter() - t0

    params = {
        "n_clusters": n_clusters,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": random_state,
    }
    return make_clustering_result(
        "kmeans",
        model,
        model.labels_,
        params,
        runtime,
        cluster_centers=model.cluster_centers_,  # (k, d)
        probabilities=None,  # K-Means không có xác suất
        extra={"inertia": model.inertia_},
    )


def run_gmm(
    X: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "full",
    max_iter: int = 100,
    random_state: int = 42,
) -> dict:
    """Gaussian Mixture Model (xem make_clustering_result: cluster_centers=means_, có probabilities)."""
    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)
    runtime = time.perf_counter() - t0

    params = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "max_iter": max_iter,
        "random_state": random_state,
    }
    return make_clustering_result(
        "gmm",
        model,
        labels,
        params,
        runtime,
        cluster_centers=model.means_,  # (k, d)
        probabilities=probs,  # (n, k)
        extra={"bic": float(model.bic(X)), "aic": float(model.aic(X))},
    )


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict:
    """DBSCAN. Nhãn noise = -1; không có predict (supports_predict=False)."""
    t0 = time.perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    runtime = time.perf_counter() - t0

    params = {"eps": eps, "min_samples": min_samples}
    return make_clustering_result(
        "dbscan",
        model,
        model.labels_,  # noise = -1
        params,
        runtime,
        cluster_centers=None,
        probabilities=None,
        extra={"core_sample_indices": model.core_sample_indices_},
    )