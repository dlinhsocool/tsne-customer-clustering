"""
models/evaluation.py
====================
Người phụ trách: Duy Lĩnh (đánh giá & benchmark)

Gồm 3 nhóm:
  1. Clustering Evaluation: Silhouette, Davies-Bouldin, ARI, evaluate_clustering()
  2. Dimensionality Reduction Evaluation: Trustworthiness, Continuity, k-NN Preservation,
     Spearman, Stability, compare_dimensionality_reduction_method()
  3. Multi-space Benchmark: benchmark_clustering_spaces() (4 không gian x 3 thuật toán = 12 cấu hình)
"""
from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from models.clustering import estimate_dbscan_eps, run_clustering
from utils.constants import ALGORITHMS, DEFAULT_PARAMS, FEATURE_SPACES, NOISE_LABEL


# ---------------------------------------------------------------------------
# NHÓM 1: CLUSTERING EVALUATION
# ---------------------------------------------------------------------------
def get_valid_mask(labels: np.ndarray) -> np.ndarray:
    """Mask các dòng không phải noise."""
    return np.asarray(labels) != NOISE_LABEL


def is_computable(labels: np.ndarray) -> tuple[bool, str | None]:
    """Kiểm tra có tính được Silhouette / Davies-Bouldin không."""
    labels = np.asarray(labels)
    mask = get_valid_mask(labels)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return False, "Toàn bộ điểm là noise nên không tính được chỉ số."
    k = len(np.unique(labels[mask]))
    if k < 2:
        return False, f"Chỉ có {k} cụm (cần ít nhất 2) nên không tính được chỉ số."
    if n_valid <= k:
        return False, "Số điểm không đủ nhiều hơn số cụm."
    return True, None


def calculate_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 10000,
    random_state: int = 42,
) -> float | None:
    """Silhouette Score trung bình (loại noise)."""
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X, labels = np.asarray(X), np.asarray(labels)
    mask = get_valid_mask(labels)
    Xv, lv = X[mask], labels[mask]
    try:
        if sample_size is not None and len(lv) > sample_size:
            return float(
                silhouette_score(Xv, lv, sample_size=sample_size, random_state=random_state)
            )
        return float(silhouette_score(Xv, lv))
    except ValueError:
        return None


def calculate_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Davies-Bouldin Index (loại noise). Càng NHỎ càng tốt."""
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X, labels = np.asarray(X), np.asarray(labels)
    mask = get_valid_mask(labels)
    try:
        return float(davies_bouldin_score(X[mask], labels[mask]))
    except ValueError:
        return None


def calculate_ari(labels_true: np.ndarray | None, labels_pred: np.ndarray) -> float | None:
    """Adjusted Rand Index so với nhãn tham chiếu."""
    if labels_true is None:
        return None
    return float(adjusted_rand_score(np.asarray(labels_true), np.asarray(labels_pred)))


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    labels_true: np.ndarray | None = None,
    runtime_seconds: float | None = None,
) -> dict:
    """Gom các chỉ số của MỘT kết quả phân cụm."""
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    mask = get_valid_mask(labels)
    n_noise = int((~mask).sum())
    n_clusters = int(len(np.unique(labels[mask])))
    _, note = is_computable(labels)
    return {
        "silhouette_score": calculate_silhouette_score(X, labels),
        "davies_bouldin_score": calculate_davies_bouldin_score(X, labels),
        "ari_score": calculate_ari(labels_true, labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / n) if n > 0 else 0.0,
        "runtime_seconds": runtime_seconds,
        "note": note,
    }


# ---------------------------------------------------------------------------
# NHÓM 2: DIMENSIONALITY REDUCTION EVALUATION
# ---------------------------------------------------------------------------
def _validate_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(f"{name} phải có shape (n_samples >= 2, n_features >= 1).")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} chứa NaN hoặc Inf. Hãy làm sạch dữ liệu trước khi đánh giá.")
    return X


def calculate_dbcv_score(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float | None:
    """Tính Density-Based Clustering Validation (DBCV), giữ nguyên nhãn noise -1."""
    X = _validate_matrix(X)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError("labels phải là mảng 1 chiều và có cùng số dòng với X.")
    ok, _ = is_computable(labels)
    if not ok:
        return None
    try:
        from hdbscan.validity import validity_index
    except ImportError as exc:
        raise ImportError(
            "Chưa có thư viện hdbscan để tính DBCV. Hãy cài bằng `pip install hdbscan`."
        ) from exc
    score = float(validity_index(X, labels.astype(int), metric=metric))
    return score if np.isfinite(score) else None


def calculate_noise_ratio(labels: np.ndarray) -> float:
    """Tỷ lệ noise theo phần trăm (0–100)."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        return 0.0
    return float(np.mean(labels == NOISE_LABEL) * 100.0)


def calculate_n_clusters(labels: np.ndarray) -> int:
    """Đếm số cụm, không tính nhãn noise -1."""
    labels = np.asarray(labels)
    return int(len(np.unique(labels[labels != NOISE_LABEL])))


def _fit_tsne_embedding(
    X: np.ndarray,
    tsne_params: dict | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Chạy sklearn t-SNE; tương thích cả tham số n_iter và max_iter."""
    X = _validate_matrix(X, "X_input")
    params = {
        "n_components": 2,
        "perplexity": 30.0,
        "learning_rate": "auto",
        "init": "pca",
        "max_iter": 1000,
        "random_state": random_state,
    }
    params.update(tsne_params or {})
    params["random_state"] = random_state
    tsne_signature = inspect.signature(TSNE).parameters
    if "max_iter" in tsne_signature:
        if "n_iter" in params and "max_iter" not in (tsne_params or {}):
            params["max_iter"] = params.pop("n_iter")
        else:
            params.pop("n_iter", None)
    else:
        if "max_iter" in params and "n_iter" not in (tsne_params or {}):
            params["n_iter"] = params.pop("max_iter")
        else:
            params.pop("max_iter", None)
    params["perplexity"] = min(float(params["perplexity"]), float(len(X) - 1))
    if params["perplexity"] <= 0:
        raise ValueError("t-SNE cần perplexity lớn hơn 0 và nhỏ hơn số mẫu.")
    return np.asarray(TSNE(**params).fit_transform(X), dtype=np.float64)


def calculate_dbscan_eps(
    X: np.ndarray,
    min_samples: int = 5,
    quantile: float = 90.0,
) -> float:
    """Ước lượng eps khởi tạo từ phân vị của khoảng cách k-NN."""
    return estimate_dbscan_eps(X, min_samples=min_samples, quantile=quantile)


def _fit_cluster_labels(
    X: np.ndarray,
    algorithm: str,
    params: dict | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Fit một thuật toán clustering và trả về labels theo cùng thứ tự dòng."""
    params = dict(params or {})
    if algorithm == "kmeans":
        config = {"n_clusters": 5, "n_init": 10, "max_iter": 300}
        config.update(params)
        config["random_state"] = random_state
        model = KMeans(**config)
    elif algorithm == "gmm":
        config = {
            "n_components": 5,
            "covariance_type": "full",
            "n_init": 1,
            "max_iter": 100,
        }
        config.update(params)
        config["random_state"] = random_state
        model = GaussianMixture(**config)
    elif algorithm == "dbscan":
        config = {"eps": None, "min_samples": min(5, len(X)), "metric": "euclidean"}
        config.update(params)
        if config["eps"] is None:
            config["eps"] = calculate_dbscan_eps(X, min_samples=config["min_samples"])
        model = DBSCAN(**config)
    else:
        raise ValueError("algorithm phải là 'kmeans', 'gmm' hoặc 'dbscan'.")
    return np.asarray(model.fit_predict(X), dtype=int)


def compare_cluster_method(
    X_tsne: np.ndarray,
    X_input: np.ndarray | None = None,
    clustering_params: dict[str, dict] | None = None,
    tsne_params: dict | None = None,
    random_states: tuple[int, ...] | list[int] = (42, 43, 44, 45, 46),
    silhouette_sample_size: int | None = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """So sánh K-Means, GMM và DBSCAN trên cùng embedding t-SNE."""
    X_tsne = _validate_matrix(X_tsne, "X_tsne")
    if X_input is not None:
        X_input = _validate_matrix(X_input, "X_input")
        if len(X_input) != len(X_tsne):
            raise ValueError("X_input và X_tsne phải có cùng số dòng, cùng thứ tự khách hàng.")

    default_params = {
        "kmeans": {"n_clusters": 5, "n_init": 10, "max_iter": 300},
        "gmm": {"n_components": 5, "covariance_type": "full", "n_init": 1, "max_iter": 100},
        "dbscan": {"eps": None, "min_samples": 5, "metric": "euclidean"},
    }
    for name, config in (clustering_params or {}).items():
        if name not in default_params:
            raise ValueError(f"Thuật toán không hỗ trợ: {name}.")
        default_params[name].update(config)
    supplied = clustering_params or {}
    if "n_clusters" not in supplied.get("kmeans", {}):
        default_params["kmeans"]["n_clusters"] = min(5, len(X_tsne))
    if "n_components" not in supplied.get("gmm", {}):
        default_params["gmm"]["n_components"] = min(5, len(X_tsne))
    if "min_samples" not in supplied.get("dbscan", {}):
        default_params["dbscan"]["min_samples"] = min(5, len(X_tsne))

    columns = [
        "algorithm", "n_clusters", "n_noise", "noise_ratio_percent", "dbscan_eps",
        "dbcv_score", "silhouette_score", "stability_ari",
        "runtime_seconds", "note",
    ]
    seeds = list(dict.fromkeys(int(seed) for seed in random_states))
    stability_labels = {name: [] for name in ("kmeans", "gmm", "dbscan")}
    stability_note = None
    if X_input is not None and len(seeds) >= 2:
        try:
            for seed in seeds:
                embedding = _fit_tsne_embedding(X_input, tsne_params, seed)
                for name in stability_labels:
                    stability_labels[name].append(
                        _fit_cluster_labels(embedding, name, default_params[name], seed)
                    )
        except (ValueError, FloatingPointError) as exc:
            stability_note = f"Không tính được stability: {exc}"
            stability_labels = {name: [] for name in stability_labels}
    elif X_input is None:
        stability_note = "Chưa có X_input nên không chạy lại t-SNE để đo stability."
    else:
        stability_note = "Cần ít nhất hai random_state để đo stability."

    rows = []
    for algorithm in ("kmeans", "gmm", "dbscan"):
        start = time.perf_counter()
        labels = _fit_cluster_labels(
            X_tsne, algorithm, default_params[algorithm], random_state
        )
        runtime = time.perf_counter() - start
        notes = []
        try:
            dbcv = calculate_dbcv_score(X_tsne, labels)
        except ImportError as exc:
            dbcv = None
            notes.append(str(exc))
        except (ValueError, FloatingPointError) as exc:
            dbcv = None
            notes.append(f"Không tính được DBCV: {exc}")

        silhouette = calculate_silhouette_score(
            X_tsne, labels, sample_size=silhouette_sample_size,
            random_state=random_state,
        )
        if silhouette is None:
            _, reason = is_computable(labels)
            if reason:
                notes.append(f"Silhouette: {reason}")

        stability = None
        repeated_labels = stability_labels[algorithm]
        if len(repeated_labels) >= 2:
            stability_scores = [
                adjusted_rand_score(repeated_labels[i], repeated_labels[j])
                for i in range(len(repeated_labels))
                for j in range(i + 1, len(repeated_labels))
            ]
            stability = float(np.mean(stability_scores)) if stability_scores else None
        if stability_note:
            notes.append(stability_note)

        actual_eps = None
        if algorithm == "dbscan":
            actual_eps = (
                calculate_dbscan_eps(
                    X_tsne,
                    min_samples=default_params[algorithm].get("min_samples", 5),
                )
                if default_params[algorithm].get("eps") is None
                else float(default_params[algorithm]["eps"])
            )

        rows.append({
            "algorithm": algorithm,
            "n_clusters": calculate_n_clusters(labels),
            "n_noise": int(np.sum(labels == NOISE_LABEL)),
            "noise_ratio_percent": calculate_noise_ratio(labels),
            "dbscan_eps": actual_eps,
            "dbcv_score": dbcv,
            "silhouette_score": silhouette,
            "stability_ari": stability,
            "runtime_seconds": float(runtime),
            "note": " ".join(notes) if notes else None,
        })
    return pd.DataFrame(rows, columns=columns)


def _nearest_neighbor_indices(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Lấy k láng giềng gần nhất, loại chính điểm truy vấn khỏi kết quả."""
    X = _validate_matrix(X)
    if n_neighbors < 1 or n_neighbors >= len(X):
        raise ValueError("n_neighbors phải nằm trong [1, n_samples - 1].")
    model = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X)), metric="euclidean")
    model.fit(X)
    raw_indices = model.kneighbors(X, return_distance=False)
    result = np.empty((len(X), n_neighbors), dtype=int)
    for i, neighbors in enumerate(raw_indices):
        neighbors = neighbors[neighbors != i]
        if len(neighbors) < n_neighbors:
            raise ValueError("Không tìm đủ láng giềng; hãy kiểm tra dữ liệu trùng lặp.")
        result[i] = neighbors[:n_neighbors]
    return result


def calculate_trustworthiness(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Trustworthiness của embedding; phát hiện các láng giềng giả."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    if n_neighbors < 1 or n_neighbors >= len(X_original) / 2:
        raise ValueError("n_neighbors phải lớn hơn 0 và nhỏ hơn n_samples / 2.")
    return float(trustworthiness(X_original, X_reduced, n_neighbors=n_neighbors))


def calculate_continuity(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Continuity: đo láng giềng trong dữ liệu gốc bị tách xa ở embedding."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    n = len(X_original)
    if n_neighbors < 1 or n_neighbors >= n / 2:
        raise ValueError("n_neighbors phải lớn hơn 0 và nhỏ hơn n_samples / 2.")

    neighbors_original = _nearest_neighbor_indices(X_original, n_neighbors)
    neighbors_reduced = _nearest_neighbor_indices(X_reduced, n_neighbors)
    penalties = 0.0
    for i in range(n):
        reduced_set = set(neighbors_reduced[i].tolist())
        missing = [j for j in neighbors_original[i] if int(j) not in reduced_set]
        if not missing:
            continue
        distances = pairwise_distances(X_reduced[i:i + 1], X_reduced)[0]
        ranks = rankdata(distances, method="average")
        penalties += float(np.maximum(ranks[missing] - n_neighbors, 0.0).sum())
    denominator = n * n_neighbors * (2 * n - 3 * n_neighbors - 1)
    if denominator <= 0:
        return float("nan")
    return float(1.0 - (2.0 * penalties / denominator))


def calculate_knn_preservation(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Tỷ lệ bảo toàn k-NN bằng Jaccard trung bình giữa hai tập láng giềng."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    original_neighbors = _nearest_neighbor_indices(X_original, n_neighbors)
    reduced_neighbors = _nearest_neighbor_indices(X_reduced, n_neighbors)
    scores = []
    for original_row, reduced_row in zip(original_neighbors, reduced_neighbors):
        intersection = len(set(original_row.tolist()) & set(reduced_row.tolist()))
        union = 2 * n_neighbors - intersection
        scores.append(intersection / union if union else 1.0)
    return float(np.mean(scores))


def calculate_spearman_rank_correlation(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    sample_size: int | None = 2000,
    random_state: int = 42,
) -> float | None:
    """Spearman correlation giữa thứ hạng khoảng cách cặp điểm ở hai không gian."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    if sample_size is not None and len(X_original) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(X_original), size=sample_size, replace=False))
        X_original, X_reduced = X_original[indices], X_reduced[indices]
    if len(X_original) < 3:
        return None
    original_distances = pdist(X_original, metric="euclidean")
    reduced_distances = pdist(X_reduced, metric="euclidean")
    score = spearmanr(original_distances, reduced_distances).statistic
    return float(score) if np.isfinite(score) else None


def calculate_embedding_stability(
    embeddings: list[np.ndarray] | tuple[np.ndarray, ...],
    sample_size: int | None = 2000,
    random_state: int = 42,
) -> float | None:
    """Đo ổn định giữa các embedding bằng Spearman của khoảng cách từng cặp."""
    if len(embeddings) < 2:
        return None
    arrays = [_validate_matrix(item, "embedding") for item in embeddings]
    n = len(arrays[0])
    if any(len(item) != n for item in arrays):
        raise ValueError("Các embedding phải có cùng số dòng và thứ tự mẫu.")
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        arrays = [item[indices] for item in arrays]
    distance_vectors = [pdist(item, metric="euclidean") for item in arrays]
    scores = []
    for i in range(len(distance_vectors)):
        for j in range(i + 1, len(distance_vectors)):
            score = spearmanr(distance_vectors[i], distance_vectors[j]).statistic
            if np.isfinite(score):
                scores.append(float(score))
    return float(np.mean(scores)) if scores else None


def _fit_reduction_embedding(
    X: np.ndarray,
    method: str,
    n_components: int,
    random_state: int,
    method_params: dict | None = None,
) -> np.ndarray:
    """Fit một thuật toán giảm chiều bằng thư viện chuyên dụng."""
    params = dict(method_params or {})
    if method == "pca":
        config = {"n_components": n_components, "random_state": random_state}
        config.update(params)
        model = PCA(**config)
    elif method == "tsne":
        config = {
            "n_components": n_components,
            "perplexity": 30.0,
            "learning_rate": "auto",
            "init": "pca",
        }
        config.update(params)
        return _fit_tsne_embedding(X, config, random_state)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "Chưa có thư viện umap-learn. Hãy cài bằng `pip install umap-learn`."
            ) from exc
        config = {
            "n_components": n_components,
            "n_neighbors": min(15, len(X) - 1),
            "min_dist": 0.1,
            "metric": "euclidean",
        }
        config.update(params)
        config["random_state"] = random_state
        model = UMAP(**config)
    else:
        raise ValueError("method phải là 'pca', 'tsne' hoặc 'umap'.")
    return np.asarray(model.fit_transform(X), dtype=np.float64)


def compare_dimensionality_reduction_method(
    X_scaled: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 10,
    random_states: tuple[int, ...] | list[int] = (42, 43, 44, 45, 46),
    pca_params: dict | None = None,
    tsne_params: dict | None = None,
    umap_params: dict | None = None,
    sample_size: int | None = 2000,
    random_state: int = 42,
    compute_stability: bool = False,
    precomputed: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """So sánh Original, PCA, UMAP và t-SNE bằng metric local, global, engineering."""
    X_scaled = _validate_matrix(X_scaled, "X_scaled")
    if n_components < 1 or n_components > min(X_scaled.shape):
        raise ValueError("n_components phải từ 1 đến min(n_samples, n_features).")
    if sample_size is not None:
        if sample_size < 3:
            raise ValueError("sample_size phải từ 3 trở lên hoặc None.")
        if len(X_scaled) > sample_size:
            rng = np.random.default_rng(random_state)
            indices = np.sort(rng.choice(len(X_scaled), size=sample_size, replace=False))
            X_eval = X_scaled[indices]
        else:
            X_eval = X_scaled
    else:
        X_eval = X_scaled

    if n_neighbors < 1 or n_neighbors >= len(X_eval) / 2:
        raise ValueError(
            f"n_neighbors phải lớn hơn 0 và nhỏ hơn n_samples/2 (n_samples={len(X_eval)})."
        )

    seeds = list(dict.fromkeys(int(seed) for seed in random_states)) if compute_stability else [int(random_state)]
    if not seeds:
        seeds = [int(random_state)]

    method_configs = {
        "pca": pca_params,
        "umap": umap_params,
        "tsne": tsne_params,
    }
    columns = [
        "method", "n_components", "trustworthiness", "continuity", "knn_preservation",
        "spearman_rank_correlation", "runtime_seconds", "stability_score",
        "n_samples", "n_neighbors", "note",
    ]
    rows = [
        {
            "method": "original",
            "n_components": int(X_eval.shape[1]),
            "trustworthiness": None,
            "continuity": None,
            "knn_preservation": None,
            "spearman_rank_correlation": None,
            "runtime_seconds": 0.0,
            "stability_score": None,
            "n_samples": int(len(X_eval)),
            "n_neighbors": int(n_neighbors),
            "note": "Dữ liệu chuẩn hóa gốc (không gian tham chiếu).",
        }
    ]

    for method in ("pca", "umap", "tsne"):
        embeddings: list[np.ndarray] = []
        runtime: float | None = None
        note: str | None = None
        actual_dim = int(n_components)

        if precomputed and method in precomputed and precomputed[method] is not None:
            item = precomputed[method]
            emb0 = _validate_matrix(item["embedding"], f"{method}_embedding")
            if len(emb0) != len(X_eval):
                raise ValueError(f"precomputed['{method}'] không khớp số dòng X_eval.")
            embeddings.append(emb0)
            actual_dim = int(emb0.shape[1])
            runtime = float(item.get("runtime_seconds", 0.0))
            if compute_stability and len(seeds) > 1:
                for seed in seeds[1:]:
                    embeddings.append(
                        _fit_reduction_embedding(X_eval, method, actual_dim, seed, method_configs[method])
                    )
        else:
            try:
                for run_index, seed in enumerate(seeds):
                    start = time.perf_counter()
                    embedding = _fit_reduction_embedding(
                        X_eval, method, n_components, seed, method_configs[method]
                    )
                    elapsed = time.perf_counter() - start
                    if run_index == 0:
                        runtime = float(elapsed)
                        actual_dim = int(embedding.shape[1])
                    embeddings.append(embedding)
            except ImportError as exc:
                rows.append({
                    "method": method,
                    "n_components": None,
                    "trustworthiness": None,
                    "continuity": None,
                    "knn_preservation": None,
                    "spearman_rank_correlation": None,
                    "runtime_seconds": None,
                    "stability_score": None,
                    "n_samples": int(len(X_eval)),
                    "n_neighbors": int(n_neighbors),
                    "note": str(exc),
                })
                continue

        main_embedding = embeddings[0]
        stability = (
            calculate_embedding_stability(embeddings, sample_size=sample_size, random_state=random_state)
            if len(embeddings) >= 2
            else None
        )
        if not compute_stability:
            note = "Chưa bật chạy nhiều seed để đo Stability."

        rows.append({
            "method": method,
            "n_components": actual_dim,
            "trustworthiness": calculate_trustworthiness(X_eval, main_embedding, n_neighbors),
            "continuity": calculate_continuity(X_eval, main_embedding, n_neighbors),
            "knn_preservation": calculate_knn_preservation(X_eval, main_embedding, n_neighbors),
            "spearman_rank_correlation": calculate_spearman_rank_correlation(
                X_eval, main_embedding, sample_size=sample_size, random_state=random_state
            ),
            "runtime_seconds": runtime,
            "stability_score": stability,
            "n_samples": int(len(X_eval)),
            "n_neighbors": int(n_neighbors),
            "note": note,
        })
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# NHÓM 3: BENCHMARK CLUSTERING TRÊN NHIỀU KHÔNG GIAN
# ---------------------------------------------------------------------------
def benchmark_clustering_spaces(
    X_spaces: dict[str, dict[str, Any] | np.ndarray],
    clustering_params: dict[str, dict[str, Any]] | None = None,
    X_original: np.ndarray | None = None,
    labels_true: np.ndarray | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Chạy K-Means, GMM, DBSCAN trên từng không gian ('original', 'pca', 'umap', 'tsne').

    Args:
        X_spaces: dict dạng:
            {
                "original": {"X": X_scaled, "reduction_runtime": 0.0},
                "pca": {"X": X_pca, "reduction_runtime": 0.12},
                "umap": {"X": X_umap, "reduction_runtime": 1.45},
                "tsne": {"X": X_tsne, "reduction_runtime": 3.20},
            }
            (cũng chấp nhận truyền trực tiếp ndarray cho từng value, khi đó reduction_runtime = 0.0).
        clustering_params: tham số theo thuật toán hoặc theo (space, algo).
            Hỗ trợ dạng {"kmeans": {...}, "gmm": {...}, "dbscan": {...}}
            hoặc {"dbscan": {"original": {"eps": 1.5}, "pca": {"eps": 1.2}, ...}}.
        X_original: ma trận X_scaled dùng làm mặt bằng chung để tính
            silhouette_original và davies_bouldin_original. Nếu None sẽ lấy từ X_spaces["original"].
        labels_true: nhãn thật (n,) để tính ARI nếu có.
        random_state: seed cố định.

    Returns:
        dict:
            "table": pd.DataFrame (tối đa 12 dòng: 4 không gian x 3 thuật toán)
            "results": dict[str, dict[str, dict]] - {space: {algo: ClusteringResult}}
    """
    ordered_spaces = [s for s in ("original", "pca", "umap", "tsne") if s in X_spaces and X_spaces[s] is not None]
    if not ordered_spaces:
        raise ValueError("X_spaces phải chứa ít nhất một không gian dữ liệu hợp lệ.")

    parsed_spaces: dict[str, tuple[np.ndarray, float]] = {}
    for space in ordered_spaces:
        item = X_spaces[space]
        if isinstance(item, dict):
            mat = _validate_matrix(item["X"], f"X_spaces['{space}']['X']")
            red_rt = float(item.get("reduction_runtime") or 0.0)
        else:
            mat = _validate_matrix(item, f"X_spaces['{space}']")
            red_rt = 0.0
        parsed_spaces[space] = (mat, red_rt)

    if X_original is not None:
        X_orig_arr = _validate_matrix(X_original, "X_original")
    elif "original" in parsed_spaces:
        X_orig_arr = parsed_spaces["original"][0]
    else:
        X_orig_arr = None

    columns = [
        "space", "algorithm", "n_samples", "n_features",
        "n_clusters", "n_noise", "noise_ratio",
        "reduction_runtime", "clustering_runtime", "total_runtime",
        "silhouette_space", "davies_bouldin_space",
        "silhouette_original", "davies_bouldin_original",
        "ari_score", "note",
    ]
    rows: list[dict[str, Any]] = []
    all_results: dict[str, dict[str, dict]] = {}

    for space in ordered_spaces:
        X_mat, red_rt = parsed_spaces[space]
        n_samples, n_features = X_mat.shape
        all_results[space] = {}

        for algo in ("kmeans", "gmm", "dbscan"):
            raw_cfg = (clustering_params or {}).get(algo, {})
            if space in raw_cfg and isinstance(raw_cfg[space], dict):
                algo_cfg = {**DEFAULT_PARAMS[algo], **raw_cfg[space]}
            else:
                algo_cfg = {**DEFAULT_PARAMS[algo], **raw_cfg}

            # DBSCAN bắt buộc thích ứng eps theo không gian nếu người dùng không ép riêng
            if algo == "dbscan":
                space_specific = isinstance(raw_cfg.get(space), dict) and "eps" in raw_cfg[space]
                if not space_specific and raw_cfg.get("eps") is None:
                    algo_cfg["eps"] = estimate_dbscan_eps(
                        X_mat, min_samples=int(algo_cfg.get("min_samples", 5))
                    )

            res = run_clustering(
                X_mat,
                algo,
                params=algo_cfg,
                feature_space=space,
                random_state=random_state,
            )
            all_results[space][algo] = res

            labels = res["labels"]
            clust_rt = float(res["runtime_seconds"] or 0.0)
            total_rt = float(red_rt + clust_rt)

            eval_space = evaluate_clustering(X_mat, labels, labels_true=labels_true, runtime_seconds=clust_rt)
            if X_orig_arr is not None and len(X_orig_arr) == n_samples:
                sil_orig = calculate_silhouette_score(X_orig_arr, labels, random_state=random_state)
                db_orig = calculate_davies_bouldin_score(X_orig_arr, labels)
            else:
                sil_orig = None
                db_orig = None

            note_parts = []
            if eval_space["note"]:
                note_parts.append(eval_space["note"])
            if algo == "dbscan":
                note_parts.append(f"eps={res['params']['eps']:.3f}")

            rows.append({
                "space": space,
                "algorithm": algo,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_clusters": int(eval_space["n_clusters"]),
                "n_noise": int(eval_space["n_noise"]),
                "noise_ratio": float(eval_space["noise_ratio"]),
                "reduction_runtime": float(red_rt),
                "clustering_runtime": float(clust_rt),
                "total_runtime": float(total_rt),
                "silhouette_space": eval_space["silhouette_score"],
                "davies_bouldin_space": eval_space["davies_bouldin_score"],
                "silhouette_original": sil_orig,
                "davies_bouldin_original": db_orig,
                "ari_score": eval_space["ari_score"],
                "note": "; ".join(note_parts) if note_parts else None,
            })

    return {
        "table": pd.DataFrame(rows, columns=columns),
        "results": all_results,
    }


def compare_models(
    results: dict[str, dict],
    X_scaled: np.ndarray,
    X_pca: np.ndarray | None = None,
    labels_true: np.ndarray | None = None,
    X_umap: np.ndarray | None = None,
    X_tsne: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bảng so sánh các thuật toán trên không gian tương ứng của từng kết quả."""
    columns = [
        "algorithm", "n_clusters", "n_noise", "silhouette_score",
        "davies_bouldin_score", "ari_score", "runtime_seconds",
    ]
    space_map = {
        "original": X_scaled,
        "scaled": X_scaled,
        "pca": X_pca,
        "umap": X_umap,
        "tsne": X_tsne,
    }
    rows = []
    for algo, res in results.items():
        space = res.get("feature_space") or "original"
        X = space_map.get(space)
        if X is None:
            raise ValueError(f"'{algo}' chạy trên '{space}' nhưng chưa truyền ma trận của không gian này.")
        m = evaluate_clustering(X, res["labels"], labels_true, res.get("runtime_seconds"))
        rows.append({"algorithm": algo, **{c: m[c] for c in columns[1:]}})
    return pd.DataFrame(rows, columns=columns)