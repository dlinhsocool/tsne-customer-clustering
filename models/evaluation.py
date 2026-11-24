"""
models/evaluation.py
====================
Người phụ trách: Duy Lĩnh (đánh giá)

NHIỆM VỤ FILE: Silhouette, Davies-Bouldin, ARI và bảng so sánh các thuật toán.

QUY TẮC "KHÔNG TÍNH CHỈ SỐ KHÔNG HỢP LỆ":
  * Điểm noise (-1) bị LOẠI khỏi tính Silhouette và Davies-Bouldin.
  * Cần ít nhất 2 cụm (không tính noise) và nhiều điểm hơn số cụm. Không đủ => trả None.
  * ARI trả None nếu labels_true là None. ARI tính trên TẤT CẢ dòng (noise coi như một nhóm riêng).
  * Dùng hàm is_computable() bên dưới để quyết định, đừng tự viết lại điều kiện.

QUY ĐỊNH CHUNG: xem đầu file models/data_processing.py.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from utils.constants import NOISE_LABEL

# ---------------------------------------------------------------------------
# ĐÃ VIẾT SẴN - DÙNG, KHÔNG SỬA
# ---------------------------------------------------------------------------
def get_valid_mask(labels: np.ndarray) -> np.ndarray:
    """Mask các dòng không phải noise. Dùng để loại noise trước khi tính chỉ số."""
    return np.asarray(labels) != NOISE_LABEL


def is_computable(labels: np.ndarray) -> tuple[bool, str | None]:
    """Có tính được Silhouette / Davies-Bouldin không?

    Returns:
        (True, None) nếu tính được; (False, "lý do tiếng Việt") nếu không.
    """
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


# ---------------------------------------------------------------------------
# DUY LĨNH VIẾT THÂN HÀM
# ---------------------------------------------------------------------------
def calculate_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 10000,
    random_state: int = 42,
) -> float | None:
    """Silhouette Score trung bình (loại noise).

    Args:
        X (np.ndarray): (n, d) CÙNG không gian mà thuật toán đã chạy (X_scaled hoặc X_pca).
        labels (np.ndarray): (n,) nhãn cụm, noise = -1.
        sample_size (int | None): tính trên mẫu cho nhanh khi n lớn. None = dùng hết. Mặc định 10000.
        random_state (int): seed lấy mẫu. Mặc định 42.

    Returns:
        float trong [-1, 1], hoặc None nếu is_computable(labels) báo không tính được.
    """
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X, labels = np.asarray(X), np.asarray(labels)
    mask = get_valid_mask(labels)
    Xv, lv = X[mask], labels[mask]
    try:
        if sample_size is not None and len(lv) > sample_size:
            return float(silhouette_score(Xv, lv, sample_size=sample_size,
                                          random_state=random_state))
        return float(silhouette_score(Xv, lv))
    except ValueError:  # mẫu con chỉ rơi vào 1 cụm
        return None


def calculate_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Davies-Bouldin Index (loại noise). Càng NHỎ càng tốt.

    Returns:
        float >= 0, hoặc None nếu không tính được.
    """
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X, labels = np.asarray(X), np.asarray(labels)
    mask = get_valid_mask(labels)
    return float(davies_bouldin_score(X[mask], labels[mask]))


def calculate_ari(labels_true: np.ndarray | None, labels_pred: np.ndarray) -> float | None:
    """Adjusted Rand Index so với nhãn tham chiếu.

    Args:
        labels_true (np.ndarray | None): nhãn thật (n,) hoặc None nếu không có.
        labels_pred (np.ndarray): nhãn do thuật toán tìm ra (n,).

    Returns:
        float (1 = khớp hoàn toàn, ~0 = ngẫu nhiên), hoặc None nếu labels_true là None.
    """
    if labels_true is None:
        return None
    return float(adjusted_rand_score(np.asarray(labels_true), np.asarray(labels_pred)))


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    labels_true: np.ndarray | None = None,
    runtime_seconds: float | None = None,
) -> dict:
    """Gom các chỉ số của MỘT kết quả phân cụm.

    Returns:
        dict (đủ key, không đổi tên):
            "silhouette_score": float | None
            "davies_bouldin_score": float | None
            "ari_score": float | None
            "n_clusters": int              - không tính noise
            "n_noise": int
            "noise_ratio": float           - n_noise / n, trong [0, 1]
            "runtime_seconds": float | None
            "note": str | None             - lý do nếu có chỉ số None (lấy từ is_computable)

    Ví dụ gọi:
        m = interface.evaluate_clustering(X_scaled, res["labels"], labels_true=None, runtime_seconds=res["runtime_seconds"])
    """
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
# ĐÁNH GIÁ CLUSTERING TRÊN KHÔNG GIAN t-SNE (THỰC NGHIỆM RIÊNG)
# ---------------------------------------------------------------------------
def _validate_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Kiểm tra ma trận số 2 chiều dùng trong các phép đánh giá mới."""
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
    """Tính Density-Based Clustering Validation (DBCV), giữ nguyên nhãn noise -1.

    DBCV được triển khai bởi hdbscan.validity.validity_index. Không lọc noise
    trước khi gọi thư viện vì thuật toán DBCV tự xử lý nhãn noise.

    Returns:
        float trong [-1, 1], hoặc None nếu phân cụm không đủ điều kiện.

    Raises:
        ImportError: nếu chưa cài ``hdbscan``.
    """
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
    score = validity_index(X, labels.astype(int), metric=metric)
    score = float(score)
    return score if np.isfinite(score) else None


def calculate_noise_ratio(labels: np.ndarray) -> float:
    """Tỷ lệ noise theo phần trăm (0–100); thuật toán không có noise trả về 0."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels phải là mảng 1 chiều.")
    if len(labels) == 0:
        return 0.0
    return float(np.mean(labels == NOISE_LABEL) * 100.0)


def calculate_n_clusters(labels: np.ndarray) -> int:
    """Đếm số cụm, không tính nhãn noise -1."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels phải là mảng 1 chiều.")
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
    # sklearn đổi tên n_iter thành max_iter ở các phiên bản mới.
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
    """Ước lượng eps khởi tạo từ phân vị của khoảng cách k-NN.

    Đây là giá trị khởi tạo có thể thay thế bằng eps do người dùng tinh chỉnh.
    Dùng phân vị cao của khoảng cách tới láng giềng thứ min_samples để tránh
    dùng eps=0.5 cố định vốn không phù hợp với thang tọa độ t-SNE.
    """
    X = _validate_matrix(X)
    if min_samples < 1 or min_samples > len(X):
        raise ValueError("min_samples phải nằm trong [1, n_samples].")
    if not 0 < quantile <= 100:
        raise ValueError("quantile phải thuộc (0, 100].")
    # DBSCAN tính cả chính điểm đó vào min_samples, còn kneighbors() không
    # trả chính điểm truy vấn; vì vậy cần min_samples - 1 láng giềng khác.
    n_neighbors = max(1, min_samples - 1)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    distances = model.kneighbors(return_distance=True)[0][:, -1]
    eps = float(np.percentile(distances, quantile))
    if not np.isfinite(eps) or eps <= 0:
        positive = distances[distances > 0]
        eps = float(np.median(positive)) if len(positive) else 1e-6
    return eps


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
        # eps=None: ước lượng theo phân vị khoảng cách k-NN của embedding hiện tại.
        if config["eps"] is None:
            config["eps"] = calculate_dbscan_eps(X, min_samples=config["min_samples"])
        # DBSCAN không có random_state.
        model = DBSCAN(**config)
    else:
        raise ValueError("algorithm phải là 'kmeans', 'gmm' hoặc 'dbscan'.")
    return np.asarray(model.fit_predict(X), dtype=int)


def calculate_clustering_stability(
    X_input: np.ndarray,
    algorithm: str,
    clustering_params: dict | None = None,
    tsne_params: dict | None = None,
    random_states: tuple[int, ...] | list[int] = (42, 43, 44, 45, 46),
) -> float | None:
    """Đo stability bằng ARI trung bình giữa các lần t-SNE + clustering.

    Mỗi random_state tạo một embedding t-SNE mới và chạy lại cùng thuật toán.
    ARI được tính trên toàn bộ dòng, trong đó noise -1 của DBSCAN vẫn được giữ
    như một nhãn nhóm riêng. Cần ít nhất hai random_state.
    """
    X_input = _validate_matrix(X_input, "X_input")
    seeds = list(dict.fromkeys(int(seed) for seed in random_states))
    if len(seeds) < 2:
        return None
    all_labels = []
    for seed in seeds:
        embedding = _fit_tsne_embedding(X_input, tsne_params, seed)
        labels = _fit_cluster_labels(embedding, algorithm, clustering_params, seed)
        all_labels.append(labels)
    scores = [
        adjusted_rand_score(all_labels[i], all_labels[j])
        for i in range(len(all_labels))
        for j in range(i + 1, len(all_labels))
    ]
    return float(np.mean(scores)) if scores else None


def compare_cluster_method(
    X_tsne: np.ndarray,
    X_input: np.ndarray | None = None,
    clustering_params: dict[str, dict] | None = None,
    tsne_params: dict | None = None,
    random_states: tuple[int, ...] | list[int] = (42, 43, 44, 45, 46),
    silhouette_sample_size: int | None = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """So sánh K-Means, GMM và DBSCAN trên cùng embedding t-SNE.

    Args:
        X_tsne: Tọa độ t-SNE (n, 2) hoặc (n, 3) dùng để fit ba thuật toán.
        X_input: Dữ liệu trước t-SNE. Truyền đối số này để tính clustering
            stability bằng cách chạy lại t-SNE với nhiều random_state.
        clustering_params: Tham số riêng cho từng thuật toán, ví dụ:
            {"kmeans": {"n_clusters": 5}, "gmm": {"n_components": 5},
             "dbscan": {"eps": 1.2, "min_samples": 5}}. Nếu bỏ qua eps,
            chương trình ước lượng eps từ phân vị 90% khoảng cách k-NN.
        tsne_params: Tham số t-SNE dùng khi tính stability. Cần trùng với cấu
            hình t-SNE đã dùng để tạo X_tsne.
        random_states: Các seed dùng cho clustering stability; mặc định 5 lần.
        silhouette_sample_size: Số mẫu tối đa dùng cho Silhouette.
        random_state: Seed lấy mẫu Silhouette.

    Returns:
        DataFrame gồm một dòng cho mỗi thuật toán và các chỉ số DBCV,
        Silhouette (đã loại noise), stability ARI, số cụm, noise ratio (%) và runtime.

    Lưu ý:
        X_input là bắt buộc nếu cần cột stability_ari có giá trị. DBCV cần
        ``pip install hdbscan``; nếu chưa cài, cột DBCV là None và note nêu rõ.
    """
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
    # Tạo các embedding t-SNE chỉ một lần cho cả ba thuật toán để tránh
    # chạy lặp lại 3 x số_seed lần không cần thiết.
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
    # Luôn báo cáo đủ ba thuật toán để việc so sánh không bị thiếu dòng.
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
            actual_eps = calculate_dbscan_eps(
                X_tsne,
                min_samples=default_params[algorithm].get("min_samples", 5),
            ) if default_params[algorithm].get("eps") is None else float(default_params[algorithm]["eps"])

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


# ---------------------------------------------------------------------------
# CÁC HÀM ĐÁNH GIÁ CHẤT LƯỢNG GIẢM CHIỀU
# ---------------------------------------------------------------------------
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
    """Continuity: đo láng giềng trong dữ liệu gốc bị tách xa ở embedding.

    Tính theo công thức rank-penalty đối ngẫu với Trustworthiness. Mỗi hàng
    được xử lý riêng để tránh tạo một ma trận khoảng cách n x n trong RAM.
    """
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
    """Đo ổn định giữa các embedding bằng Spearman của khoảng cách từng cặp.

    So sánh khoảng cách cặp điểm thay vì tọa độ trực tiếp để không bị ảnh hưởng
    bởi phép quay, tịnh tiến hoặc phản chiếu embedding.
    """
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
        config = {"n_components": n_components, "svd_solver": "full"}
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
            from umap.umap_ import UMAP
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
) -> pd.DataFrame:
    """So sánh PCA, t-SNE và UMAP bằng metric local, global, engineering.

    Args:
        X_scaled: Ma trận dữ liệu trước giảm chiều (n, d).
        n_components: Số chiều đích, mặc định 2 để so sánh công bằng.
        n_neighbors: k dùng cho Trustworthiness, Continuity và k-NN Preservation.
        random_states: 5–10 seed dùng để đo Stability; mặc định 5 seed.
        pca_params, tsne_params, umap_params: Tham số bổ sung cho từng framework.
        sample_size: Số dòng tối đa dùng cho toàn bộ phép so sánh. None = dùng hết.
            Mặc định 2000 để khống chế bộ nhớ của ma trận khoảng cách cặp điểm.
        random_state: Seed lấy mẫu cố định.

    Returns:
        DataFrame một dòng cho mỗi phương pháp, gồm Trustworthiness,
        Continuity, k-NN Preservation, Spearman, runtime và stability.
        Nếu chưa cài umap-learn, dòng UMAP vẫn được giữ và note ghi rõ lý do.
    """
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

    seeds = list(dict.fromkeys(int(seed) for seed in random_states))
    if not seeds:
        seeds = [int(random_state)]
    method_configs = {
        "pca": pca_params,
        "tsne": tsne_params,
        "umap": umap_params,
    }
    columns = [
        "method", "trustworthiness", "continuity", "knn_preservation",
        "spearman_rank_correlation", "runtime_seconds", "stability_score",
        "n_samples", "n_neighbors", "note",
    ]
    rows = []
    for method in ("pca", "tsne", "umap"):
        embeddings = []
        runtime = None
        note = None
        try:
            for run_index, seed in enumerate(seeds):
                start = time.perf_counter()
                embedding = _fit_reduction_embedding(
                    X_eval, method, n_components, seed, method_configs[method]
                )
                elapsed = time.perf_counter() - start
                if run_index == 0:
                    runtime = float(elapsed)
                embeddings.append(embedding)
        except ImportError as exc:
            rows.append({
                "method": method,
                "trustworthiness": None,
                "continuity": None,
                "knn_preservation": None,
                "spearman_rank_correlation": None,
                "runtime_seconds": None,
                "stability_score": None,
                "n_samples": len(X_eval),
                "n_neighbors": n_neighbors,
                "note": str(exc),
            })
            continue

        main_embedding = embeddings[0]
        rows.append({
            "method": method,
            "trustworthiness": calculate_trustworthiness(
                X_eval, main_embedding, n_neighbors
            ),
            "continuity": calculate_continuity(
                X_eval, main_embedding, n_neighbors
            ),
            "knn_preservation": calculate_knn_preservation(
                X_eval, main_embedding, n_neighbors
            ),
            "spearman_rank_correlation": calculate_spearman_rank_correlation(
                X_eval, main_embedding, sample_size=sample_size,
                random_state=random_state,
            ),
            "runtime_seconds": runtime,
            "stability_score": calculate_embedding_stability(
                embeddings, sample_size=sample_size, random_state=random_state
            ),
            "n_samples": len(X_eval),
            "n_neighbors": n_neighbors,
            "note": note,
        })
    return pd.DataFrame(rows, columns=columns)


def compare_models(
    results: dict[str, dict],
    X_scaled: np.ndarray,
    X_pca: np.ndarray | None = None,
    labels_true: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bảng so sánh nhiều thuật toán.

    Args:
        results (dict): {"kmeans": ClusteringResult, "gmm": ..., "dbscan": ...} (có thể thiếu).
        X_scaled (np.ndarray): (n, d).
        X_pca (np.ndarray | None): (n, k). Với mỗi kết quả, chọn X theo result["feature_space"]
            ("scaled" -> X_scaled, "pca" -> X_pca).
        labels_true (np.ndarray | None): nhãn tham chiếu cho ARI.

    Returns:
        pd.DataFrame, MỖI THUẬT TOÁN MỘT DÒNG, đúng các cột:
            "algorithm" (str: "kmeans"|"gmm"|"dbscan"), "n_clusters", "n_noise",
            "silhouette_score", "davies_bouldin_score", "ari_score", "runtime_seconds"
        Giá trị không tính được là None/NaN (không xóa cột, không xóa dòng).
    """
    columns = ["algorithm", "n_clusters", "n_noise", "silhouette_score",
               "davies_bouldin_score", "ari_score", "runtime_seconds"]
    rows = []
    for algo, res in results.items():
        if res.get("feature_space") == "pca":
            if X_pca is None:
                raise ValueError(f"'{algo}' chạy trên PCA nhưng X_pca = None")
            X = X_pca
        else:
            X = X_scaled
        m = evaluate_clustering(X, res["labels"], labels_true, res.get("runtime_seconds"))
        rows.append({"algorithm": algo, **{c: m[c] for c in columns[1:]}})
    return pd.DataFrame(rows, columns=columns)  # results rỗng vẫn giữ đủ cột
