"""
models/evaluation_gpu.py
========================
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
import warnings
from typing import Any

import numpy as np
import pandas as pd
import models.evaluation_cpu as _cpu
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



# GPU helpers. Metrics without a trustworthy CUDA implementation deliberately
# retain their CPU implementation (DBCV, GMM, Spearman and multi-space pipeline).
try:
    import torch
except Exception:  # pragma: no cover - optional GPU dependency
    torch = None


def _cuda_device():
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("evaluation_gpu được chọn nhưng CUDA không khả dụng.")
    return torch.device("cuda")


def _torch_matrix(X, device=None):
    device = device or _cuda_device()
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)


def _pairwise_rank_matrix(X, device):
    xt = _torch_matrix(X, device)
    distances = torch.cdist(xt, xt, p=2)
    distances.fill_diagonal_(float("inf"))
    order = torch.argsort(distances, dim=1)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, len(X) + 1, device=device, dtype=order.dtype).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return distances, order, ranks


def _gpu_kmeans(X, config, random_state):
    device = _cuda_device()
    data = _torch_matrix(X, device)
    n_samples = data.shape[0]
    n_clusters = int(config.get("n_clusters", 5))
    if not 1 <= n_clusters <= n_samples:
        raise ValueError("n_clusters phải nằm trong [1, n_samples].")
    n_init = max(1, int(config.get("n_init", 10)))
    max_iter = max(1, int(config.get("max_iter", 300)))
    tol = float(config.get("tol", 1e-4))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_state))
    best_labels, best_inertia = None, float("inf")
    for init_idx in range(n_init):
        # k-means++ initialization; all distance work remains on CUDA.
        first = torch.randint(n_samples, (1,), generator=generator, device=device).item()
        centers = [data[first]]
        closest = torch.sum((data - centers[0]) ** 2, dim=1)
        for _ in range(1, n_clusters):
            total = closest.sum()
            if not torch.isfinite(total) or total <= 0:
                candidate = torch.randint(n_samples, (1,), generator=generator, device=device).item()
            else:
                probs = closest / total
                candidate = torch.multinomial(probs, 1, generator=generator).item()
            centers.append(data[candidate])
            closest = torch.minimum(closest, torch.sum((data - data[candidate]) ** 2, dim=1))
        centroids = torch.stack(centers).clone()
        for _ in range(max_iter):
            distances = torch.cdist(data, centroids, p=2)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.bincount(labels, minlength=n_clusters).to(data.dtype)
            new_centroids.index_add_(0, labels, data)
            nonempty = counts > 0
            new_centroids[nonempty] /= counts[nonempty, None]
            if (~nonempty).any():
                nearest = distances.min(dim=1).values
                farthest = torch.argsort(nearest, descending=True)
                empty_ids = torch.where(~nonempty)[0]
                for j, cid in enumerate(empty_ids):
                    new_centroids[cid] = data[farthest[j % len(farthest)]]
            shift = torch.linalg.vector_norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= tol:
                break
        final_dist = torch.cdist(data, centroids, p=2)
        labels = final_dist.argmin(dim=1)
        inertia = torch.sum((data - centroids[labels]) ** 2).item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.detach().cpu().numpy().astype(int)
        generator.manual_seed(int(random_state) + init_idx + 1)
    return best_labels


def _gpu_silhouette(X, labels, sample_size, random_state):
    device = _cuda_device()
    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = get_valid_mask(labels)
    Xv, lv = X[mask], labels[mask]
    if sample_size is not None and len(lv) > int(sample_size):
        rng = np.random.default_rng(random_state)
        ids = np.sort(rng.choice(len(lv), size=int(sample_size), replace=False))
        Xv, lv = Xv[ids], lv[ids]
    unique, inverse = np.unique(lv, return_inverse=True)
    if len(unique) < 2 or len(lv) <= len(unique):
        return None
    data = _torch_matrix(Xv, device)
    labels_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
    n = len(Xv)
    cluster_masks = [labels_t == i for i in range(len(unique))]
    cluster_counts = torch.tensor([int(m.sum()) for m in cluster_masks], dtype=data.dtype, device=device)
    scores = []
    # Chunk rows to bound VRAM; this computes the exact sampled silhouette.
    chunk = max(32, min(512, int(2_000_000 / max(n, 1))))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        dist = torch.cdist(data[start:stop], data, p=2)
        rows = torch.arange(start, stop, device=device)
        dist[torch.arange(stop-start, device=device), rows] = 0.0
        means = []
        for cid, cmask in enumerate(cluster_masks):
            sums = dist[:, cmask].sum(dim=1)
            denom = cluster_counts[cid] - (labels_t[start:stop] == cid).to(data.dtype)
            mean = sums / torch.clamp(denom, min=1.0)
            means.append(mean)
        means_t = torch.stack(means, dim=1)
        own = means_t.gather(1, labels_t[start:stop, None]).squeeze(1)
        own_count = cluster_counts[labels_t[start:stop]]
        alternatives = means_t.clone()
        alternatives.scatter_(1, labels_t[start:stop, None], float("inf"))
        nearest_other = alternatives.min(dim=1).values
        denom = torch.maximum(own, nearest_other)
        row_scores = torch.where((own_count > 1) & (denom > 0), (nearest_other-own)/denom, 0.0)
        scores.append(row_scores)
    return float(torch.cat(scores).mean().item())


def _gpu_davies_bouldin(X, labels):
    device = _cuda_device()
    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = get_valid_mask(labels)
    Xv, lv = X[mask], labels[mask]
    unique, inverse = np.unique(lv, return_inverse=True)
    if len(unique) < 2 or len(Xv) <= len(unique):
        return None
    data = _torch_matrix(Xv, device)
    labels_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
    k = len(unique)
    counts = torch.bincount(labels_t, minlength=k).to(data.dtype)
    centroids = torch.zeros((k, data.shape[1]), dtype=data.dtype, device=device)
    centroids.index_add_(0, labels_t, data)
    centroids /= counts[:, None]
    scatter = torch.zeros(k, dtype=data.dtype, device=device)
    point_dist = torch.linalg.vector_norm(data - centroids[labels_t], dim=1)
    scatter.index_add_(0, labels_t, point_dist)
    scatter /= counts
    centroid_dist = torch.cdist(centroids, centroids, p=2)
    centroid_dist.fill_diagonal_(float("inf"))
    ratios = (scatter[:, None] + scatter[None, :]) / torch.clamp(centroid_dist, min=1e-12)
    ratios.fill_diagonal_(float("-inf"))
    return float(ratios.max(dim=1).values.mean().item())


def _gpu_ari(labels_true, labels_pred):
    device = _cuda_device()
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("Hai mảng nhãn phải 1 chiều và cùng độ dài.")
    if len(a) < 2:
        return float(adjusted_rand_score(a, b))
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n_a, n_b = int(ai.max()) + 1, int(bi.max()) + 1
    ai_t = torch.as_tensor(ai, dtype=torch.long, device=device)
    bi_t = torch.as_tensor(bi, dtype=torch.long, device=device)
    contingency = torch.bincount(ai_t * n_b + bi_t, minlength=n_a*n_b).reshape(n_a, n_b).to(torch.float64)
    comb2 = lambda v: v * (v - 1.0) / 2.0
    sum_cells = comb2(contingency).sum()
    sum_rows = comb2(contingency.sum(dim=1)).sum()
    sum_cols = comb2(contingency.sum(dim=0)).sum()
    total = comb2(torch.tensor(float(len(a)), device=device, dtype=torch.float64))
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    denom = maximum - expected
    if torch.abs(denom) < 1e-15:
        return 1.0
    return float(((sum_cells - expected) / denom).item())


def _gpu_knn_indices(X, n_neighbors):
    device = _cuda_device()
    data = _torch_matrix(X, device)
    distances = torch.cdist(data, data, p=2)
    distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=int(n_neighbors), largest=False, dim=1).indices.detach().cpu().numpy()


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
    """Silhouette trên CUDA khi có thể; fallback sang implementation sklearn."""
    ok, _ = is_computable(labels)
    if not ok:
        return None
    try:
        return _gpu_silhouette(X, labels, sample_size, random_state)
    except Exception as exc:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(silhouette_score(
            np.asarray(X)[get_valid_mask(labels)],
            np.asarray(labels)[get_valid_mask(labels)],
            sample_size=sample_size if sample_size and int(get_valid_mask(labels).sum()) > sample_size else None,
            random_state=random_state,
        ))



def calculate_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Davies-Bouldin trên CUDA khi có thể; fallback sklearn nếu GPU lỗi."""
    ok, _ = is_computable(labels)
    if not ok:
        return None
    mask = get_valid_mask(labels)
    try:
        return _gpu_davies_bouldin(X, labels)
    except Exception:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(davies_bouldin_score(np.asarray(X)[mask], np.asarray(labels)[mask]))



def calculate_ari(labels_true: np.ndarray | None, labels_pred: np.ndarray) -> float | None:
    """ARI trên CUDA khi có thể; giữ sklearn làm fallback."""
    if labels_true is None:
        return None
    try:
        return _gpu_ari(labels_true, labels_pred)
    except Exception:
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


_CUMl_TSNE_FALLBACK_WARNED = False


def _as_numpy_embedding(embedding: Any) -> np.ndarray:
    """Convert NumPy/CuPy/cuDF outputs returned by framework estimators to NumPy."""
    if hasattr(embedding, "to_numpy"):
        embedding = embedding.to_numpy()
    elif hasattr(embedding, "get"):
        embedding = embedding.get()
    result = np.asarray(embedding, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("t-SNE framework trả về embedding không hợp lệ.")
    return result


def _fit_tsne_embedding(
    X: np.ndarray,
    tsne_params: dict | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Run framework t-SNE for clustering-stability experiments.

    Backend order:
      1. RAPIDS cuML TSNE on CUDA, when CUDA and cuML are available.
      2. scikit-learn TSNE on CPU otherwise (or when cuML rejects the request).

    Deliberately does NOT use the hand-written CustomTSNE from
    dimensionality_reduction_gpu.py. Stability must use a framework estimator
    so its implementation is independent from the project's custom t-SNE.
    """
    global _CUMl_TSNE_FALLBACK_WARNED

    X = _validate_matrix(X, "X_input")
    params = dict(tsne_params or {})
    n_components = int(params.pop("n_components", 2))
    perplexity = min(float(params.pop("perplexity", 30.0)), len(X) - 1.0)
    if perplexity <= 0:
        raise ValueError("t-SNE cần perplexity lớn hơn 0 và nhỏ hơn số mẫu.")

    # Support both sklearn generations: `max_iter` (new) and `n_iter` (old).
    n_iter = int(params.pop("max_iter", params.pop("n_iter", 1000)))
    learning_rate = params.pop("learning_rate", "auto")
    if isinstance(learning_rate, str) and learning_rate.lower() != "auto":
        raise ValueError("learning_rate dạng chuỗi chỉ chấp nhận giá trị 'auto'.")
    init = params.pop("init", "pca")

    # cuML TSNE currently supports 2D output. Requests for 3D therefore use
    # sklearn rather than silently changing the requested embedding dimension.
    cuda_available = torch is not None and torch.cuda.is_available()
    cuml_failure: str | None = None
    if cuda_available and n_components == 2:
        try:
            from cuml.manifold import TSNE as CuMLTSNE

            gpu_params: dict[str, Any] = {
                "n_components": n_components,
                "perplexity": perplexity,
                "learning_rate": 200.0 if isinstance(learning_rate, str) and learning_rate.lower() == "auto" else float(learning_rate),
                "init": init,
                "max_iter": n_iter,
                "random_state": int(random_state),
                "method": "fft",
                "output_type": "numpy",
            }
            # cuML uses `learning_rate_method='adaptive'` rather than sklearn's
            # literal `learning_rate='auto'`.
            if isinstance(learning_rate, str) and learning_rate.lower() == "auto":
                gpu_params["learning_rate_method"] = "adaptive"
            else:
                gpu_params["learning_rate_method"] = "none"

            # Forward additional options only when supported by the installed
            # cuML version. This keeps the adapter compatible across releases.
            gpu_params.update(params)
            try:
                signature = inspect.signature(CuMLTSNE)
                parameters = signature.parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                if parameters and not accepts_kwargs:
                    supported = set(parameters)
                    gpu_params = {key: value for key, value in gpu_params.items() if key in supported}
            except (TypeError, ValueError):
                # Some compiled bindings do not expose an inspectable signature;
                # the documented stable parameter set above remains usable.
                pass

            embedding = CuMLTSNE(**gpu_params).fit_transform(
                np.ascontiguousarray(X, dtype=np.float32)
            )
            result = _as_numpy_embedding(embedding)
            if result.shape != (len(X), n_components):
                raise ValueError(
                    f"cuML t-SNE trả shape {result.shape}, mong đợi {(len(X), n_components)}."
                )
            return result
        except Exception as exc:
            cuml_failure = f"cuML TSNE không chạy được: {exc}"
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    elif cuda_available and n_components != 2:
        cuml_failure = "cuML TSNE hiện chỉ hỗ trợ embedding 2D; yêu cầu này sẽ dùng sklearn CPU."

    # Framework CPU fallback: scikit-learn TSNE. Keep its own `auto` learning
    # rate semantics and adapt max_iter/n_iter to the installed sklearn version.
    cpu_params: dict[str, Any] = {
        "n_components": n_components,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "init": init,
        "random_state": int(random_state),
        "max_iter": n_iter,
    }
    cpu_params.update(params)
    sklearn_signature = inspect.signature(TSNE).parameters
    if "max_iter" not in sklearn_signature:
        cpu_params["n_iter"] = cpu_params.pop("max_iter")
    cpu_params = {key: value for key, value in cpu_params.items() if key in sklearn_signature}

    if cuml_failure and not _CUMl_TSNE_FALLBACK_WARNED:
        warnings.warn(
            f"{cuml_failure}. Stability đang fallback sang scikit-learn TSNE trên CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        _CUMl_TSNE_FALLBACK_WARNED = True
    elif not cuda_available and not _CUMl_TSNE_FALLBACK_WARNED:
        # No warning is needed for an ordinary CPU-only machine.
        _CUMl_TSNE_FALLBACK_WARNED = True

    result = np.asarray(TSNE(**cpu_params).fit_transform(X), dtype=np.float64)
    if result.shape != (len(X), n_components) or not np.isfinite(result).all():
        raise ValueError("scikit-learn TSNE trả về embedding không hợp lệ.")
    return result



def calculate_dbscan_eps(
    X: np.ndarray,
    min_samples: int = 5,
    quantile: float = 90.0,
) -> float:
    """Dùng nguyên hàm ước lượng eps của codebase để giữ kết quả tương thích."""
    return float(estimate_dbscan_eps(X, min_samples=min_samples, quantile=quantile))


def _fit_cluster_labels(
    X: np.ndarray,
    algorithm: str,
    params: dict | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """K-Means chạy CUDA; GMM và DBSCAN giữ backend CPU an toàn."""
    params = dict(params or {})
    if algorithm == "kmeans":
        try:
            return _gpu_kmeans(X, params, random_state)
        except Exception:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            config = {"n_clusters": 5, "n_init": 10, "max_iter": 300}
            config.update(params)
            config["random_state"] = random_state
            return np.asarray(KMeans(**config).fit_predict(X), dtype=int)
    if algorithm == "gmm":
        config = {"n_components": 5, "covariance_type": "full", "n_init": 1, "max_iter": 100}
        config.update(params)
        config["random_state"] = random_state
        return np.asarray(GaussianMixture(**config).fit_predict(X), dtype=int)
    if algorithm == "dbscan":
        config = {"eps": None, "min_samples": min(5, len(X)), "metric": "euclidean"}
        config.update(params)
        if config["eps"] is None:
            config["eps"] = calculate_dbscan_eps(X, min_samples=config["min_samples"])
        # Avoid O(N^2) GPU DBSCAN allocations; sklearn is the safe universal backend.
        return np.asarray(DBSCAN(**config).fit_predict(X), dtype=int)
    raise ValueError("algorithm phải là 'kmeans', 'gmm' hoặc 'dbscan'.")



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
                calculate_ari(repeated_labels[i], repeated_labels[j])
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
    """k-NN trên CUDA theo batch; CPU fallback nếu CUDA không đủ bộ nhớ."""
    X = _validate_matrix(X)
    if n_neighbors < 1 or n_neighbors >= len(X):
        raise ValueError("n_neighbors phải nằm trong [1, n_samples - 1].")
    try:
        return _gpu_knn_indices(X, n_neighbors)
    except Exception:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    """Trustworthiness tính bằng rank pairwise trên CUDA, CPU fallback khi cần."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    n = len(X_original)
    if n_neighbors < 1 or n_neighbors >= n / 2:
        raise ValueError("n_neighbors phải lớn hơn 0 và nhỏ hơn n_samples / 2.")
    try:
        device = _cuda_device()
        _, reduced_order, _ = _pairwise_rank_matrix(X_reduced, device)
        _, _, original_ranks = _pairwise_rank_matrix(X_original, device)
        penalties = 0.0
        orig_knn = torch.topk(torch.where(torch.eye(n, device=device, dtype=torch.bool), float("inf"), torch.cdist(_torch_matrix(X_original, device), _torch_matrix(X_original, device))), k=n_neighbors, largest=False).indices
        reduced_knn = reduced_order[:, :n_neighbors]
        for i in range(n):
            mask = ~torch.isin(reduced_knn[i], orig_knn[i])
            intrusions = reduced_knn[i][mask]
            if intrusions.numel():
                penalties += torch.clamp(original_ranks[i, intrusions].to(torch.float64) - n_neighbors, min=0).sum().item()
        denom = n * n_neighbors * (2*n - 3*n_neighbors - 1)
        return float(1.0 - (2.0 * penalties / denom))
    except Exception:
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()
        return float(trustworthiness(X_original, X_reduced, n_neighbors=n_neighbors))



def calculate_continuity(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Continuity tính bằng rank pairwise trên CUDA, CPU fallback khi cần."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    n = len(X_original)
    if n_neighbors < 1 or n_neighbors >= n / 2:
        raise ValueError("n_neighbors phải lớn hơn 0 và nhỏ hơn n_samples / 2.")
    try:
        device = _cuda_device()
        x_orig = _torch_matrix(X_original, device)
        x_red = _torch_matrix(X_reduced, device)
        d_orig = torch.cdist(x_orig, x_orig); d_orig.fill_diagonal_(float("inf"))
        d_red = torch.cdist(x_red, x_red); d_red.fill_diagonal_(float("inf"))
        orig_knn = torch.topk(d_orig, k=n_neighbors, largest=False).indices
        red_order = torch.argsort(d_red, dim=1)
        red_ranks = torch.empty_like(red_order)
        red_ranks.scatter_(1, red_order, torch.arange(1, n+1, device=device).expand_as(red_order))
        penalties = 0.0
        for i in range(n):
            missing = orig_knn[i][~torch.isin(orig_knn[i], red_order[i, :n_neighbors])]
            if missing.numel():
                penalties += torch.clamp(red_ranks[i, missing].to(torch.float64)-n_neighbors, min=0).sum().item()
        denom = n * n_neighbors * (2*n - 3*n_neighbors - 1)
        return float(1.0 - (2.0 * penalties / denom))
    except Exception:
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()
        # Original implementation from CPU backend.
        neighbors_original = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_original).kneighbors(X_original, return_distance=False)
        neighbors_reduced = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_reduced).kneighbors(X_reduced, return_distance=False)
        penalties = 0.0
        for i in range(n):
            orig = [j for j in neighbors_original[i] if j != i][:n_neighbors]
            red = [j for j in neighbors_reduced[i] if j != i][:n_neighbors]
            missing = [j for j in orig if j not in set(red)]
            if missing:
                distances = pairwise_distances(X_reduced[i:i+1], X_reduced)[0]
                ranks = rankdata(distances, method="average")
                penalties += float(np.maximum(ranks[missing] - n_neighbors, 0.0).sum())
        denom = n * n_neighbors * (2*n - 3*n_neighbors - 1)
        return float(1.0 - (2.0 * penalties / denom))



def calculate_knn_preservation(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """k-NN Jaccard trên CUDA khi đủ bộ nhớ; fallback sklearn."""
    X_original = _validate_matrix(X_original, "X_original")
    X_reduced = _validate_matrix(X_reduced, "X_reduced")
    if len(X_original) != len(X_reduced):
        raise ValueError("X_original và X_reduced phải có cùng số dòng.")
    try:
        original = torch.as_tensor(_gpu_knn_indices(X_original, n_neighbors), device=_cuda_device())
        reduced = torch.as_tensor(_gpu_knn_indices(X_reduced, n_neighbors), device=_cuda_device())
        intersections = (original[:, :, None] == reduced[:, None, :]).any(dim=2).sum(dim=1).to(torch.float32)
        return float((intersections / (2*n_neighbors - intersections)).mean().item())
    except Exception:
        original = _cpu.calculate_knn_preservation(X_original, X_reduced, n_neighbors) if '_cpu' in globals() else None
        if original is not None: return float(original)
        original_neighbors = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_original).kneighbors(X_original, return_distance=False)
        reduced_neighbors = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_reduced).kneighbors(X_reduced, return_distance=False)
        vals=[]
        for i, (a,b) in enumerate(zip(original_neighbors, reduced_neighbors)):
            sa=set(a[a!=i][:n_neighbors].tolist()); sb=set(b[b!=i][:n_neighbors].tolist())
            vals.append(len(sa & sb)/len(sa | sb))
        return float(np.mean(vals))



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
    """PCA/t-SNE ưu tiên CUDA; UMAP CUDA qua cuML nếu cài, còn lại CPU."""
    params = dict(method_params or {})
    if method == "pca":
        from models.dimensionality_reduction_gpu import run_pca
        config = dict(params)
        result = run_pca(X, n_components=config.pop("n_components", n_components), random_state=random_state)
        return np.asarray(result["X_pca"], dtype=np.float64)
    if method == "tsne":
        config = {"n_components": n_components, "perplexity": 30.0, "learning_rate": "auto", "n_iter": 1000}
        config.update(params)
        return _fit_tsne_embedding(X, config, random_state)
    if method == "umap":
        config = {"n_components": n_components, "n_neighbors": min(15, len(X)-1), "min_dist": 0.1, "metric": "euclidean"}
        config.update(params)
        try:
            from cuml.manifold import UMAP as CuMLUMAP
            model = CuMLUMAP(**config, random_state=int(random_state))
            result = model.fit_transform(_torch_matrix(X, _cuda_device()))
            if hasattr(result, "get"): result = result.get()
            return np.asarray(result, dtype=np.float64)
        except ImportError:
            from umap import UMAP
            return np.asarray(UMAP(**config, random_state=int(random_state)).fit_transform(X), dtype=np.float64)
    raise ValueError("method phải là 'pca', 'tsne' hoặc 'umap'.")



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