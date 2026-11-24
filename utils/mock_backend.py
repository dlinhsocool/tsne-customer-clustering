"""
utils/mock_backend.py
=====================
BACKEND GIẢ LẬP - chỉ để giao diện chạy được khi module thật chưa viết xong (hợp đồng v2.0).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models.clustering import make_clustering_result
from models.evaluation import is_computable
from utils.constants import DEFAULT_PARAMS, NOISE_LABEL

SAMPLE_FEATURES = ["spending", "purchase_frequency", "online_time", "age", "income", "cancellation_rate"]


def generate_sample_dataset(n: int = 600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    centers = np.array([
        [900, 20, 120, 40, 90, 0.05],
        [400, 25, 90, 35, 55, 0.08],
        [150, 6, 200, 22, 25, 0.45],
        [120, 3, 40, 45, 40, 0.20],
        [200, 8, 70, 50, 95, 0.15],
    ], dtype=float)
    scales = np.array([120, 4, 25, 6, 10, 0.06])
    seg = rng.choice(5, size=n, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    X = centers[seg] + rng.normal(0, 1, (n, 6)) * scales
    X = np.clip(X, 0, None)
    X[:, 3] = np.clip(np.round(X[:, 3]), 18, 80)
    X[:, 5] = np.clip(X[:, 5], 0, 1)
    df = pd.DataFrame(X, columns=SAMPLE_FEATURES).round(3)
    df.insert(0, "customer_id", [f"C{i:05d}" for i in range(n)])
    df["true_segment"] = seg
    miss = rng.choice(n, size=max(3, n // 50), replace=False)
    df.loc[miss, "income"] = np.nan
    return df


class MockScaler:
    def __init__(self, mean_: np.ndarray, scale_: np.ndarray):
        self.mean_ = mean_
        self.scale_ = scale_

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


class MockPCA:
    def __init__(self, mean_, components_, explained_variance_ratio_):
        self.mean_ = mean_
        self.components_ = components_
        self.explained_variance_ratio_ = explained_variance_ratio_

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T


class MockCentroidModel:
    def __init__(self, centers: np.ndarray, with_proba: bool = False):
        self.centers = centers
        if with_proba:
            self.predict_proba = self._predict_proba

    def _dist(self, X):
        X = np.asarray(X, dtype=float)
        return np.linalg.norm(X[:, None, :] - self.centers[None, :, :], axis=2)

    def predict(self, X):
        return self._dist(X).argmin(axis=1)

    def _predict_proba(self, X):
        d = self._dist(X)
        w = np.exp(-(d ** 2) / 2.0)
        return w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)


class MockDBSCANModel:
    def __init__(self, eps: float):
        self.eps = eps


def load_dataset(file, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(file, sep=sep, encoding=encoding)
    if df.empty:
        raise ValueError("File CSV không có dòng dữ liệu nào.")
    if df.shape[1] < 2:
        raise ValueError("File chỉ có 1 cột.")
    return df


def validate_dataset(df: pd.DataFrame, feature_columns: list[str] | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    id_cols = [c for c in df.columns if "id" in c.lower() or "name" in c.lower()]
    dt_cols = [c for c in df.columns if "date" in c.lower() and c not in id_cols]
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in id_cols]
    categorical = [c for c in df.columns if c not in id_cols + dt_cols + numeric]
    missing = df.isna().sum()
    n_dup = int(df.duplicated().sum())
    if len(numeric) + len(categorical) + len(dt_cols) < 2:
        errors.append("Cần ít nhất 2 cột đặc trưng.")
    for c, v in missing[missing > 0].items():
        warnings.append(f"Cột {c} có {int(v)} giá trị thiếu.")
    if n_dup:
        warnings.append(f"Có {n_dup} dòng trùng hoàn toàn.")
    return {
        "is_valid": not errors,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(v) for c, v in missing.items()},
        "n_duplicates": n_dup,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": dt_cols,
        "id_columns": id_cols,
        "errors": errors,
        "warnings": warnings,
    }


def preprocess_data(
    df, feature_columns, missing_strategy="median", drop_duplicates=True,
    outlier_method=None, log_transform_columns=None, sample_size=None, random_state=42
) -> dict:
    from models.data_processing import preprocess_data as real_preprocess
    return real_preprocess(
        df, feature_columns, missing_strategy=missing_strategy,
        drop_duplicates=drop_duplicates, outlier_method=outlier_method,
        log_transform_columns=log_transform_columns, sample_size=sample_size,
        random_state=random_state,
    )


def scale_data(X, scaler=None) -> dict:
    cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    arr = np.asarray(X, dtype=float)
    if scaler is None:
        std = arr.std(axis=0)
        scaler = MockScaler(arr.mean(axis=0), np.where(std == 0, 1.0, std))
    return {"X_scaled": scaler.transform(arr), "scaler": scaler, "feature_columns": cols}


def _svd(X: np.ndarray):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc, U, S, Vt


def run_pca(X_scaled, n_components=0.95, random_state=42) -> dict:
    t0 = time.perf_counter()
    X = np.asarray(X_scaled, dtype=float)
    _, _, S, Vt = _svd(X)
    ratio = (S ** 2) / np.sum(S ** 2)
    cum = np.cumsum(ratio)
    if isinstance(n_components, float) and n_components < 1:
        k = int(np.searchsorted(cum, n_components) + 1)
    else:
        k = int(n_components)
    k = max(1, min(k, X.shape[1], X.shape[0]))
    model = MockPCA(X.mean(axis=0), Vt[:k], ratio[:k])
    return {
        "pca_model": model,
        "X_pca": model.transform(X),
        "n_components": k,
        "explained_variance_ratio": ratio[:k],
        "cumulative_variance": cum[:k],
        "components": Vt[:k],
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def get_explained_variance(pca_model) -> pd.DataFrame:
    r = np.asarray(pca_model.explained_variance_ratio_)
    return pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(r))],
        "explained_variance_ratio": r,
        "cumulative_variance": np.cumsum(r),
    })


def run_umap(X_input, n_components=5, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42) -> dict:
    t0 = time.perf_counter()
    X = np.asarray(X_input, dtype=float)
    rng = np.random.default_rng(random_state)
    _, U, S, _ = _svd(X)
    k = min(int(n_components), U.shape[1])
    emb = U[:, :k] * S[:k]
    if k < int(n_components):
        emb = np.hstack([emb, rng.normal(0, 1, (len(X), int(n_components) - k))])
    params = {"n_components": int(n_components), "n_neighbors": n_neighbors, "min_dist": min_dist, "metric": metric}
    return {
        "model": SimpleNamespace(params=params),
        "embedding": emb,
        "n_components": int(emb.shape[1]),
        "params": params,
        "runtime_seconds": float(time.perf_counter() - t0),
        "backend": "mock",
    }


def run_tsne(
    X_input, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000,
    random_state=42, use_opentsne=False
) -> dict:
    t0 = time.perf_counter()
    X = np.asarray(X_input, dtype=float)
    rng = np.random.default_rng(random_state)
    _, U, S, _ = _svd(X)
    k = min(n_components, U.shape[1])
    emb = U[:, :k] * S[:k]
    emb = emb / (emb.std(axis=0, keepdims=True) + 1e-9) * 12.0
    if k < n_components:
        emb = np.hstack([emb, rng.normal(0, 1, (len(X), n_components - k))])
    emb = emb + rng.normal(0, 0.4 + abs(perplexity - 30) / 60, emb.shape)
    model = SimpleNamespace(X_fit=X, embedding=emb)
    return {
        "embedding": emb,
        "n_components": int(n_components),
        "perplexity": float(perplexity),
        "learning_rate": learning_rate,
        "n_iter": int(n_iter),
        "kl_divergence": None,
        "tsne_model": model,
        "backend": "mock",
        "device": "cpu",
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def _kmeans(X: np.ndarray, k: int, seed: int, n_iter: int = 20):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        lab = d.argmin(axis=1)
        for j in range(k):
            if (lab == j).any():
                centers[j] = X[lab == j].mean(axis=0)
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return d.argmin(axis=1), centers, d


def run_clustering(X, algorithm, params=None, feature_space="original", random_state=42) -> dict:
    X = np.asarray(X, dtype=float)
    norm_space = "original" if feature_space == "scaled" else feature_space
    p = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    if p.get("eps") is None and algorithm == "dbscan":
        p["eps"] = 0.5
    t0 = time.perf_counter()
    if algorithm == "kmeans":
        k = max(1, min(int(p["n_clusters"]), len(X)))
        lab, centers, d = _kmeans(X, k, random_state)
        model = MockCentroidModel(centers)
        res = make_clustering_result(
            "kmeans", model, lab, p, time.perf_counter() - t0,
            cluster_centers=centers,
            extra={"inertia": float((d[np.arange(len(X)), lab] ** 2).sum())},
        )
    elif algorithm == "gmm":
        k = max(1, min(int(p["n_components"]), len(X)))
        lab, centers, _ = _kmeans(X, k, random_state)
        model = MockCentroidModel(centers, with_proba=True)
        res = make_clustering_result(
            "gmm", model, lab, p, time.perf_counter() - t0,
            cluster_centers=centers, probabilities=model.predict_proba(X),
        )
    else:
        k = 4
        lab, centers, d = _kmeans(X, k, random_state)
        ratio = min(0.2, 0.02 + 0.006 * int(p["min_samples"]) + 0.02 / max(float(p["eps"]), 0.1))
        dist = d[np.arange(len(X)), lab]
        cut = np.quantile(dist, 1 - ratio)
        lab = np.where(dist > cut, NOISE_LABEL, lab)
        remap = {old: new for new, old in enumerate(sorted(set(lab.tolist()) - {NOISE_LABEL}))}
        lab = np.array([remap.get(v, NOISE_LABEL) for v in lab])
        res = make_clustering_result(
            "dbscan", MockDBSCANModel(float(p["eps"])), lab, p, time.perf_counter() - t0
        )
    res["feature_space"] = norm_space
    return res


def evaluate_clustering(X, labels, labels_true=None, runtime_seconds=None) -> dict:
    from models.evaluation import evaluate_clustering as real_eval
    return real_eval(X, labels, labels_true=labels_true, runtime_seconds=runtime_seconds)


def compare_models(results, X_scaled, X_pca=None, labels_true=None, X_umap=None, X_tsne=None) -> pd.DataFrame:
    from models.evaluation import compare_models as real_cmp
    return real_cmp(results, X_scaled, X_pca=X_pca, labels_true=labels_true, X_umap=X_umap, X_tsne=X_tsne)


def compare_dimensionality_reduction_method(*a, **k) -> pd.DataFrame:
    from models.evaluation import compare_dimensionality_reduction_method as real_dr_cmp
    return real_dr_cmp(*a, **k)


def benchmark_clustering_spaces(*a, **k) -> dict:
    from models.evaluation import benchmark_clustering_spaces as real_bench
    return real_bench(*a, **k)


def calculate_cluster_profiles(df_clean, labels, feature_columns, include_noise=True) -> dict:
    from models.customer_profiling import calculate_cluster_profiles as real_prof
    return real_prof(df_clean, labels, feature_columns, include_noise=include_noise)


def generate_cluster_description(cluster_id, profiles, feature_labels=None, top_n=2, z_threshold=0.5) -> dict:
    from models.customer_profiling import generate_cluster_description as real_desc
    return real_desc(cluster_id, profiles, feature_labels=feature_labels, top_n=top_n, z_threshold=z_threshold)


def preprocess_new_customer(new_data, preprocess_config, scaler, pca_model=None) -> dict:
    from models.prediction import preprocess_new_customer as real_pnc
    return real_pnc(new_data, preprocess_config, scaler, pca_model=pca_model)


def predict_customer_cluster(x_new, clustering_result, X_reference, method="auto", k=5) -> dict:
    from models.prediction import predict_customer_cluster as real_pcc
    return real_pcc(x_new, clustering_result, X_reference, method=method, k=k)


def transform_new_customer(x_new, tsne_result, X_reference, method="knn", k=5) -> dict:
    from models.prediction import transform_new_customer as real_tnc
    return real_tnc(x_new, tsne_result, X_reference, method=method, k=k)