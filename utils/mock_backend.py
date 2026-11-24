"""
utils/mock_backend.py
=====================
BACKEND GIẢ LẬP - chỉ để giao diện chạy được khi các module thật chưa viết xong.

  * Mỗi hàm có CÙNG tên, tham số và cấu trúc output với models/*.py (đúng hợp đồng).
  * Kết quả ở đây KHÔNG phải kết quả thực nghiệm. Giao diện luôn hiện cảnh báo "MOCK"
    khi một hàm đang dùng bản giả này.
  * Không import streamlit. Chỉ dùng numpy/pandas.
  * Các bạn không cần đọc hay sửa file này; chỉ cần viết thân hàm trong models/*.py.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models.clustering import make_clustering_result
from models.evaluation import is_computable
from utils.constants import DEFAULT_PARAMS, NOISE_LABEL, SUPPORTS_PREDICT


# ---------------------------------------------------------------------------
# Dữ liệu mẫu (chỉ để thử giao diện; KHÔNG phải dữ liệu của đồ án)
# ---------------------------------------------------------------------------
SAMPLE_FEATURES = ["spending", "purchase_frequency", "online_time", "age", "income", "cancellation_rate"]


def generate_sample_dataset(n: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Sinh dữ liệu khách hàng giả lập gồm 5 nhóm, có vài giá trị thiếu và cột true_segment."""
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


# ---------------------------------------------------------------------------
# Đối tượng giả (chỉ có các thuộc tính mà frontend/prediction cần)
# ---------------------------------------------------------------------------
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
    """Model giả của K-Means / GMM (có predict). Dùng khoảng cách tới tâm."""

    def __init__(self, centers: np.ndarray, with_proba: bool = False):
        self.centers = centers
        self.with_proba = with_proba
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
    """Model giả của DBSCAN: cố tình KHÔNG có predict()."""

    def __init__(self, eps: float):
        self.eps = eps


# ---------------------------------------------------------------------------
# A. Data processing
# ---------------------------------------------------------------------------
def load_dataset(file, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(file, sep=sep, encoding=encoding)
    if df.empty:
        raise ValueError("File CSV không có dòng dữ liệu nào.")
    if df.shape[1] < 2:
        raise ValueError("File chỉ có 1 cột. Hãy kiểm tra lại ký tự phân tách (dấu , hoặc ;).")
    return df


def validate_dataset(df: pd.DataFrame, feature_columns: list[str] | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    missing = df.isna().sum()
    n_dup = int(df.duplicated().sum())
    if len(numeric) < 2:
        errors.append("Cần ít nhất 2 cột số để phân cụm.")
    if len(df) < 10:
        errors.append("Dữ liệu quá ít dòng (cần ít nhất 10).")
    for c, v in missing[missing > 0].items():
        warnings.append(f"Cột {c} có {int(v)} giá trị thiếu.")
    if n_dup:
        warnings.append(f"Có {n_dup} dòng trùng hoàn toàn.")
    for c in df.columns:
        if "id" in c.lower() and df[c].duplicated().any():
            warnings.append(f"Cột {c} có giá trị trùng (một khách xuất hiện nhiều dòng?).")
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
        "errors": errors,
        "warnings": warnings,
    }


def preprocess_data(df, feature_columns, missing_strategy="median", drop_duplicates=True,
                    outlier_method=None, log_transform_columns=None, sample_size=None,
                    random_state=42) -> dict:
    log_cols = [c for c in (log_transform_columns or []) if c in feature_columns]
    work = df.copy()
    n_before = len(work)
    steps: list[str] = []

    if drop_duplicates:
        before = len(work)
        work = work.drop_duplicates()
        steps.append(f"Bỏ {before - len(work)} dòng trùng.")

    fill_values: dict = {}
    if missing_strategy == "drop":
        before = len(work)
        work = work.dropna(subset=feature_columns)
        steps.append(f"Bỏ {before - len(work)} dòng có giá trị thiếu.")
    else:
        for c in feature_columns:
            if work[c].isna().any():
                if missing_strategy == "mean":
                    val = work[c].mean()
                elif missing_strategy == "most_frequent":
                    val = work[c].mode().iloc[0]
                else:
                    val = work[c].median()
                fill_values[c] = float(val)
                steps.append(f"Điền {int(work[c].isna().sum())} giá trị thiếu của {c} bằng {missing_strategy} = {val:.3f}.")
                work[c] = work[c].fillna(val)
        for c in feature_columns:  # fill_values đầy đủ để dùng cho khách mới
            fill_values.setdefault(c, float(work[c].median() if missing_strategy != "mean" else work[c].mean()))

    if outlier_method in ("iqr", "zscore"):
        feats = work[feature_columns]
        if outlier_method == "iqr":
            q1, q3 = feats.quantile(0.25), feats.quantile(0.75)
            iqr = q3 - q1
            bad = ((feats < q1 - 1.5 * iqr) | (feats > q3 + 1.5 * iqr)).any(axis=1)
        else:
            z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
            bad = (z.abs() > 3).any(axis=1)
        work = work[~bad]
        steps.append(f"Loại {int(bad.sum())} dòng ngoại lai ({outlier_method}).")

    if sample_size and len(work) > sample_size:
        work = work.sample(n=int(sample_size), random_state=random_state).sort_index()
        steps.append(f"Lấy mẫu {int(sample_size)} dòng (random_state={random_state}).")

    work = work.reset_index(drop=True)
    X = work[feature_columns].astype(float).copy()
    for c in log_cols:
        X[c] = np.log1p(X[c].clip(lower=0))
    if log_cols:
        steps.append(f"log1p các cột: {', '.join(log_cols)}.")

    config = {
        "feature_columns": list(feature_columns),
        "missing_strategy": missing_strategy,
        "fill_values": fill_values,
        "log_transform_columns": log_cols,
        "outlier_method": outlier_method,
        "feature_ranges": {c: (float(work[c].min()), float(work[c].max())) for c in feature_columns},
    }
    return {
        "df_clean": work, "X": X, "feature_columns": list(feature_columns),
        "preprocess_config": config, "n_rows_before": int(n_before),
        "n_rows_after": int(len(work)), "n_removed": int(n_before - len(work)),
        "steps_log": steps,
    }


def scale_data(X, scaler=None) -> dict:
    cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    arr = np.asarray(X, dtype=float)
    if scaler is None:
        std = arr.std(axis=0)
        scaler = MockScaler(arr.mean(axis=0), np.where(std == 0, 1.0, std))
    return {"X_scaled": scaler.transform(arr), "scaler": scaler, "feature_columns": cols}


# ---------------------------------------------------------------------------
# B. Dimensionality reduction
# ---------------------------------------------------------------------------
def _svd(X: np.ndarray):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc, U, S, Vt


def run_pca(X_scaled, n_components=0.95, random_state=42) -> dict:
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
        "pca_model": model, "X_pca": model.transform(X), "n_components": k,
        "explained_variance_ratio": ratio[:k], "cumulative_variance": cum[:k],
        "components": Vt[:k],
    }


def get_explained_variance(pca_model) -> pd.DataFrame:
    r = np.asarray(pca_model.explained_variance_ratio_)
    return pd.DataFrame({
        "component": np.arange(1, len(r) + 1),
        "explained_variance_ratio": r,
        "cumulative_variance": np.cumsum(r),
    })


def run_tsne(X_input, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000,
             random_state=42, use_opentsne=False) -> dict:
    X = np.asarray(X_input, dtype=float)
    rng = np.random.default_rng(random_state)
    Xc, U, S, _ = _svd(X)
    k = min(n_components, U.shape[1])
    emb = U[:, :k] * S[:k]
    emb = emb / (emb.std(axis=0, keepdims=True) + 1e-9) * 12.0
    if k < n_components:
        emb = np.hstack([emb, rng.normal(0, 1, (len(X), n_components - k))])
    emb = emb + rng.normal(0, 0.4 + abs(perplexity - 30) / 60, emb.shape)
    model = SimpleNamespace(X_fit=X, embedding=emb)
    return {
        "embedding": emb, "n_components": int(n_components), "perplexity": float(perplexity),
        "learning_rate": learning_rate, "n_iter": int(n_iter), "kl_divergence": None,
        "tsne_model": model, "backend": "mock",
    }


# ---------------------------------------------------------------------------
# C. Clustering
# ---------------------------------------------------------------------------
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


def run_clustering(X, algorithm, params=None, feature_space="scaled", random_state=42) -> dict:
    X = np.asarray(X, dtype=float)
    p = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    t0 = time.perf_counter()
    if algorithm == "kmeans":
        k = max(1, min(int(p["n_clusters"]), len(X)))
        lab, centers, d = _kmeans(X, k, random_state)
        model = MockCentroidModel(centers)
        res = make_clustering_result("kmeans", model, lab, p, time.perf_counter() - t0,
                                     cluster_centers=centers,
                                     extra={"inertia": float((d[np.arange(len(X)), lab] ** 2).sum())})
    elif algorithm == "gmm":
        k = max(1, min(int(p["n_components"]), len(X)))
        lab, centers, _ = _kmeans(X, k, random_state)
        model = MockCentroidModel(centers, with_proba=True)
        res = make_clustering_result("gmm", model, lab, p, time.perf_counter() - t0,
                                     cluster_centers=centers, probabilities=model.predict_proba(X),
                                     extra={"bic": None, "aic": None})
    else:
        k = 4
        lab, centers, d = _kmeans(X, k, random_state)
        ratio = min(0.2, 0.02 + 0.006 * int(p["min_samples"]) + 0.02 / max(float(p["eps"]), 0.1))
        dist = d[np.arange(len(X)), lab]
        cut = np.quantile(dist, 1 - ratio)
        lab = np.where(dist > cut, NOISE_LABEL, lab)
        # đánh số lại để nhãn liên tục 0..k-1
        remap = {old: new for new, old in enumerate(sorted(set(lab.tolist()) - {NOISE_LABEL}))}
        lab = np.array([remap.get(v, NOISE_LABEL) for v in lab])
        res = make_clustering_result("dbscan", MockDBSCANModel(float(p["eps"])), lab, p,
                                     time.perf_counter() - t0)
    res["feature_space"] = feature_space
    return res


# ---------------------------------------------------------------------------
# D. Evaluation (cài đặt numpy đơn giản, đúng quy tắc None của hợp đồng)
# ---------------------------------------------------------------------------
def calculate_silhouette_score(X, labels, sample_size=10000, random_state=42):
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    X, labels = X[mask], labels[mask]
    cap = min(sample_size or 1500, 1500)
    if len(X) > cap:
        idx = np.random.default_rng(random_state).choice(len(X), cap, replace=False)
        X, labels = X[idx], labels[idx]
    ids = np.unique(labels)
    if len(ids) < 2:
        return None
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    onehot = (labels[:, None] == ids[None, :]).astype(float)
    sums = D @ onehot
    cnt = onehot.sum(axis=0)
    own = onehot.argmax(axis=1)
    a = sums[np.arange(len(X)), own] / np.clip(cnt[own] - 1, 1, None)
    other = sums / np.clip(cnt[None, :], 1, None)
    other[np.arange(len(X)), own] = np.inf
    b = other.min(axis=1)
    s = (b - a) / np.clip(np.maximum(a, b), 1e-12, None)
    s[cnt[own] == 1] = 0.0
    return float(s.mean())


def calculate_davies_bouldin_score(X, labels):
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    X, labels = X[mask], labels[mask]
    ids = np.unique(labels)
    cent = np.array([X[labels == i].mean(axis=0) for i in ids])
    s = np.array([np.linalg.norm(X[labels == i] - cent[j], axis=1).mean() for j, i in enumerate(ids)])
    m = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(m, np.inf)
    r = (s[:, None] + s[None, :]) / np.clip(m, 1e-12, None)
    np.fill_diagonal(r, -np.inf)
    return float(r.max(axis=1).mean())


def calculate_ari(labels_true, labels_pred):
    if labels_true is None:
        return None
    a = pd.factorize(np.asarray(labels_true))[0]
    b = pd.factorize(np.asarray(labels_pred))[0]
    table = pd.crosstab(a, b).to_numpy().astype(float)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(axis=1)).sum()
    sum_b = comb(table.sum(axis=0)).sum()
    total = comb(float(table.sum()))
    expected = sum_a * sum_b / total if total else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def evaluate_clustering(X, labels, labels_true=None, runtime_seconds=None) -> dict:
    labels = np.asarray(labels)
    _, note = is_computable(labels)
    n_noise = int((labels == NOISE_LABEL).sum())
    return {
        "silhouette_score": calculate_silhouette_score(X, labels),
        "davies_bouldin_score": calculate_davies_bouldin_score(X, labels),
        "ari_score": calculate_ari(labels_true, labels),
        "n_clusters": int(len(set(labels.tolist()) - {NOISE_LABEL})),
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / max(len(labels), 1)),
        "runtime_seconds": runtime_seconds,
        "note": note,
    }


def compare_models(results, X_scaled, X_pca=None, labels_true=None) -> pd.DataFrame:
    rows = []
    for algo, res in results.items():
        X = X_pca if (res.get("feature_space") == "pca" and X_pca is not None) else X_scaled
        m = evaluate_clustering(X, res["labels"], labels_true, res.get("runtime_seconds"))
        rows.append({"algorithm": algo, "n_clusters": m["n_clusters"], "n_noise": m["n_noise"],
                     "silhouette_score": m["silhouette_score"],
                     "davies_bouldin_score": m["davies_bouldin_score"],
                     "ari_score": m["ari_score"], "runtime_seconds": m["runtime_seconds"]})
    cols = ["algorithm", "n_clusters", "n_noise", "silhouette_score",
            "davies_bouldin_score", "ari_score", "runtime_seconds"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# E. Customer profiling (mô tả sinh từ số liệu thật của dữ liệu đang chạy)
# ---------------------------------------------------------------------------
def calculate_cluster_profiles(df_clean, labels, feature_columns, include_noise=True) -> dict:
    d = df_clean[feature_columns].copy()
    d["_cluster"] = np.asarray(labels).astype(int)
    if not include_noise:
        d = d[d["_cluster"] != NOISE_LABEL]
    grp = d.groupby("_cluster")
    means = grp[feature_columns].mean()
    sizes = grp.size()
    table = means.copy()
    table.insert(0, "ratio", sizes / max(len(df_clean), 1))
    table.insert(0, "size", sizes)
    table.index.name = "cluster_id"
    g_mean = df_clean[feature_columns].mean()
    g_std = df_clean[feature_columns].std(ddof=0).replace(0, 1.0)
    z = (means - g_mean) / g_std
    z.index.name = "cluster_id"
    return {"profile_table": table, "z_scores": z, "global_mean": g_mean,
            "cluster_ids": [int(i) for i in table.index]}


def generate_cluster_description(cluster_id, profiles, feature_labels=None, top_n=2, z_threshold=0.5) -> dict:
    labels_map = feature_labels or {}
    row = profiles["z_scores"].loc[cluster_id]
    table = profiles["profile_table"].loc[cluster_id]
    size, ratio = int(table["size"]), float(table["ratio"])
    if cluster_id == NOISE_LABEL:
        return {"cluster_id": cluster_id, "name": "Noise: khách bất thường",
                "description": f"{size} khách ({ratio:.1%}) không thuộc cụm nào, nên xem riêng.",
                "top_features": []}
    ranked = row.reindex(row.abs().sort_values(ascending=False).index)
    tops = [{"feature": f, "direction": "high" if z > 0 else "low", "z_score": float(z)}
            for f, z in ranked.items() if abs(z) >= z_threshold][:top_n]
    disp = lambda f: labels_map.get(f, f)  # noqa: E731
    if tops:
        title = ", ".join(f"{disp(t['feature'])} {'cao' if t['direction'] == 'high' else 'thấp'}" for t in tops)
        detail = "; ".join(
            f"{disp(t['feature'])} trung bình {table[t['feature']]:.2f} "
            f"(toàn bộ {profiles['global_mean'][t['feature']]:.2f})" for t in tops)
        desc = f"{size} khách ({ratio:.1%}). Nổi bật: {detail}."
    else:
        title = "gần mức trung bình"
        desc = f"{size} khách ({ratio:.1%}). Không đặc trưng nào lệch rõ so với toàn bộ."
    return {"cluster_id": int(cluster_id), "name": f"Nhóm {cluster_id}: {title}",
            "description": desc, "top_features": tops}


# ---------------------------------------------------------------------------
# F. Out-of-sample
# ---------------------------------------------------------------------------
def preprocess_new_customer(new_data, preprocess_config, scaler, pca_model=None) -> dict:
    cols = preprocess_config["feature_columns"]
    row = pd.DataFrame([new_data]) if isinstance(new_data, dict) else new_data.copy()
    warnings: list[str] = []
    for c in cols:
        if c not in row.columns or row[c].isna().any():
            row[c] = preprocess_config["fill_values"].get(c, 0.0)
            warnings.append(f"Thiếu {c}: đã điền giá trị {preprocess_config['fill_values'].get(c, 0.0):.3f}.")
    raw = row[cols].astype(float).reset_index(drop=True)
    out = [c for c in cols if not (preprocess_config["feature_ranges"][c][0]
                                   <= raw.loc[0, c] <= preprocess_config["feature_ranges"][c][1])]
    Xt = raw.copy()
    for c in preprocess_config["log_transform_columns"]:
        Xt[c] = np.log1p(Xt[c].clip(lower=0))
    Xs = scaler.transform(Xt.to_numpy())
    Xp = pca_model.transform(Xs) if pca_model is not None else None
    return {"X_new_raw": raw, "X_new_scaled": Xs, "X_new_pca": Xp,
            "out_of_range_features": out, "warnings": warnings}


def _knn(x, X_ref, k):
    d = np.linalg.norm(np.asarray(X_ref, dtype=float) - np.asarray(x, dtype=float).reshape(1, -1), axis=1)
    idx = np.argsort(d)[:k]
    return idx, d[idx]


def predict_customer_cluster(x_new, clustering_result, X_reference, method="auto", k=5) -> dict:
    x = np.asarray(x_new, dtype=float).reshape(1, -1)
    labels = np.asarray(clustering_result["labels"])
    supports = clustering_result["supports_predict"]
    warnings: list[str] = []
    if method == "auto":
        method = "model_predict" if supports else "knn"
    if method == "model_predict" and not supports:
        warnings.append("Thuật toán này không có predict(); đã chuyển sang kNN.")
        method = "knn"
    idx, dist = _knn(x, X_reference, k)
    probs, conf = None, None
    if method == "model_predict":
        model = clustering_result["model"]
        cluster = int(model.predict(x)[0])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            conf = float(probs.max())
    else:
        votes = labels[idx]
        vals, cnts = np.unique(votes, return_counts=True)
        cluster = int(vals[cnts.argmax()])
        conf = float(cnts.max() / len(idx))
        if clustering_result["algorithm"] == "dbscan":
            eps = float(clustering_result["params"].get("eps", 0.5))
            if dist[0] > eps:
                cluster, conf = NOISE_LABEL, None
                warnings.append(f"Hàng xóm gần nhất cách {dist[0]:.2f} > eps={eps}: coi là noise.")
    ref = np.asarray(X_reference, dtype=float)
    sample = ref[np.random.default_rng(0).choice(len(ref), min(len(ref), 300), replace=False)]
    ref_d = [np.sort(np.linalg.norm(ref - s, axis=1))[1:k + 1].mean() for s in sample]
    thr = float(np.percentile(ref_d, 97))
    is_out = bool(dist.mean() > thr)
    if is_out:
        warnings.append("Khách mới xa mọi cụm hơn 97% khách trong dữ liệu huấn luyện.")
    return {"cluster": cluster, "is_noise": cluster == NOISE_LABEL, "method_used": method,
            "confidence": conf, "probabilities": probs, "neighbor_indices": idx,
            "neighbor_distances": dist, "mean_neighbor_distance": float(dist.mean()),
            "is_outlier": is_out, "warnings": warnings}


def transform_new_customer(x_new, tsne_result, X_reference, method="knn", k=5) -> dict:
    warnings: list[str] = []
    emb = np.asarray(tsne_result["embedding"], dtype=float)
    idx, _ = _knn(x_new, X_reference, k)
    coords = emb[idx].mean(axis=0, keepdims=True)
    used = "knn"
    if method == "opentsne":
        model = tsne_result.get("tsne_model")
        if model is not None and hasattr(model, "transform"):
            coords = np.asarray(model.transform(np.asarray(x_new, dtype=float).reshape(1, -1)))
            used = "opentsne"
        else:
            warnings.append("Không có tsne_model.transform(); đã dùng kNN.")
    return {"coordinates": coords, "n_components": int(coords.shape[1]), "method_used": used,
            "neighbor_indices": idx, "warnings": warnings}
