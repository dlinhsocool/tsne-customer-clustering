"""models/prediction.py - Tiền xử lý, chiếu tọa độ và dự đoán cụm cho khách hàng mới."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def preprocess_new_customer(
    new_customer: dict[str, Any] | pd.DataFrame,
    preprocess_info: dict[str, Any],
    scale_info: Any,
) -> dict[str, Any]:
    """Biến đổi khách hàng mới theo đúng pipeline đã huấn luyện và trả về dict chuẩn."""
    if isinstance(new_customer, dict):
        df_new = pd.DataFrame([new_customer])
    else:
        df_new = new_customer.copy()

    cfg = preprocess_info.get("preprocess_config", preprocess_info)

    # 1. Xử lý cột ngày tháng nếu có
    date_cols = cfg.get("date_columns", [])
    if date_cols:
        parsed_new = {
            c: pd.to_datetime(df_new[c], errors="coerce", format="mixed")
            for c in date_cols
            if c in df_new.columns
        }
        if parsed_new:
            ref_str = cfg.get("ref_date")
            ref_date = (
                pd.to_datetime(ref_str)
                if ref_str
                else max(s.max() for s in parsed_new.values())
            )
            first_col = next((c for c in parsed_new if "first" in c.lower()), None)
            last_col = next((c for c in parsed_new if "last" in c.lower()), None)
            if last_col:
                df_new["recency_days"] = (ref_date - parsed_new[last_col]).dt.days.fillna(0).clip(lower=0)
            if first_col:
                df_new["tenure_days"] = (ref_date - parsed_new[first_col]).dt.days.fillna(0).clip(lower=0)
            if first_col and last_col:
                df_new["active_span_days"] = (
                    parsed_new[last_col] - parsed_new[first_col]
                ).dt.days.fillna(0).clip(lower=0)
            for col in parsed_new:
                if col not in {first_col, last_col}:
                    df_new[f"{col}_days_ago"] = (ref_date - parsed_new[col]).dt.days.fillna(0).clip(lower=0)
            df_new = df_new.drop(columns=list(parsed_new.keys()))

    # 2. Log-transform các cột đã chỉ định ở bước preprocess_data
    for col in cfg.get("log_transform_columns", []):
        if col in df_new.columns:
            vals = pd.to_numeric(df_new[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            df_new[col] = np.log1p(np.maximum(vals, 0.0))

    # 3. Mã hóa biến phân loại
    cat_cols = [c for c in cfg.get("categorical_columns", []) if c in df_new.columns]
    if cat_cols:
        df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=False, dtype=float)

    # 4. Căn chỉnh đúng thứ tự cột huấn luyện
    feature_columns: list[str] = cfg["feature_columns"]
    medians: dict[str, float] = cfg.get("medians", {})
    aligned = pd.DataFrame(0.0, index=df_new.index, columns=feature_columns)
    for col in feature_columns:
        if col in df_new.columns:
            fill_v = medians.get(col, 0.0)
            aligned[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(fill_v)

    X_arr = aligned.to_numpy(dtype=float)

    # 5. Chuẩn hóa theo scaler
    if isinstance(scale_info, dict):
        scaler = scale_info["scaler"]
        log_cols_idx = scale_info.get("log_cols_idx", getattr(scaler, "_custom_log_cols_idx", []))
    else:
        scaler = scale_info
        log_cols_idx = getattr(scaler, "_custom_log_cols_idx", [])

    for j in log_cols_idx:
        if j < X_arr.shape[1]:
            X_arr[:, j] = np.log1p(np.maximum(X_arr[:, j], 0.0))

    X_scaled = scaler.transform(X_arr)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "X_new": aligned,
        "X_new_scaled": X_scaled,
        "feature_columns": feature_columns,
    }


def predict_customer_cluster(
    X_new_scaled: np.ndarray,
    clustering_result: dict[str, Any],
    X_train_scaled: np.ndarray | None = None,
    method: str = "auto",
    n_neighbors: int = 5,
) -> dict[str, Any]:
    """Dự đoán nhãn cụm cho khách hàng mới từ mô hình K-Means, GMM hoặc DBSCAN."""
    X_q = np.atleast_2d(np.asarray(X_new_scaled, dtype=float))
    algo = str(clustering_result.get("algorithm", "")).lower()
    model = clustering_result.get("model")

    use_knn = method.lower() == "knn"
    if not use_knn and algo in {"kmeans", "gmm"} and model is not None and hasattr(model, "predict"):
        preds = model.predict(X_q).astype(int)
        probs = model.predict_proba(X_q) if hasattr(model, "predict_proba") else None
        return {
            "cluster": int(preds[0]),
            "clusters": preds,
            "probabilities": probs,
            "confidence": float(np.max(probs[0])) if probs is not None else None,
            "method": algo,
        }

    if X_train_scaled is None:
        raise ValueError("Cần truyền X_train_scaled để dự đoán cụm bằng k-NN / DBSCAN.")

    train_labels = np.asarray(clustering_result["labels"], dtype=int)
    k = min(max(1, int(n_neighbors)), len(X_train_scaled))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train_scaled)
    dists, indices = nn.kneighbors(X_q)

    pred_clusters: list[int] = []
    confidences: list[float] = []
    for row_idx in indices:
        nbr_labels = train_labels[row_idx]
        vals, counts = np.unique(nbr_labels, return_counts=True)
        best_pos = int(np.argmax(counts))
        pred_clusters.append(int(vals[best_pos]))
        confidences.append(float(counts[best_pos] / k))

    preds_arr = np.array(pred_clusters, dtype=int)
    return {
        "cluster": int(preds_arr[0]),
        "clusters": preds_arr,
        "confidence": confidences[0],
        "neighbor_distances": dists,
        "method": "knn_fallback",
    }


def transform_new_customer(
    X_new_scaled: np.ndarray,
    tsne_result: dict[str, Any],
    X_train_scaled: np.ndarray | None = None,
    method: str = "auto",
    n_neighbors: int = 10,
) -> dict[str, Any]:
    """Chiếu khách hàng mới lên không gian 2D của t-SNE (hỗ trợ cả openTSNE và nội suy k-NN)."""
    # Tự xử lý nếu caller truyền nhầm thứ tự (X_new_scaled, X_train_scaled, tsne_result)
    if isinstance(tsne_result, np.ndarray) and isinstance(X_train_scaled, dict):
        X_train_scaled, tsne_result = tsne_result, X_train_scaled

    X_q = np.atleast_2d(np.asarray(X_new_scaled, dtype=float))
    emb_train = np.asarray(tsne_result["embedding"], dtype=float)

    embedding_obj = tsne_result.get("embedding_obj") or tsne_result.get("model")
    has_opentsne = embedding_obj is not None and hasattr(embedding_obj, "transform")

    use_method = method.lower()
    if use_method in {"auto", "opentsne"} and has_opentsne:
        coords_2d = np.asarray(embedding_obj.transform(X_q), dtype=float)
        used = "opentsne"
    else:
        if X_train_scaled is None:
            raise ValueError("Cần truyền X_train_scaled khi dùng method='knn'.")
        X_tr = np.asarray(X_train_scaled, dtype=float)
        k = min(max(1, int(n_neighbors)), len(X_tr))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(X_tr)
        dists, indices = nn.kneighbors(X_q)

        weights = 1.0 / np.maximum(dists, 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        coords_2d = np.einsum("ij,ijk->ik", weights, emb_train[indices])
        used = "knn"

    return {
        "embedding": coords_2d,
        "coordinates": coords_2d,
        "x": float(coords_2d[0, 0]),
        "y": float(coords_2d[0, 1]),
        "method": used,
    }