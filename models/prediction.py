"""models/prediction.py - Tiền xử lý, gán cụm và chiếu tọa độ cho khách hàng mới.

Nguyên tắc:
  * Khách mới đi qua ĐÚNG pipeline của khách cũ: cùng ngày tham chiếu, cùng công thức n_orders,
    cùng log1p, cùng danh sách mức One-Hot, cùng scaler đã fit (chỉ transform, không fit lại).
  * Cụm được quyết định ở không gian phân cụm; vị trí trên bản đồ t-SNE chỉ để quan sát
    (t-SNE không có phép chiếu cố định cho điểm mới).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_REFERENCE_SAMPLE = 2000  # số điểm huấn luyện dùng để ước lượng phân phối khoảng cách hàng xóm


def _as_2d_float(values: Any, name: str) -> np.ndarray:
    """Chuyển input thành ma trận float 2 chiều và kiểm tra giá trị."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} phải là ma trận 2 chiều.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} không được rỗng.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} chứa NaN hoặc Inf.")
    return array


def _get_scaler(scale_info: Any) -> tuple[Any, list[int]]:
    """Lấy scaler và các cột cần log1p trước khi scale."""
    if isinstance(scale_info, dict):
        scaler = scale_info.get("scaler")
        if scaler is None:
            raise KeyError("scale_info phải chứa key 'scaler'.")
        log_cols_idx = scale_info.get("log_cols_idx", getattr(scaler, "_custom_log_cols_idx", []))
    else:
        scaler = scale_info
        log_cols_idx = getattr(scaler, "_custom_log_cols_idx", [])
    return scaler, [int(i) for i in log_cols_idx]


def _get_pca_model(pca_info: Any) -> Any | None:
    """Nhận PCA model trực tiếp hoặc từ dict kết quả run_pca."""
    if pca_info is None:
        return None
    if isinstance(pca_info, dict):
        return pca_info.get("pca_model") or pca_info.get("model")
    return pca_info if hasattr(pca_info, "transform") else None


# ---------------------------------------------------------------------------
# 1. Tiền xử lý khách mới
# ---------------------------------------------------------------------------
def preprocess_new_customer(
    new_customer: dict[str, Any] | pd.DataFrame,
    preprocess_info: dict[str, Any],
    scale_info: Any,
    pca_info: Any | None = None,
    *,
    pca_model: Any | None = None,
) -> dict[str, Any]:
    """Biến đổi khách hàng mới theo pipeline đã fit.

    `new_customer` có thể dùng:
      * tên đặc trưng sau xử lý: recency_days, n_orders, total_spent, loyalty_points, support_tickets,
        subscription, favorite_category (khớp form), hoặc
      * tên cột thô: first_purchase_date, last_purchase_date, avg_order_value... (recency_days và n_orders
        được tính lại bằng đúng công thức của dữ liệu huấn luyện).

    Lỗi dữ liệu không hợp lý (âm, ngày đảo ngược, thiếu cột bắt buộc) -> ValueError tiếng Việt.
    Giá trị ngoài khoảng đã thấy khi huấn luyện -> cảnh báo (out_of_range_features), không chặn.
    """
    if pca_model is not None:
        if pca_info is not None:
            raise ValueError("Chỉ truyền một trong hai tham số pca_info hoặc pca_model.")
        pca_info = pca_model

    if isinstance(new_customer, dict):
        df_new = pd.DataFrame([new_customer])
    elif isinstance(new_customer, pd.DataFrame):
        df_new = new_customer.copy()
    else:
        raise TypeError("new_customer phải là dict hoặc pandas DataFrame.")
    if df_new.empty:
        raise ValueError("new_customer không được rỗng.")

    cfg = preprocess_info.get("preprocess_config", preprocess_info)
    if "feature_columns" not in cfg:
        raise KeyError("preprocess_info phải chứa 'feature_columns'.")

    warnings: list[str] = []
    medians = cfg.get("medians", {})
    num_cols = list(cfg.get("numeric_columns", []))
    bin_cols = list(cfg.get("binary_columns", []))
    cat_cols = list(cfg.get("categorical_columns", []))

    # 1. Cột ngày thô -> recency_days / tenure_days bằng NGÀY THAM CHIẾU ĐÃ LƯU.
    ref_str = cfg.get("ref_date")
    first_col, last_col = cfg.get("first_date_column"), cfg.get("last_date_column")
    parsed: dict[str, pd.Series] = {}
    for col in cfg.get("date_columns", []):
        if col in df_new.columns:
            parsed[col] = pd.to_datetime(df_new[col], errors="coerce", format="mixed")
            if parsed[col].isna().any():
                raise ValueError(f"Ngày không hợp lệ ở cột '{col}'. Dùng định dạng YYYY-MM-DD.")
    if parsed and ref_str:
        ref_ts = pd.Timestamp(ref_str)
        if first_col in parsed and last_col in parsed and (parsed[last_col] < parsed[first_col]).any():
            raise ValueError("Ngày mua gần nhất không được trước ngày mua đầu tiên.")
        for col, series in parsed.items():
            if (series > ref_ts).any():
                raise ValueError(
                    f"'{col}' sau ngày tham chiếu {ref_ts.date()} của dữ liệu huấn luyện; "
                    "hãy nhập số ngày từ lần mua cuối (recency_days) thay vì ngày hôm nay."
                )
        if last_col in parsed and "recency_days" not in df_new.columns:
            df_new["recency_days"] = (ref_ts - parsed[last_col]).dt.days.clip(lower=0)
        if first_col in parsed and "tenure_days" in cfg.get("feature_columns", []) and "tenure_days" not in df_new.columns:
            df_new["tenure_days"] = (ref_ts - parsed[first_col]).dt.days.clip(lower=0)

    # 2. n_orders = round(total_spent / avg_order_value) nếu chưa có.
    src = cfg.get("n_orders_source")
    if "n_orders" in num_cols and "n_orders" not in df_new.columns and src:
        spent, avg = src
        if spent in df_new.columns and avg in df_new.columns:
            denom = pd.to_numeric(df_new[avg], errors="coerce")
            if (denom <= 0).any() or denom.isna().any():
                raise ValueError(f"'{avg}' phải là số dương để tính số đơn hàng.")
            df_new["n_orders"] = (pd.to_numeric(df_new[spent], errors="coerce") / denom).round().clip(lower=1)

    # 3. Kiểm tra cột số bắt buộc, giá trị âm, thiếu.
    for col in num_cols:
        if col not in df_new.columns:
            warnings.append(f"Thiếu '{col}': dùng trung vị của dữ liệu huấn luyện ({medians.get(col, 0.0):g}).")
            df_new[col] = float(medians.get(col, 0.0))
        values = pd.to_numeric(df_new[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            warnings.append(f"'{col}' trống/không hợp lệ: dùng trung vị của dữ liệu huấn luyện.")
            values = values.fillna(float(medians.get(col, 0.0)))
        if (values < 0).any():
            raise ValueError(f"'{col}' không được âm.")
        df_new[col] = values

    # 4. Khoảng đã thấy khi huấn luyện (thang đo gốc, trước log).
    ranges = cfg.get("raw_feature_ranges") or {}
    out_of_range: list[str] = []
    for col in num_cols:
        if col in ranges:
            lo, hi = ranges[col]
            if (df_new[col] < lo).any() or (df_new[col] > hi).any():
                out_of_range.append(col)
    if out_of_range:
        warnings.append(
            "Giá trị nằm ngoài khoảng đã thấy khi huấn luyện; kết quả kém chắc chắn hơn: "
            + ", ".join(out_of_range) + "."
        )

    # 5. log1p đúng các cột đã log lúc huấn luyện.
    for col in cfg.get("log_transform_columns", []):
        if col in df_new.columns:
            df_new[col] = np.log1p(np.maximum(df_new[col].to_numpy(dtype=float), 0.0))

    # 6. Nhị phân.
    for col in bin_cols:
        if col not in df_new.columns:
            df_new[col] = float(medians.get(col, 0.0))
        raw = df_new[col]
        if raw.dtype == object:
            raw = raw.astype(str).str.strip().str.lower().map(
                {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0}
            )
        df_new[col] = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(0, 1).astype(float)

    # 7. One-Hot theo danh sách mức CỐ ĐỊNH của dữ liệu huấn luyện.
    levels_cfg = cfg.get("category_levels", {})
    onehot = pd.DataFrame(index=df_new.index)
    for col in cat_cols:
        levels = levels_cfg.get(col, [])
        values = df_new[col].astype(str) if col in df_new.columns else pd.Series("Unknown", index=df_new.index)
        unknown = ~values.isin(levels)
        if unknown.any():
            warnings.append(
                f"'{col}' có giá trị chưa từng thấy ({', '.join(sorted(set(values[unknown])))}): "
                "mọi cột One-Hot của biến này được đặt 0."
            )
        for level in levels:
            onehot[f"{col}_{level}"] = (values == level).astype(float)

    # 8. Căn chỉnh theo đúng thứ tự cột huấn luyện.
    feature_columns = list(cfg["feature_columns"])
    full = pd.concat([df_new, onehot], axis=1)
    aligned = pd.DataFrame(0.0, index=df_new.index, columns=feature_columns)
    for col in feature_columns:
        if col in full.columns:
            aligned[col] = pd.to_numeric(full[col], errors="coerce").fillna(float(medians.get(col, 0.0)))
    X_raw = aligned.to_numpy(dtype=float)

    # 9. Scale bằng scaler ĐÃ FIT (không fit lại).
    scaler, log_cols_idx = _get_scaler(scale_info)
    X_for_scaler = X_raw.copy()
    for idx in log_cols_idx:
        if 0 <= idx < X_for_scaler.shape[1]:
            X_for_scaler[:, idx] = np.log1p(np.maximum(X_for_scaler[:, idx], 0.0))
    X_scaled = np.nan_to_num(np.asarray(scaler.transform(X_for_scaler), dtype=float))

    # Cảnh báo thống kê bổ sung: |z| > 4 trên cột liên tục.
    passthrough = set(getattr(scaler, "passthrough_idx", []))
    extreme = [
        feature_columns[j] for j in range(X_scaled.shape[1])
        if j not in passthrough and np.any(np.abs(X_scaled[:, j]) > 4.0) and feature_columns[j] not in out_of_range
    ]
    if extreme:
        warnings.append("Giá trị rất xa trung bình huấn luyện (|z| > 4): " + ", ".join(extreme) + ".")

    X_pca = None
    pca = _get_pca_model(pca_info)
    if pca is not None:
        X_pca = np.asarray(pca.transform(X_scaled), dtype=float)

    return {
        "X_new_raw": aligned.copy(),
        "X_new_scaled": X_scaled,
        "X_new_pca": X_pca,
        "out_of_range_features": sorted(out_of_range),
        "warnings": warnings,
        # alias cũ
        "X_new": aligned,
        "feature_columns": feature_columns,
    }


# ---------------------------------------------------------------------------
# 2. Gán cụm
# ---------------------------------------------------------------------------
def _reference_distance_threshold(X_train: np.ndarray, k: int, quantile: float = 95.0) -> float:
    """Ngưỡng 'xa bất thường': phân vị của khoảng cách TB tới k hàng xóm của chính dữ liệu huấn luyện."""
    n = len(X_train)
    if n <= k + 1:
        return float("inf")
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(_REFERENCE_SAMPLE, n), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_train)
    dist, _ = nn.kneighbors(X_train[idx])
    return float(np.percentile(dist[:, 1:].mean(axis=1), quantile))


def predict_customer_cluster(
    X_new_scaled: np.ndarray,
    clustering_result: dict[str, Any],
    X_train_scaled: np.ndarray | None = None,
    method: str = "auto",
    n_neighbors: int = 5,
    *,
    k: int | None = None,
    X_new_space: np.ndarray | None = None,
) -> dict[str, Any]:
    """Gán khách mới vào cụm.

    method:
      * "auto"          K-Means/GMM -> predict() của mô hình; DBSCAN -> kNN.
      * "knn"           bỏ phiếu k hàng xóm trong X_train_scaled (dùng nhãn đã học, mọi thuật toán).
      * "model_predict" gọi model.predict(); DBSCAN không hỗ trợ -> ValueError.

    X_new_space: điểm mới trong đúng không gian mà mô hình đã huấn luyện (feature_space).
    Không cần khi feature_space = "original". Với "pca" truyền X_new_pca; với "umap"/"tsne" không có
    phép biến đổi cho điểm mới nên chỉ dùng được kNN trên X_train_scaled.

    Luôn tính k hàng xóm trong X_train_scaled (để hiển thị và đo độ bất thường) khi có X_train_scaled.
    """
    if k is not None:
        n_neighbors = int(k)

    X_query = _as_2d_float(X_new_scaled, "X_new_scaled")
    algorithm = str(clustering_result.get("algorithm", "")).lower()
    if not algorithm:
        raise ValueError("clustering_result thiếu key 'algorithm'.")
    requested = str(method).lower()
    if requested not in {"auto", "knn", "model_predict"}:
        raise ValueError("method chỉ nhận 'auto', 'knn' hoặc 'model_predict'.")
    if requested == "model_predict" and algorithm == "dbscan":
        raise ValueError("DBSCAN không có predict(); hãy chọn phương pháp kNN.")

    model = clustering_result.get("model")
    space = clustering_result.get("feature_space") or "original"
    space = "original" if space == "scaled" else space
    can_predict = algorithm in {"kmeans", "gmm"} and model is not None and hasattr(model, "predict")

    if space == "original":
        X_for_model = X_query
    elif X_new_space is not None:
        X_for_model = _as_2d_float(X_new_space, "X_new_space")
    else:
        X_for_model = None

    warnings: list[str] = []
    use_model = can_predict and requested in {"auto", "model_predict"}
    if use_model and X_for_model is None:
        if requested == "model_predict":
            raise ValueError(
                f"Mô hình được huấn luyện trên không gian '{space}': cần truyền X_new_space "
                "(điểm mới trong cùng không gian) để dùng predict()."
            )
        warnings.append(
            f"Mô hình huấn luyện trên không gian '{space}' nên không gọi predict() được cho điểm mới; "
            "đã dùng kNN trên dữ liệu chuẩn hóa."
        )
        use_model = False

    # --- hàng xóm trong X_train_scaled -------------------------------------------------------
    indices = distances = None
    outlier = None
    mean_dist = None
    train_labels = None
    if X_train_scaled is not None:
        X_train = _as_2d_float(X_train_scaled, "X_train_scaled")
        if X_train.shape[1] != X_query.shape[1]:
            raise ValueError("X_new_scaled và X_train_scaled phải có cùng số đặc trưng.")
        raw_labels = clustering_result.get("labels")
        if raw_labels is not None:
            train_labels = np.asarray(raw_labels, dtype=int)
            if train_labels.ndim != 1 or len(train_labels) != len(X_train):
                raise ValueError("clustering_result['labels'] phải có một nhãn cho mỗi dòng X_train_scaled.")
        kk = min(max(1, int(n_neighbors)), len(X_train))
        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean").fit(X_train)
        distances, indices = nn.kneighbors(X_query)
        mean_dist = float(np.mean(distances[0]))
        outlier = bool(mean_dist > _reference_distance_threshold(X_train, kk))
    elif not use_model:
        raise ValueError("Cần truyền X_train_scaled để dự đoán bằng kNN hoặc với DBSCAN.")

    # --- gán cụm -------------------------------------------------------------------------------
    probabilities = None
    if use_model:
        cluster = int(np.asarray(model.predict(X_for_model), dtype=int)[0])
        used = "model_predict"
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X_for_model), dtype=float)
            confidence = float(probabilities.max())
            confidence_kind = "xác suất GMM"
        elif train_labels is not None and indices is not None:
            confidence = float(np.mean(train_labels[indices[0]] == cluster))
            confidence_kind = "tỷ lệ hàng xóm cùng cụm"
        else:
            confidence, confidence_kind = None, None
    else:
        if train_labels is None:
            raise ValueError("clustering_result thiếu 'labels' để bỏ phiếu kNN.")
        vals, counts = np.unique(train_labels[indices[0]], return_counts=True)
        best = int(np.argmax(counts))
        cluster = int(vals[best])
        confidence = float(counts[best] / indices.shape[1])
        confidence_kind = "tỷ lệ hàng xóm cùng cụm"
        used = "knn"
        if algorithm != "dbscan" and requested == "knn":
            warnings.append(f"{algorithm} được gán bằng kNN theo yêu cầu, không gọi model.predict().")

    is_noise = cluster == -1
    if is_noise:
        warnings.append(
            "Phần lớn hàng xóm gần nhất là noise (-1): khách này chưa thuộc cụm mật độ nào đã học."
        )
    if outlier:
        warnings.append("Khách này xa các hàng xóm hơn 95% khách huấn luyện: kết quả kém chắc chắn.")

    return {
        "cluster": cluster,
        "clusters": np.asarray([cluster], dtype=int),
        "is_noise": bool(is_noise),
        "method_used": used,
        "confidence": confidence,
        "confidence_kind": confidence_kind,
        "probabilities": probabilities,
        "neighbor_indices": indices,
        "neighbor_distances": distances,
        "mean_neighbor_distance": mean_dist,
        "is_outlier": outlier,
        "warnings": warnings,
        # alias cũ
        "method": algorithm,
    }


# ---------------------------------------------------------------------------
# 3. Vị trí trên bản đồ t-SNE (chỉ để quan sát)
# ---------------------------------------------------------------------------
def transform_new_customer(
    X_new_scaled: np.ndarray,
    tsne_result: dict[str, Any],
    X_train_scaled: np.ndarray | None = None,
    method: str = "auto",
    n_neighbors: int = 10,
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Đặt khách mới lên embedding t-SNE: openTSNE.transform() hoặc nội suy k hàng xóm (trọng số 1/d).

    t-SNE không có ánh xạ cố định cho điểm mới; tọa độ trả về chỉ để quan sát, không dùng để gán cụm.
    """
    if k is not None:
        n_neighbors = int(k)

    if isinstance(tsne_result, np.ndarray) and isinstance(X_train_scaled, dict):
        X_train_scaled, tsne_result = tsne_result, X_train_scaled

    X_query = _as_2d_float(X_new_scaled, "X_new_scaled")
    if not isinstance(tsne_result, dict) or "embedding" not in tsne_result:
        raise ValueError("tsne_result phải là dict có key 'embedding'.")

    train_embedding = _as_2d_float(tsne_result["embedding"], "tsne_result['embedding']")
    if train_embedding.shape[1] < 2:
        raise ValueError("Embedding t-SNE phải có ít nhất 2 chiều để hiển thị.")

    embedding_model = None
    for key in ("embedding_obj", "tsne_model", "model"):
        candidate = tsne_result.get(key)
        if candidate is not None and hasattr(candidate, "transform"):
            embedding_model = candidate
            break

    requested = str(method).lower()
    if requested not in {"auto", "knn", "opentsne"}:
        raise ValueError("method chỉ nhận 'auto', 'knn' hoặc 'opentsne'.")

    warnings: list[str] = []
    neighbor_indices = None

    if requested in {"auto", "opentsne"} and embedding_model is not None:
        coordinates = np.asarray(embedding_model.transform(X_query), dtype=float)
        used_method = "opentsne"
    else:
        if requested == "opentsne":
            warnings.append("Không có mô hình t-SNE hỗ trợ transform(); đã chuyển sang nội suy kNN.")
        if X_train_scaled is None:
            raise ValueError("Cần truyền X_train_scaled để nội suy tọa độ t-SNE bằng kNN.")
        X_train = _as_2d_float(X_train_scaled, "X_train_scaled")
        if len(X_train) != len(train_embedding):
            raise ValueError("X_train_scaled và embedding t-SNE phải cùng số dòng, cùng thứ tự khách hàng.")
        if X_train.shape[1] != X_query.shape[1]:
            raise ValueError("X_new_scaled và X_train_scaled phải có cùng số đặc trưng.")

        kk = min(max(1, int(n_neighbors)), len(X_train))
        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean").fit(X_train)
        distances, neighbor_indices = nn.kneighbors(X_query)

        coordinates = np.zeros((len(X_query), train_embedding.shape[1]), dtype=float)
        for r, (drow, irow) in enumerate(zip(distances, neighbor_indices)):
            exact = drow <= 1e-12
            if np.any(exact):
                coordinates[r] = train_embedding[irow[exact]].mean(axis=0)
            else:
                w = 1.0 / np.maximum(drow, 1e-12)
                w /= w.sum()
                coordinates[r] = np.sum(train_embedding[irow] * w[:, None], axis=0)
        used_method = "knn"

    coordinates = _as_2d_float(coordinates, "coordinates")
    result = {
        "coordinates": coordinates,
        "embedding": coordinates,
        "n_components": int(coordinates.shape[1]),
        "method_used": used_method,
        "neighbor_indices": neighbor_indices,
        "warnings": warnings,
        # alias cũ
        "method": used_method,
        "x": float(coordinates[0, 0]),
        "y": float(coordinates[0, 1]),
    }
    if coordinates.shape[1] >= 3:
        result["z"] = float(coordinates[0, 2])
    return result
