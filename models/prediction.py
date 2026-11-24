"""models/prediction.py - Tiền xử lý, chiếu tọa độ và dự đoán cụm cho khách hàng mới."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


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
        log_cols_idx = scale_info.get(
            "log_cols_idx", getattr(scaler, "_custom_log_cols_idx", [])
        )
    else:
        scaler = scale_info
        log_cols_idx = getattr(scaler, "_custom_log_cols_idx", [])

    return scaler, [int(index) for index in log_cols_idx]


def _detect_out_of_range_features(
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    scaler: Any,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Tìm đặc trưng vượt miền huấn luyện khi scaler có lưu được miền đó.

    Với StandardScaler, không có min/max huấn luyện; dùng ngưỡng |z| > 3 như
    một cảnh báo thống kê, không coi đó là giới hạn tuyệt đối của dữ liệu.
    """
    found: set[str] = set()
    warnings: list[str] = []

    data_min = getattr(scaler, "data_min_", None)
    data_max = getattr(scaler, "data_max_", None)
    if data_min is not None and data_max is not None:
        lower = np.asarray(data_min, dtype=float).reshape(-1)
        upper = np.asarray(data_max, dtype=float).reshape(-1)
        for index, feature in enumerate(feature_columns):
            if index < len(lower) and index < len(upper):
                if X_raw[0, index] < lower[index] or X_raw[0, index] > upper[index]:
                    found.add(feature)
        if found:
            warnings.append(
                "Một số đặc trưng nằm ngoài khoảng min/max đã thấy khi huấn luyện."
            )
        return sorted(found), warnings

    if scaler.__class__.__name__ == "StandardScaler":
        for index, feature in enumerate(feature_columns):
            if index < X_scaled.shape[1] and np.any(np.abs(X_scaled[:, index]) > 3.0):
                found.add(feature)
        if found:
            warnings.append(
                "Một số đặc trưng có |z-score| > 3; đây là cảnh báo thống kê, "
                "không phải giới hạn min/max tuyệt đối."
            )

    return sorted(found), warnings


def _get_pca_model(pca_info: Any) -> Any | None:
    """Nhận PCA model trực tiếp hoặc từ dict kết quả run_pca."""
    if pca_info is None:
        return None
    if isinstance(pca_info, dict):
        return pca_info.get("pca_model") or pca_info.get("model")
    if hasattr(pca_info, "transform"):
        return pca_info
    return None


def preprocess_new_customer(
    new_customer: dict[str, Any] | pd.DataFrame,
    preprocess_info: dict[str, Any],
    scale_info: Any,
    pca_info: Any | None = None,
    *,
    pca_model: Any | None = None,
) -> dict[str, Any]:
    """Biến đổi khách hàng mới theo pipeline đã fit và trả về dict chuẩn.

    X_new_raw là DataFrame đã làm sạch/mã hóa/căn chỉnh cột nhưng chưa scale.
    X_new_scaled là ma trận sau scaler. X_new_pca là None nếu không truyền PCA model.

    `pca_info` là tên tham số cũ; `pca_model` được hỗ trợ để tương thích với
    pages/new_customer.py. Có thể truyền model PCA trực tiếp hoặc dict từ run_pca().
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

    # 1. Xử lý ngày tháng giống preprocess_data.
    date_cols = cfg.get("date_columns", [])
    if date_cols:
        parsed_new = {
            column: pd.to_datetime(df_new[column], errors="coerce", format="mixed")
            for column in date_cols
            if column in df_new.columns
        }
        if parsed_new:
            ref_str = cfg.get("ref_date")
            if ref_str:
                ref_date = pd.to_datetime(ref_str)
            else:
                valid_maxes = [series.max() for series in parsed_new.values() if series.notna().any()]
                ref_date = max(valid_maxes) if valid_maxes else pd.NaT

            first_col = next(
                (column for column in parsed_new if "first" in column.lower()), None
            )
            last_col = next(
                (column for column in parsed_new if "last" in column.lower()), None
            )

            if pd.isna(ref_date):
                warnings.append(
                    "Không xác định được ngày tham chiếu; các đặc trưng ngày thiếu được gán 0."
                )
            else:
                if last_col:
                    df_new["recency_days"] = (
                        ref_date - parsed_new[last_col]
                    ).dt.days.fillna(0).clip(lower=0)
                if first_col:
                    df_new["tenure_days"] = (
                        ref_date - parsed_new[first_col]
                    ).dt.days.fillna(0).clip(lower=0)
                if first_col and last_col:
                    df_new["active_span_days"] = (
                        parsed_new[last_col] - parsed_new[first_col]
                    ).dt.days.fillna(0).clip(lower=0)
                for column in parsed_new:
                    if column not in {first_col, last_col}:
                        df_new[f"{column}_days_ago"] = (
                            ref_date - parsed_new[column]
                        ).dt.days.fillna(0).clip(lower=0)

            df_new = df_new.drop(columns=list(parsed_new.keys()))

    # 2. Điền missing bằng median học từ tập train TRƯỚC khi log-transform,
    # đúng thứ tự xử lý trong preprocess_data().
    medians = cfg.get("medians", {})
    for column in cfg.get("numeric_columns", []):
        fill_value = float(medians.get(column, 0.0))
        if column not in df_new.columns:
            df_new[column] = fill_value
        else:
            df_new[column] = (
                pd.to_numeric(df_new[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(fill_value)
            )

    # 3. Log-transform các cột được chỉ định ở bước preprocessing.
    for column in cfg.get("log_transform_columns", []):
        if column in df_new.columns:
            values = pd.to_numeric(df_new[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            df_new[column] = np.log1p(np.maximum(values, 0.0))

    # 4. One-Hot Encode đúng những biến phân loại đã biết.
    categorical_cols = list(cfg.get("categorical_columns", []))
    for column in categorical_cols:
        if column not in df_new.columns:
            df_new[column] = "Unknown"
        else:
            df_new[column] = df_new[column].fillna("Unknown").astype(str)
    if categorical_cols:
        df_new = pd.get_dummies(
            df_new, columns=categorical_cols, drop_first=False, dtype=float
        )

    # 5. Căn chỉnh thứ tự và số lượng cột theo dữ liệu huấn luyện.
    feature_columns = list(cfg["feature_columns"])
    aligned = pd.DataFrame(0.0, index=df_new.index, columns=feature_columns)

    extra_columns = [column for column in df_new.columns if column not in feature_columns]
    if extra_columns:
        warnings.append(
            "Bỏ qua các cột không xuất hiện trong dữ liệu huấn luyện: "
            + ", ".join(map(str, extra_columns[:10]))
        )

    for column in feature_columns:
        if column in df_new.columns:
            fill_value = float(medians.get(column, 0.0))
            aligned[column] = (
                pd.to_numeric(df_new[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(fill_value)
            )

    X_raw = aligned.to_numpy(dtype=float)

    # 6. Áp dụng log1p ở scaler nếu scale_data đã tự động bật cho cột đó.
    scaler, log_cols_idx = _get_scaler(scale_info)
    X_for_scaler = X_raw.copy()
    for index in log_cols_idx:
        if 0 <= index < X_for_scaler.shape[1]:
            X_for_scaler[:, index] = np.log1p(
                np.maximum(X_for_scaler[:, index], 0.0)
            )

    X_scaled = np.asarray(scaler.transform(X_for_scaler), dtype=float)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    out_of_range_features, range_warnings = _detect_out_of_range_features(
        X_for_scaler, X_scaled, scaler, feature_columns
    )
    warnings.extend(range_warnings)

    # 7. PCA chỉ được tính khi caller truyền PCA model đã fit trên X_scaled.
    X_pca = None
    pca_model = _get_pca_model(pca_info)
    if pca_model is not None:
        X_pca = np.asarray(pca_model.transform(X_scaled), dtype=float)

    return {
        "X_new_raw": aligned.copy(),
        "X_new_scaled": X_scaled,
        "X_new_pca": X_pca,
        "out_of_range_features": out_of_range_features,
        "warnings": warnings,
        # Alias cũ.
        "X_new": aligned,
        "feature_columns": feature_columns,
    }


def predict_customer_cluster(
    X_new_scaled: np.ndarray,
    clustering_result: dict[str, Any],
    X_train_scaled: np.ndarray | None = None,
    method: str = "auto",
    n_neighbors: int = 5,
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Dự đoán cụm mới; mọi thuật toán trả về cùng một cấu trúc dict.

    K-Means/GMM dùng model.predict() ở chế độ auto. DBSCAN không có predict(),
    nên sử dụng k-NN trên X_train_scaled và nhãn đã học.
    """
    if k is not None:
        n_neighbors = int(k)

    X_query = _as_2d_float(X_new_scaled, "X_new_scaled")
    algorithm = str(clustering_result.get("algorithm", "")).lower()
    model = clustering_result.get("model")
    method_requested = str(method).lower()
    if method_requested not in {"auto", "knn"}:
        raise ValueError("method chỉ nhận 'auto' hoặc 'knn'.")

    if not algorithm:
        raise ValueError("clustering_result thiếu key 'algorithm'.")

    # Nhánh dự đoán trực tiếp bằng mô hình đã fit.
    if (
        method_requested == "auto"
        and algorithm in {"kmeans", "gmm"}
        and model is not None
        and hasattr(model, "predict")
    ):
        predictions = np.asarray(model.predict(X_query), dtype=int)
        probabilities = (
            np.asarray(model.predict_proba(X_query), dtype=float)
            if hasattr(model, "predict_proba")
            else None
        )
        confidence = (
            np.max(probabilities, axis=1).astype(float)
            if probabilities is not None
            else None
        )
        clusters = predictions
        first_cluster = int(clusters[0])
        return {
            "cluster": first_cluster,
            "clusters": clusters,
            "is_noise": bool(first_cluster == -1),
            "method_used": "model_predict",
            "confidence": float(confidence[0]) if confidence is not None else None,
            "probabilities": probabilities,
            "neighbor_indices": None,
            "neighbor_distances": None,
            "mean_neighbor_distance": None,
            "is_outlier": None,
            "warnings": [],
            # Alias cũ.
            "method": algorithm,
        }

    if X_train_scaled is None:
        raise ValueError(
            "Cần truyền X_train_scaled để dự đoán bằng k-NN hoặc với DBSCAN."
        )

    X_train = _as_2d_float(X_train_scaled, "X_train_scaled")
    train_labels = np.asarray(clustering_result.get("labels"), dtype=int)
    if train_labels.ndim != 1 or len(train_labels) != len(X_train):
        raise ValueError(
            "clustering_result['labels'] phải có một nhãn cho mỗi dòng X_train_scaled."
        )
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError("X_new_scaled và X_train_scaled phải có cùng số đặc trưng.")

    k = min(max(1, int(n_neighbors)), len(X_train))
    neighbors_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors_model.fit(X_train)
    distances, indices = neighbors_model.kneighbors(X_query)

    predictions: list[int] = []
    confidences: list[float] = []
    for neighbor_row in indices:
        neighbor_labels = train_labels[neighbor_row]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        best_position = int(np.argmax(counts))
        predictions.append(int(values[best_position]))
        confidences.append(float(counts[best_position] / len(neighbor_row)))

    clusters = np.asarray(predictions, dtype=int)
    first_cluster = int(clusters[0])
    is_noise = first_cluster == -1
    mean_neighbor_distances = np.mean(distances, axis=1)
    mean_neighbor_distance = float(mean_neighbor_distances[0])

    warnings: list[str] = []
    # Hợp đồng output chỉ chấp nhận hai giá trị chuẩn:
    # model_predict (dùng model.predict) hoặc knn (dùng láng giềng gần nhất).
    used_method = "knn"
    if algorithm != "dbscan":
        warnings.append(
            f"{algorithm} được dự đoán bằng k-NN theo yêu cầu method='knn', "
            "không gọi model.predict()."
        )

    if is_noise:
        warnings.append(
            "Các láng giềng gần nhất bỏ phiếu cho nhãn noise (-1); "
            "khách hàng mới chưa thuộc cụm mật độ đã học."
        )

    return {
        "cluster": first_cluster,
        "clusters": clusters,
        "is_noise": bool(is_noise),
        "method_used": used_method,
        "confidence": float(confidences[0]),
        "probabilities": None,
        "neighbor_indices": indices,
        "neighbor_distances": distances,
        "mean_neighbor_distance": mean_neighbor_distance,
        "is_outlier": bool(is_noise) if algorithm == "dbscan" else None,
        "warnings": warnings,
        # Alias cũ.
        "method": "knn_fallback",
        "mean_neighbor_distances": mean_neighbor_distances,
    }


def transform_new_customer(
    X_new_scaled: np.ndarray,
    tsne_result: dict[str, Any],
    X_train_scaled: np.ndarray | None = None,
    method: str = "auto",
    n_neighbors: int = 10,
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Chiếu khách hàng mới lên embedding t-SNE bằng openTSNE hoặc nội suy k-NN."""
    if k is not None:
        n_neighbors = int(k)

    # Tương thích với cách gọi cũ có thể truyền nhầm thứ tự hai đối số đầu.
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

    requested_method = str(method).lower()
    if requested_method not in {"auto", "knn", "opentsne"}:
        raise ValueError("method chỉ nhận 'auto', 'knn' hoặc 'opentsne'.")

    warnings: list[str] = []
    neighbor_indices = None

    if requested_method in {"auto", "opentsne"} and embedding_model is not None:
        coordinates = np.asarray(embedding_model.transform(X_query), dtype=float)
        used_method = "opentsne"
    else:
        if requested_method == "opentsne":
            warnings.append(
                "Không tìm thấy mô hình có transform(); đã chuyển sang nội suy k-NN."
            )
        if X_train_scaled is None:
            raise ValueError(
                "Cần truyền X_train_scaled để nội suy tọa độ t-SNE bằng k-NN."
            )

        X_train = _as_2d_float(X_train_scaled, "X_train_scaled")
        if len(X_train) != len(train_embedding):
            raise ValueError(
                "X_train_scaled và embedding t-SNE phải có cùng số dòng, "
                "cùng thứ tự khách hàng."
            )
        if X_train.shape[1] != X_query.shape[1]:
            raise ValueError("X_new_scaled và X_train_scaled phải có cùng số đặc trưng.")

        k = min(max(1, int(n_neighbors)), len(X_train))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(X_train)
        distances, neighbor_indices = nn.kneighbors(X_query)

        # Nếu trùng chính xác một điểm huấn luyện, chỉ lấy trung bình các điểm trùng.
        coordinates = np.zeros((len(X_query), train_embedding.shape[1]), dtype=float)
        for row_index, (distance_row, index_row) in enumerate(
            zip(distances, neighbor_indices)
        ):
            exact_matches = distance_row <= 1e-12
            if np.any(exact_matches):
                coordinates[row_index] = train_embedding[index_row[exact_matches]].mean(axis=0)
            else:
                weights = 1.0 / np.maximum(distance_row, 1e-12)
                weights /= weights.sum()
                coordinates[row_index] = np.sum(
                    train_embedding[index_row] * weights[:, None], axis=0
                )
        used_method = "knn"

    coordinates = _as_2d_float(coordinates, "coordinates")
    result = {
        "coordinates": coordinates,
        "embedding": coordinates,
        "n_components": int(coordinates.shape[1]),
        "method_used": used_method,
        "neighbor_indices": neighbor_indices,
        "warnings": warnings,
        # Alias cũ.
        "method": used_method,
        "x": float(coordinates[0, 0]),
        "y": float(coordinates[0, 1]),
    }
    if coordinates.shape[1] >= 3:
        result["z"] = float(coordinates[0, 2])
    return result
