"""models/customer_profiling.py - Thống kê và mô tả đặc trưng từng cụm."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_cluster_profiles(
    df: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str] | None = None,
    include_noise: bool = True,
) -> dict[str, Any]:
    """Tính profile theo cụm từ ma trận đã tiền xử lý.

    Nên truyền `prep["X"]` thay vì `prep["df_clean"]`, vì X đã chứa toàn bộ
    cột sau Date Feature Extraction và One-Hot Encoding.

    Output có `profile_table`, `z_scores`, `global_mean`, `cluster_ids` theo
    contract và giữ alias cũ để tương thích những phần code khác.
    """
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("labels phải là mảng 1 chiều.")
    if labels_arr.size == 0:
        raise ValueError("labels không được rỗng.")
    try:
        labels_arr = labels_arr.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels phải chứa nhãn cụm dạng số nguyên.") from exc

    if isinstance(df, pd.DataFrame):
        if feature_columns is not None:
            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                preview = missing[:8]
                raise ValueError(
                    "DataFrame đầu vào thiếu cột đã mã hóa. Hãy truyền prep['X'] "
                    "thay vì prep['df_clean']. "
                    f"Thiếu {len(missing)} cột, ví dụ: {preview}"
                )
            data = df.loc[:, feature_columns].copy()
        else:
            data = df.select_dtypes(include=[np.number]).copy()
        if data.shape[1] == 0:
            raise ValueError("Không tìm thấy cột số để tính hồ sơ cụm.")
        data = data.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        arr = np.asarray(df, dtype=float)
        if arr.ndim != 2:
            raise ValueError("df dạng ndarray phải là ma trận 2 chiều.")
        if arr.shape[1] == 0:
            raise ValueError("Dữ liệu phải có ít nhất một đặc trưng.")
        if feature_columns is not None and len(feature_columns) != arr.shape[1]:
            raise ValueError(
                "Số feature_columns không khớp số cột của ma trận đầu vào."
            )
        cols = list(feature_columns) if feature_columns is not None else [
            f"feature_{i}" for i in range(arr.shape[1])
        ]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        data = pd.DataFrame(arr, columns=cols)

    if len(data) != len(labels_arr):
        raise ValueError(
            f"Số dòng dữ liệu ({len(data)}) không khớp số nhãn ({len(labels_arr)})."
        )

    if not include_noise:
        keep_mask = labels_arr != -1
        data = data.loc[keep_mask].reset_index(drop=True)
        labels_arr = labels_arr[keep_mask]
        if labels_arr.size == 0:
            raise ValueError("Không còn cụm nào sau khi loại noise.")

    # Dùng population standard deviation để z-score ổn định cả khi chỉ có 1 điểm/cụm.
    global_mean_series = data.mean(axis=0)
    global_std = data.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    global_mean = {str(col): float(global_mean_series[col]) for col in data.columns}

    cluster_ids = sorted(int(cid) for cid in np.unique(labels_arr))
    profile_rows: list[dict[str, Any]] = []
    legacy_profile_rows: list[dict[str, Any]] = []
    z_rows: list[dict[str, float]] = []
    cluster_sizes: dict[int, int] = {}
    cluster_percentages: dict[int, float] = {}
    top_features_by_cluster: dict[int, list[dict[str, Any]]] = {}

    for cid in cluster_ids:
        mask = labels_arr == cid
        group = data.loc[mask]
        group_mean = group.mean(axis=0)
        z_values = (group_mean - global_mean_series) / global_std

        size = int(mask.sum())
        ratio = float(size / len(labels_arr))
        cluster_sizes[cid] = size
        cluster_percentages[cid] = ratio * 100.0

        row: dict[str, Any] = {
            "cluster_id": cid,
            "cluster_name": "Noise / Outliers" if cid == -1 else f"Cluster {cid}",
            "size": size,
            "ratio": ratio,
            "percentage": ratio * 100.0,
        }
        z_row: dict[str, float] = {}
        legacy_row = dict(row)
        feature_items: list[dict[str, Any]] = []

        for col in data.columns:
            mean_value = float(group_mean[col])
            z_value = float(z_values[col])
            row[str(col)] = mean_value
            legacy_row[str(col)] = mean_value
            legacy_row[f"{col}_mean"] = mean_value  # alias cũ
            legacy_row[f"{col}_zscore"] = z_value   # alias cũ
            z_row[str(col)] = z_value
            feature_items.append({
                "feature": str(col),
                "cluster_mean": mean_value,
                "global_mean": float(global_mean_series[col]),
                "zscore": z_value,
            })

        feature_items.sort(key=lambda item: abs(item["zscore"]), reverse=True)
        top_features_by_cluster[cid] = feature_items[:5]
        profile_rows.append(row)
        legacy_profile_rows.append(legacy_row)
        z_rows.append(z_row)

    profile_table = pd.DataFrame(profile_rows).set_index("cluster_id", drop=False)
    legacy_profile_table = pd.DataFrame(legacy_profile_rows).set_index("cluster_id", drop=False)
    z_scores = pd.DataFrame(z_rows, index=pd.Index(cluster_ids, name="cluster_id"))

    return {
        # Contract v2.0.
        "profile_table": profile_table,
        "z_scores": z_scores,
        "global_mean": global_mean,
        "cluster_ids": cluster_ids,
        # Alias tương thích ngược.
        "profile_df": legacy_profile_table,
        "summary_table": legacy_profile_table,
        "n_clusters": len([cid for cid in cluster_ids if cid != -1]),
        "cluster_sizes": cluster_sizes,
        "cluster_percentages": cluster_percentages,
        "feature_columns": data.columns.tolist(),
        "global_means": global_mean,
        "top_features": top_features_by_cluster,
    }


def generate_cluster_description(
    cluster_id: int | dict[str, Any] | pd.DataFrame,
    profile_info: dict[str, Any] | pd.DataFrame | None = None,
    top_k: int = 3,
    feature_labels: dict[str, str] | None = None,
    z_threshold: float = 0.25,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Sinh mô tả cụm; hỗ trợ tham số `feature_labels` từ trang Streamlit."""
    if top_k < 0:
        raise ValueError("top_k phải >= 0.")

    if isinstance(cluster_id, (int, np.integer)) and profile_info is not None:
        return _describe_one(
            int(cluster_id), profile_info, top_k, feature_labels or {}, z_threshold
        )

    source = cluster_id if profile_info is None else profile_info
    table = _get_profile_table(source)
    cluster_ids = (
        source.get("cluster_ids", table.index.tolist())
        if isinstance(source, dict)
        else table.index.tolist()
    )
    return [
        _describe_one(int(cid), source, top_k, feature_labels or {}, z_threshold)
        for cid in cluster_ids
    ]


def _get_profile_table(profile_info: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(profile_info, pd.DataFrame):
        return profile_info
    for key in ("profile_table", "profile_df", "summary_table"):
        value = profile_info.get(key)
        if isinstance(value, pd.DataFrame):
            return value
    raise KeyError("profile_info cần chứa DataFrame 'profile_table'.")


def _describe_one(
    cid: int,
    profile_info: dict[str, Any] | pd.DataFrame,
    top_k: int,
    feature_labels: dict[str, str],
    z_threshold: float,
) -> dict[str, Any]:
    table = _get_profile_table(profile_info)
    if cid in table.index:
        row = table.loc[cid]
    else:
        matched = table[table["cluster_id"] == cid]
        if matched.empty:
            raise ValueError(f"Không tìm thấy cluster_id={cid} trong profile_table.")
        row = matched.iloc[0]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    size = int(row.get("size", 0))
    ratio = float(row.get("ratio", float(row.get("percentage", 0.0)) / 100.0))
    name = "Noise / Outliers" if cid == -1 else f"Cluster {cid}"

    z_series: pd.Series | None = None
    if isinstance(profile_info, dict):
        z_table = profile_info.get("z_scores")
        if isinstance(z_table, pd.DataFrame):
            if cid in z_table.index:
                z_series = z_table.loc[cid]
            elif "cluster_id" in z_table.columns:
                found = z_table[z_table["cluster_id"] == cid]
                if not found.empty:
                    z_series = found.iloc[0].drop(labels=["cluster_id"], errors="ignore")
    if z_series is None:
        z_cols = [col for col in row.index if str(col).endswith("_zscore")]
        z_series = pd.Series({str(col)[:-7]: float(row[col]) for col in z_cols})

    top_features: list[dict[str, Any]] = []
    highlights: list[str] = []
    if cid != -1 and z_series is not None:
        pairs = [(str(feature), float(value)) for feature, value in z_series.items()]
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        for feature, z_value in pairs:
            if abs(z_value) < z_threshold or len(top_features) >= top_k:
                continue
            display = feature_labels.get(feature, feature)
            mean_value = row.get(feature, row.get(f"{feature}_mean", np.nan))
            direction = "cao hơn" if z_value > 0 else "thấp hơn"
            top_features.append({
                "feature": feature,
                "label": display,
                "mean": float(mean_value) if pd.notna(mean_value) else None,
                "zscore": z_value,
                "direction": direction,
            })
            if pd.notna(mean_value):
                highlights.append(
                    f"**{display}**: trung bình `{float(mean_value):.2f}` "
                    f"({direction} mức chung {abs(z_value):.2f}σ)"
                )
            else:
                highlights.append(f"**{display}**: {direction} mức chung {abs(z_value):.2f}σ")

    if cid == -1:
        description = (
            "Nhóm khách hàng ngoại lai có hành vi khác biệt so với các cụm chính."
        )
        highlights = ["Các khách hàng này được DBSCAN xem là điểm nhiễu (noise)."]
    elif top_features:
        description = (
            f"{name} gồm {size:,} khách hàng ({ratio:.1%}), nổi bật ở "
            + ", ".join(item["label"] for item in top_features)
            + "."
        )
    else:
        description = (
            f"{name} gồm {size:,} khách hàng ({ratio:.1%}); "
            "các đặc trưng tương đối gần mức trung bình toàn bộ."
        )
        highlights = ["Chưa có đặc trưng nào lệch đủ lớn so với mức chung."]

    return {
        "cluster_id": cid,
        "name": name,
        "description": description,
        "top_features": top_features,
        # Alias tương thích giao diện/phiên bản cũ.
        "label": name,
        "title": f"{name} ({size:,} khách - {ratio:.1%})",
        "highlights": highlights,
        "summary": description,
    }
