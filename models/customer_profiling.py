"""models/customer_profiling.py - Phân tích đặc trưng từng cụm và sinh mô tả chân dung khách hàng."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_cluster_profiles(
    df: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Tính toán hồ sơ từng cụm và trả về dict chuẩn gồm bảng profile, cluster_ids và đặc trưng nổi bật."""
    labels_arr = np.asarray(labels)

    if isinstance(df, pd.DataFrame):
        if feature_columns:
            valid_cols = [c for c in feature_columns if c in df.columns]
            if not valid_cols:
                valid_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            valid_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[valid_cols].reset_index(drop=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        arr = np.asarray(df, dtype=float)
        cols = (
            feature_columns
            if (feature_columns and len(feature_columns) == arr.shape[1])
            else [f"feature_{i}" for i in range(arr.shape[1])]
        )
        data = pd.DataFrame(arr, columns=cols)

    if len(data) != len(labels_arr):
        raise ValueError(f"Số dòng dữ liệu ({len(data)}) không khớp số nhãn ({len(labels_arr)}).")

    global_mean = data.mean()
    global_std = data.std().replace(0, 1.0)

    rows: list[dict[str, Any]] = []
    cluster_ids = sorted(int(c) for c in np.unique(labels_arr))
    cluster_sizes: dict[int, int] = {}
    cluster_percentages: dict[int, float] = {}
    top_features_by_cluster: dict[int, list[dict[str, Any]]] = {}

    for cid in cluster_ids:
        mask = labels_arr == cid
        sub = data.loc[mask]
        c_mean = sub.mean()
        z_diff = (c_mean - global_mean) / global_std

        size_val = int(mask.sum())
        pct_val = float(mask.mean() * 100.0)
        cluster_sizes[cid] = size_val
        cluster_percentages[cid] = pct_val

        row: dict[str, Any] = {
            "cluster_id": cid,
            "cluster_name": "Noise / Outliers" if cid == -1 else f"Cluster {cid}",
            "size": size_val,
            "percentage": pct_val,
        }
        feat_diffs: list[dict[str, Any]] = []
        for col in data.columns:
            m_val = float(c_mean[col])
            z_val = float(z_diff[col])
            row[f"{col}_mean"] = m_val
            row[f"{col}_zscore"] = z_val
            feat_diffs.append({
                "feature": col,
                "cluster_mean": m_val,
                "global_mean": float(global_mean[col]),
                "zscore": z_val,
            })

        feat_diffs.sort(key=lambda x: abs(x["zscore"]), reverse=True)
        top_features_by_cluster[cid] = feat_diffs[:5]
        rows.append(row)

    profile_df = pd.DataFrame(rows)
    return {
        "profile_df": profile_df,
        "summary_table": profile_df,
        "cluster_ids": cluster_ids,
        "n_clusters": len([c for c in cluster_ids if c != -1]),
        "cluster_sizes": cluster_sizes,
        "cluster_percentages": cluster_percentages,
        "feature_columns": data.columns.tolist(),
        "global_means": {col: float(global_mean[col]) for col in data.columns},
        "top_features": top_features_by_cluster,
    }


def generate_cluster_description(
    cluster_id: int | dict[str, Any] | pd.DataFrame,
    profile_info: dict[str, Any] | pd.DataFrame | None = None,
    top_k: int = 3,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Sinh mô tả định tính cho 1 cụm (khi truyền cluster_id, profile_info) hoặc toàn bộ các cụm."""
    # Hỗ trợ gọi dạng generate_cluster_description(cluster_id, profile_info) như check_backend.py
    if isinstance(cluster_id, (int, np.integer)) and profile_info is not None:
        cid = int(cluster_id)
        prof_df = (
            profile_info["profile_df"]
            if isinstance(profile_info, dict)
            else profile_info
        )
        matched = prof_df[prof_df["cluster_id"] == cid]
        if matched.empty:
            raise ValueError(f"Không tìm thấy cluster_id={cid} trong hồ sơ cụm.")
        return _describe_single_row(matched.iloc[0], top_k=top_k)

    # Hỗ trợ gọi dạng generate_cluster_description(profile_info) cho tất cả cụm
    src = cluster_id if profile_info is None else profile_info
    prof_df = src["profile_df"] if isinstance(src, dict) else src
    return [_describe_single_row(row, top_k=top_k) for _, row in prof_df.iterrows()]


def _describe_single_row(row: pd.Series, top_k: int = 3) -> dict[str, Any]:
    cid = int(row["cluster_id"])
    size = int(row["size"])
    pct = float(row["percentage"])
    z_cols = [c for c in row.index if str(c).endswith("_zscore")]

    if cid == -1:
        return {
            "cluster_id": cid,
            "label": "Noise / Outliers",
            "title": f"Noise / Ngoại lai ({size} khách hàng - {pct:.1f}%)",
            "highlights": ["Nhóm khách hàng có hành vi dị biệt, không thuộc cụm mật độ cao nào."],
            "description": "Nhóm khách hàng ngoại lai có hành vi khác biệt rõ rệt so với các nhóm chính.",
            "summary": "Cần kiểm tra riêng lẻ (khách siêu VIP đột biến hoặc dữ liệu bất thường).",
        }

    z_pairs = [(str(c)[:-7], float(row[c])) for c in z_cols]
    z_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_feats = [p for p in z_pairs if abs(p[1]) >= 0.25][:top_k]

    highlights: list[str] = []
    for feat_name, z_val in top_feats:
        mean_val = row.get(f"{feat_name}_mean", np.nan)
        direction = "cao hơn" if z_val > 0 else "thấp hơn"
        highlights.append(
            f"**{feat_name}**: trung bình `{mean_val:.2f}` ({direction} mức chung {abs(z_val):.2f}σ)"
        )

    if not highlights:
        highlights.append("Các chỉ số nằm ở mức trung bình ổn định so với toàn tập khách hàng.")

    summary = (
        f"Cụm {cid} chiếm **{pct:.1f}%** ({size} khách hàng), nổi bật với "
        + ", ".join(f"{f} ({'cao' if z > 0 else 'thấp'})" for f, z in top_feats)
        if top_feats
        else f"Cụm {cid} chiếm **{pct:.1f}%** ({size} khách hàng) với hành vi tiêu dùng trung tính."
    )

    return {
        "cluster_id": cid,
        "label": f"Cluster {cid}",
        "title": f"Cluster {cid} ({size} khách hàng - {pct:.1f}%)",
        "highlights": highlights,
        "description": summary,
        "summary": summary,
    }