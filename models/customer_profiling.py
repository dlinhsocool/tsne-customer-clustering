"""
models/customer_profiling.py
============================
Người phụ trách: Duy Lĩnh (phân tích đặc trưng cụm)

NHIỆM VỤ FILE: mô tả từng cụm khách hàng bằng số liệu và ngôn ngữ nghiệp vụ.

QUY TẮC:
  * Mô tả PHẢI sinh từ dữ liệu thật (so trung bình cụm với trung bình toàn bộ), KHÔNG gán sẵn
    "Cluster 0 = VIP". Cụm số mấy là VIP phụ thuộc kết quả chạy.
  * Profiling tính trên df_clean (THANG ĐO GỐC) để con số có nghĩa với người đọc.
  * Cụm noise (-1) nếu có thì vẫn có một dòng riêng.

QUY ĐỊNH CHUNG: xem đầu file models/data_processing.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from utils.constants import NOISE_LABEL

def calculate_cluster_profiles(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    include_noise: bool = True,
) -> dict:

    """Thống kê đặc trưng từng cụm.

    Args:
        df_clean (pd.DataFrame): dữ liệu sạch, thang đo gốc, CÙNG THỨ TỰ DÒNG với labels.
        labels (np.ndarray): (n,) nhãn cụm.
        feature_columns (list[str]): các cột cần thống kê.
        include_noise (bool): có tính cụm noise (-1) hay không. Mặc định True.

    Returns:
        dict:
            "profile_table": pd.DataFrame
                index = cluster_id (int, sắp tăng dần, -1 ở đầu nếu có)
                cột = ["size", "ratio"] + feature_columns
                size = số khách; ratio = size / tổng số dòng; các cột feature = TRUNG BÌNH của cụm
            "z_scores": pd.DataFrame
                cùng index, cột = feature_columns
                z = (trung bình cụm - trung bình toàn bộ) / độ lệch chuẩn toàn bộ
            "global_mean": pd.Series (index = feature_columns)
            "cluster_ids": list[int]
    """
    labels = np.asarray(labels).astype(int)
    n_total = len(df_clean)
    if len(labels) != n_total:
        raise ValueError(f"labels có {len(labels)} dòng nhưng df_clean có {n_total} dòng")

    # reset_index: labels khớp theo VỊ TRÍ dòng, không theo index của df_clean
    data = df_clean[feature_columns].reset_index(drop=True)
    global_mean = data.mean()
    global_std = data.std(ddof=0).replace(0, np.nan)  # cột hằng số -> z = 0

    keep = np.ones(n_total, dtype=bool) if include_noise else (labels != NOISE_LABEL)
    grouped = data[keep].groupby(labels[keep])
    cluster_mean = grouped.mean().sort_index()  # -1 tự nằm đầu nếu có
    sizes = grouped.size().sort_index()

    profile_table = cluster_mean.copy()
    profile_table.insert(0, "ratio", sizes / n_total)
    profile_table.insert(0, "size", sizes)  # thứ tự cột: size, ratio, features
    profile_table.index.name = "cluster_id"

    z_scores = ((cluster_mean - global_mean) / global_std).fillna(0.0)
    z_scores.index.name = "cluster_id"

    return {
        "profile_table": profile_table,
        "z_scores": z_scores,
        "global_mean": global_mean,
        "cluster_ids": [int(c) for c in profile_table.index],
    }


def generate_cluster_description(
    cluster_id: int,
    profiles: dict,
    feature_labels: dict[str, str] | None = None,
    top_n: int = 2,
    z_threshold: float = 0.5,
) -> dict:
    """Sinh mô tả nghiệp vụ cho MỘT cụm từ kết quả calculate_cluster_profiles().

    Cách làm gợi ý: lấy top_n đặc trưng có |z| lớn nhất và >= z_threshold; z > 0 là "cao",
    z < 0 là "thấp". Không đặc trưng nào vượt ngưỡng => "gần mức trung bình".

    Args:
        cluster_id (int): cụm cần mô tả (có thể là -1 = noise).
        profiles (dict): output của calculate_cluster_profiles().
        feature_labels (dict | None): {tên cột: tên hiển thị}. None = dùng tên cột.
        top_n (int): số đặc trưng nổi bật. Mặc định 2.
        z_threshold (float): ngưỡng |z| để coi là nổi bật. Mặc định 0.5.

    Returns:
        dict:
            "cluster_id": int
            "name": str          - tên ngắn, ví dụ "Nhóm 2: Spending cao, Cancellation Rate thấp"
            "description": str   - 1-3 câu tiếng Việt, có số liệu (kích thước, giá trị trung bình)
            "top_features": list[dict]
                mỗi phần tử: {"feature": str, "direction": "high" | "low", "z_score": float}
    """
    cluster_id = int(cluster_id)
    table, z_table = profiles["profile_table"], profiles["z_scores"]
    if cluster_id not in table.index:
        raise ValueError(f"Không có cụm {cluster_id} trong profiles")
    names = feature_labels or {}
    show = lambda f: names.get(f, f)

    # chọn đặc trưng nổi bật: |z| >= ngưỡng, lấy top_n theo |z| giảm dần
    z_row = z_table.loc[cluster_id]
    outstanding = z_row[z_row.abs() >= z_threshold]
    outstanding = outstanding.sort_values(
        key=lambda s: s.abs(), ascending=False, kind="mergesort"
    ).head(top_n)
    top_features = [
        {"feature": f, "direction": "high" if z > 0 else "low", "z_score": float(z)}
        for f, z in outstanding.items()
    ]
    word = {"high": "cao", "low": "thấp"}

    prefix = "Nhóm nhiễu (noise)" if cluster_id == NOISE_LABEL else f"Nhóm {cluster_id}"
    if top_features:
        tail = ", ".join(f"{show(t['feature'])} {word[t['direction']]}" for t in top_features)
    else:
        tail = "gần mức trung bình"
    name = f"{prefix}: {tail}"

    size = int(table.loc[cluster_id, "size"])
    ratio = float(table.loc[cluster_id, "ratio"])
    gmean = profiles["global_mean"]
    parts = [f"{prefix} gồm {size} khách hàng, chiếm {ratio:.1%} tổng số."]
    if cluster_id == NOISE_LABEL:
        parts.append("Đây là các điểm không thuộc cụm nào.")
    if top_features:
        details = [
            f"{show(t['feature'])} {word[t['direction']]} hơn mức chung "
            f"({table.loc[cluster_id, t['feature']]:.2f} so với {gmean[t['feature']]:.2f})"
            for t in top_features
        ]
        parts.append("Nổi bật: " + "; ".join(details) + ".")
    else:
        parts.append(f"Không đặc trưng nào lệch quá {z_threshold:g} độ lệch chuẩn "
                     "so với toàn bộ, nên cụm này gần mức trung bình.")

    return {
        "cluster_id": cluster_id,
        "name": name,
        "description": " ".join(parts),
        "top_features": top_features,
    }
