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
    raise NotImplementedError("TODO (Duy Lĩnh): groupby theo nhãn + z-score so với toàn bộ")


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
    raise NotImplementedError("TODO (Duy Lĩnh): chọn đặc trưng nổi bật theo z-score rồi viết mô tả")
