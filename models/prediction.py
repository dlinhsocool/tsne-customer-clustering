"""
models/prediction.py
====================
Người phụ trách: preprocess_new_customer -> Ngọc Đức
                 predict_customer_cluster -> Duy Lĩnh
                 transform_new_customer   -> Huỳnh Lân

NHIỆM VỤ FILE: xử lý KHÁCH HÀNG MỚI (out-of-sample).

HAI VIỆC KHÁC NHAU, KHÔNG ĐƯỢC GỘP:
  1) predict_customer_cluster : khách mới thuộc CỤM nào?  (dựa trên không gian phân cụm)
  2) transform_new_customer   : khách mới nằm ĐÂU trên bản đồ t-SNE? (chỉ để vẽ)
  Hai hàm có thể cho kết quả độc lập; không dùng vị trí t-SNE để quyết định cụm.

QUY TẮC:
  * Khách mới phải đi qua ĐÚNG pipeline cũ: điền thiếu theo fill_values, log1p đúng cột,
    scaler đã fit (KHÔNG fit lại), PCA đã fit (KHÔNG fit lại).
  * Không giả định mọi thuật toán có predict(): xem result["supports_predict"].
    DBSCAN không có -> dùng kNN + kiểm tra khoảng cách (xa quá eps thì là noise).
  * x_new luôn có shape (1, m). Không dùng chuyển vị, không trả scalar.

QUY ĐỊNH CHUNG: xem đầu file models/data_processing.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def preprocess_new_customer(
    new_data: dict | pd.DataFrame,
    preprocess_config: dict,
    scaler: Any,
    pca_model: Any | None = None,
) -> dict:
    """Đưa khách hàng mới qua đúng pipeline: thiếu -> log1p -> scale -> (PCA).

    Args:
        new_data (dict | pd.DataFrame): một khách, ví dụ {"spending": 500, "income": 60, ...}.
            Thiếu cột nào thì điền bằng preprocess_config["fill_values"] và thêm cảnh báo.
        preprocess_config (dict): lấy từ preprocess_data()["preprocess_config"].
        scaler: scaler đã fit từ scale_data()["scaler"] (KHÔNG fit lại).
        pca_model: PCA đã fit từ run_pca()["pca_model"], hoặc None nếu không dùng PCA.

    Returns:
        dict:
            "X_new_raw": pd.DataFrame (1, d)        - giá trị gốc theo thứ tự feature_columns
            "X_new_scaled": np.ndarray (1, d)
            "X_new_pca": np.ndarray (1, k) | None   - None nếu pca_model là None
            "out_of_range_features": list[str]      - cột có giá trị ngoài [min, max] của dữ liệu huấn luyện
                                                      (theo preprocess_config["feature_ranges"])
            "warnings": list[str]                   - tiếng Việt
    """
    raise NotImplementedError("TODO (Ngọc Đức): áp dụng lại fill_values, log1p, scaler, PCA")


def predict_customer_cluster(
    x_new: np.ndarray,
    clustering_result: dict,
    X_reference: np.ndarray,
    method: str = "auto",
    k: int = 5,
) -> dict:
    """Gán khách mới vào một cụm.

    Args:
        x_new (np.ndarray): (1, m) ở ĐÚNG không gian của clustering_result["feature_space"]
            (X_new_scaled nếu "scaled", X_new_pca nếu "pca").
        clustering_result (dict): output của run_clustering().
        X_reference (np.ndarray): (n, m) ma trận huấn luyện cùng không gian với x_new.
        method (str): "auto" | "model_predict" | "knn". Mặc định "auto":
            thuật toán có predict -> "model_predict", ngược lại (DBSCAN) -> "knn".
            Chọn "model_predict" cho thuật toán không hỗ trợ => đổi sang "knn" và ghi warnings.
        k (int): số hàng xóm cho kNN. Mặc định 5.

    Returns:
        dict:
            "cluster": int                      - nhãn cụm; -1 nếu là noise/ngoại lai (DBSCAN)
            "is_noise": bool                    - cluster == -1
            "method_used": str                  - "model_predict" | "knn"
            "confidence": float | None          - GMM: xác suất lớn nhất; kNN: tỷ lệ hàng xóm cùng cụm; else None
            "probabilities": np.ndarray | None  - (k_clusters,) nếu GMM, else None
            "neighbor_indices": np.ndarray      - (k,) chỉ số dòng của k hàng xóm gần nhất trong X_reference
            "neighbor_distances": np.ndarray    - (k,) khoảng cách Euclid tương ứng
            "mean_neighbor_distance": float
            "is_outlier": bool                  - khách xa mọi cụm bất thường (do bạn định nghĩa ngưỡng, ghi trong warnings)
            "warnings": list[str]

    Ví dụ gọi:
        out = interface.predict_customer_cluster(prep_new["X_new_scaled"], res, X_scaled, method="auto", k=5)
    """
    raise NotImplementedError("TODO (Duy Lĩnh): model.predict hoặc kNN; DBSCAN dùng kNN + eps")


def transform_new_customer(
    x_new: np.ndarray,
    tsne_result: dict,
    X_reference: np.ndarray,
    method: str = "knn",
    k: int = 5,
) -> dict:
    """Chiếu khách mới lên bản đồ t-SNE ĐÃ CÓ (chỉ để hiển thị).

    Args:
        x_new (np.ndarray): (1, m) ở cùng không gian với ma trận đã đưa vào run_tsne()
            (thường là X_new_pca). Dùng đúng không gian ấy, không đổi.
        tsne_result (dict): output của run_tsne() (cần "embedding", "tsne_model").
        X_reference (np.ndarray): (n, m) ma trận đã đưa vào run_tsne().
        method (str): "knn" | "opentsne". Mặc định "knn":
            knn      = trung bình tọa độ t-SNE của k hàng xóm gần nhất trong X_reference.
            opentsne = tsne_result["tsne_model"].transform(x_new); nếu tsne_model là None
                       thì quay về "knn" và ghi warnings.
        k (int): số hàng xóm cho kNN. Mặc định 5.

    Returns:
        dict:
            "coordinates": np.ndarray (1, n_components)
            "n_components": int
            "method_used": str               - "knn" | "opentsne"
            "neighbor_indices": np.ndarray | None
            "warnings": list[str]
    """
    raise NotImplementedError("TODO (Huỳnh Lân): kNN trên embedding hoặc openTSNE.transform")
