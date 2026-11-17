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

import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

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
    from .data_processing import scale_data

    if isinstance(new_data, dict):
        row = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.DataFrame):
        row = new_data
    else:
        raise ValueError("Dữ liệu khách mới phải là dict hoặc DataFrame.")
    if len(row) != 1:
        raise ValueError("Hãy cung cấp đúng một dòng dữ liệu khách hàng mới.")
    if scaler is None:
        raise ValueError("Cần scaler đã fit để xử lý khách hàng mới.")

    columns = preprocess_config["feature_columns"]
    fill_values = preprocess_config["fill_values"]
    raw = row.reindex(columns=columns).reset_index(drop=True)
    warnings = []
    for column in columns:
        if pd.isna(raw.at[0, column]):
            # Cấu hình cũ dùng 0 khi thiếu giá trị điền đã lưu.
            value = float(fill_values.get(column, 0.0))
            raw[column] = value
            warnings.append(f"Thiếu {column}, đã điền giá trị {value:.3f}.")

    try:
        raw = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Đặc trưng của khách hàng mới phải là giá trị số.") from exc
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Đặc trưng của khách hàng mới phải là số hữu hạn.")

    feature_ranges = preprocess_config["feature_ranges"]
    out_of_range_features = [
        column for column in columns
        if not feature_ranges[column][0] <= raw.at[0, column] <= feature_ranges[column][1]
    ]

    features = raw.copy()
    for column in preprocess_config["log_transform_columns"]:
        features[column] = np.log1p(features[column].clip(lower=0))
    X_scaled = scale_data(features, scaler=scaler)["X_scaled"]
    try:
        X_pca = pca_model.transform(X_scaled) if pca_model is not None else None
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Không thể chiếu khách hàng mới. Hãy kiểm tra PCA đã fit.") from exc

    return {
        "X_new_raw": raw,
        "X_new_scaled": X_scaled,
        "X_new_pca": X_pca,
        "out_of_range_features": out_of_range_features,
        "warnings": warnings,
    }


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
    tsne_result: dict[str, Any],
    X_reference: np.ndarray,
    method: str = "knn",
    k: int = 5,
) -> dict[str, Any]:
    """
    Ước lượng tọa độ của một khách hàng mới trên bản đồ nhúng t-SNE hiện có.

    Nguyên lý toán học:
    Vì thuật toán t-SNE gốc là ánh xạ phi tham số (non-parametric) không học hàm chiếu tường minh,
    khi có điểm dữ liệu mới x_new ở không gian đặc trưng gốc:
    - Với `method="knn"`: Sử dụng PyTorch tính khoảng cách Euclid ||x_new - x_i||_2 tới toàn bộ
      tập tham chiếu X_reference, chọn k láng giềng gần nhất và nội suy tọa độ nhúng bằng trung bình
      tọa độ t-SNE của k điểm đó.
    - Với `method="opentsne"`: Gọi phương thức `.transform()` của đối tượng openTSNE nếu mô hình
      được huấn luyện với backend openTSNE.

    Tham số:
        x_new: Vector đặc trưng của khách hàng mới, shape (n_features,) hoặc (1, n_features).
        tsne_result: Dictionary kết quả trả về từ hàm `run_tsne`.
        X_reference: Ma trận đặc trưng gốc (X_scaled hoặc X_pca) tương ứng với `tsne_result["embedding"]`.
        method: Phương pháp chiếu tọa độ ("knn" hoặc "opentsne"). Mặc định là "knn".
        k: Số lượng hàng xóm gần nhất dùng để nội suy khi method="knn". Mặc định là 5.

    Trả về:
        Dictionary gồm 5 keys:
        - "coordinates": Mảng tọa độ nhúng của khách hàng mới, shape (1, n_components).
        - "n_components": Số chiều không gian nhúng (int).
        - "method_used": Phương pháp đã sử dụng ("knn" hoặc "opentsne").
        - "neighbor_indices": Mảng chỉ số của k hàng xóm gần nhất (shape (k,)) hoặc None.
        - "warnings": Danh sách các cảnh báo trong quá trình ước lượng (list[str]).
    """
    warnings_list: list[str] = []

    # 1. Kiểm tra tính hợp lệ của dữ liệu đầu vào x_new và X_reference
    x_arr = np.asarray(x_new, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    elif x_arr.ndim == 2:
        if x_arr.shape[0] != 1:
            raise ValueError(
                f"x_new chỉ hỗ trợ 1 khách hàng mỗi lần gọi (shape (1, n_features)), nhận được {x_arr.shape}."
            )
    else:
        raise ValueError(f"x_new phải là mảng 1D hoặc 2D, nhận được {x_arr.ndim}D.")

    if np.isnan(x_arr).any() or np.isinf(x_arr).any():
        raise ValueError("x_new chứa giá trị NaN hoặc Inf.")

    X_ref_arr = np.asarray(X_reference, dtype=np.float64)
    if X_ref_arr.ndim != 2:
        raise ValueError(f"X_reference phải là mảng 2D, nhận được {X_ref_arr.ndim}D.")
    if X_ref_arr.shape[1] != x_arr.shape[1]:
        raise ValueError(
            f"Số chiều đặc trưng không khớp: x_new có {x_arr.shape[1]} đặc trưng, "
            f"trong khi X_reference có {X_ref_arr.shape[1]} đặc trưng."
        )

    if not isinstance(tsne_result, dict) or "embedding" not in tsne_result:
        raise ValueError("tsne_result phải là dictionary hợp lệ trả về từ hàm run_tsne.")

    embedding_ref = np.asarray(tsne_result["embedding"], dtype=np.float64)
    if embedding_ref.shape[0] != X_ref_arr.shape[0]:
        raise ValueError(
            f"Số lượng mẫu không khớp giữa X_reference ({X_ref_arr.shape[0]}) "
            f"và tsne_result['embedding'] ({embedding_ref.shape[0]})."
        )

    # 2. Kiểm tra tham số method và k
    method_clean = method.strip().lower()
    if method_clean not in {"knn", "opentsne"}:
        raise ValueError(f"Phương pháp '{method}' không được hỗ trợ. Chỉ chấp nhận 'knn' hoặc 'opentsne'.")

    if k <= 0:
        raise ValueError("Số lượng hàng xóm k phải lớn hơn 0.")

    # 3. Thực thi chiếu tọa độ khách hàng mới
    if method_clean == "knn":
        n_ref = X_ref_arr.shape[0]
        if k > n_ref:
            warnings_list.append(
                f"Giá trị k={k} lớn hơn tổng số mẫu tham chiếu ({n_ref}). Tự động điều chỉnh k={n_ref}."
            )
        k_valid = min(int(k), n_ref)

        # Chuyển sang PyTorch Tensor để tính khoảng cách Euclid song song
        X_ref_tensor = torch.tensor(X_ref_arr, dtype=torch.float64)
        x_new_tensor = torch.tensor(x_arr, dtype=torch.float64)
        emb_tensor = torch.tensor(embedding_ref, dtype=torch.float64)

        # Tính khoảng cách Euclid ||x_new - x_i||_2 và trích xuất k hàng xóm gần nhất
        distances = torch.cdist(x_new_tensor, X_ref_tensor, p=2).squeeze(0)
        neighbor_indices_tensor = torch.topk(distances, k=k_valid, largest=False).indices
        neighbor_indices = neighbor_indices_tensor.cpu().numpy()

        # Nội suy tọa độ nhúng bằng trung bình cộng tọa độ t-SNE của k hàng xóm gần nhất
        neighbor_embeddings = emb_tensor[neighbor_indices_tensor]
        coordinates = torch.mean(neighbor_embeddings, dim=0).cpu().numpy().reshape(1, -1)

    else:  # method_clean == "opentsne"
        tsne_model = tsne_result.get("tsne_model")
        if tsne_model is None or not hasattr(tsne_model, "transform"):
            raise ValueError(
                "Không thể dùng method='opentsne' vì tsne_result không chứa đối tượng openTSNE hợp lệ. "
                "Hãy chạy run_tsne(..., use_opentsne=True) hoặc chuyển sang method='knn'."
            )
        coordinates = np.asarray(tsne_model.transform(x_arr), dtype=np.float64).reshape(1, -1)
        neighbor_indices = None

    return {
        "coordinates": coordinates,
        "n_components": int(coordinates.shape[1]),
        "method_used": method_clean,
        "neighbor_indices": neighbor_indices,
        "warnings": warnings_list,
    }