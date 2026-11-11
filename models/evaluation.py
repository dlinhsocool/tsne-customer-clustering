"""
models/evaluation.py
====================
Người phụ trách: Duy Lĩnh (đánh giá)

NHIỆM VỤ FILE: Silhouette, Davies-Bouldin, ARI và bảng so sánh các thuật toán.

QUY TẮC "KHÔNG TÍNH CHỈ SỐ KHÔNG HỢP LỆ":
  * Điểm noise (-1) bị LOẠI khỏi tính Silhouette và Davies-Bouldin.
  * Cần ít nhất 2 cụm (không tính noise) và nhiều điểm hơn số cụm. Không đủ => trả None.
  * ARI trả None nếu labels_true là None. ARI tính trên TẤT CẢ dòng (noise coi như một nhóm riêng).
  * Dùng hàm is_computable() bên dưới để quyết định, đừng tự viết lại điều kiện.

QUY ĐỊNH CHUNG: xem đầu file models/data_processing.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.constants import NOISE_LABEL


# ---------------------------------------------------------------------------
# ĐÃ VIẾT SẴN - DÙNG, KHÔNG SỬA
# ---------------------------------------------------------------------------
def get_valid_mask(labels: np.ndarray) -> np.ndarray:
    """Mask các dòng không phải noise. Dùng để loại noise trước khi tính chỉ số."""
    return np.asarray(labels) != NOISE_LABEL


def is_computable(labels: np.ndarray) -> tuple[bool, str | None]:
    """Có tính được Silhouette / Davies-Bouldin không?

    Returns:
        (True, None) nếu tính được; (False, "lý do tiếng Việt") nếu không.
    """
    labels = np.asarray(labels)
    mask = get_valid_mask(labels)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return False, "Toàn bộ điểm là noise nên không tính được chỉ số."
    k = len(np.unique(labels[mask]))
    if k < 2:
        return False, f"Chỉ có {k} cụm (cần ít nhất 2) nên không tính được chỉ số."
    if n_valid <= k:
        return False, "Số điểm không đủ nhiều hơn số cụm."
    return True, None


# ---------------------------------------------------------------------------
# DUY LĨNH VIẾT THÂN HÀM
# ---------------------------------------------------------------------------
def calculate_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 10000,
    random_state: int = 42,
) -> float | None:
    """Silhouette Score trung bình (loại noise).

    Args:
        X (np.ndarray): (n, d) CÙNG không gian mà thuật toán đã chạy (X_scaled hoặc X_pca).
        labels (np.ndarray): (n,) nhãn cụm, noise = -1.
        sample_size (int | None): tính trên mẫu cho nhanh khi n lớn. None = dùng hết. Mặc định 10000.
        random_state (int): seed lấy mẫu. Mặc định 42.

    Returns:
        float trong [-1, 1], hoặc None nếu is_computable(labels) báo không tính được.
    """
    raise NotImplementedError("TODO (Duy Lĩnh): sklearn.metrics.silhouette_score trên dòng không noise")


def calculate_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Davies-Bouldin Index (loại noise). Càng NHỎ càng tốt.

    Returns:
        float >= 0, hoặc None nếu không tính được.
    """
    raise NotImplementedError("TODO (Duy Lĩnh): sklearn.metrics.davies_bouldin_score trên dòng không noise")


def calculate_ari(labels_true: np.ndarray | None, labels_pred: np.ndarray) -> float | None:
    """Adjusted Rand Index so với nhãn tham chiếu.

    Args:
        labels_true (np.ndarray | None): nhãn thật (n,) hoặc None nếu không có.
        labels_pred (np.ndarray): nhãn do thuật toán tìm ra (n,).

    Returns:
        float (1 = khớp hoàn toàn, ~0 = ngẫu nhiên), hoặc None nếu labels_true là None.
    """
    raise NotImplementedError("TODO (Duy Lĩnh): sklearn.metrics.adjusted_rand_score")


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    labels_true: np.ndarray | None = None,
    runtime_seconds: float | None = None,
) -> dict:
    """Gom các chỉ số của MỘT kết quả phân cụm.

    Returns:
        dict (đủ key, không đổi tên):
            "silhouette_score": float | None
            "davies_bouldin_score": float | None
            "ari_score": float | None
            "n_clusters": int              - không tính noise
            "n_noise": int
            "noise_ratio": float           - n_noise / n, trong [0, 1]
            "runtime_seconds": float | None
            "note": str | None             - lý do nếu có chỉ số None (lấy từ is_computable)

    Ví dụ gọi:
        m = interface.evaluate_clustering(X_scaled, res["labels"], labels_true=None, runtime_seconds=res["runtime_seconds"])
    """
    raise NotImplementedError("TODO (Duy Lĩnh): gọi 3 hàm calculate_* và đóng gói")


def compare_models(
    results: dict[str, dict],
    X_scaled: np.ndarray,
    X_pca: np.ndarray | None = None,
    labels_true: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bảng so sánh nhiều thuật toán.

    Args:
        results (dict): {"kmeans": ClusteringResult, "gmm": ..., "dbscan": ...} (có thể thiếu).
        X_scaled (np.ndarray): (n, d).
        X_pca (np.ndarray | None): (n, k). Với mỗi kết quả, chọn X theo result["feature_space"]
            ("scaled" -> X_scaled, "pca" -> X_pca).
        labels_true (np.ndarray | None): nhãn tham chiếu cho ARI.

    Returns:
        pd.DataFrame, MỖI THUẬT TOÁN MỘT DÒNG, đúng các cột:
            "algorithm" (str: "kmeans"|"gmm"|"dbscan"), "n_clusters", "n_noise",
            "silhouette_score", "davies_bouldin_score", "ari_score", "runtime_seconds"
        Giá trị không tính được là None/NaN (không xóa cột, không xóa dòng).
    """
    raise NotImplementedError("TODO (Duy Lĩnh): lặp results, gọi evaluate_clustering, ghép DataFrame")
