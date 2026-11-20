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
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

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

    def calculate_silhouette_score(
            X: np.ndarray,
            labels: np.ndarray,
            sample_size: int | None = 10000,
            random_state: int = 42,
    ) -> float | None:
        ok, _ = is_computable(labels)
        if not ok:
            return None
        X, labels = np.asarray(X), np.asarray(labels)
        mask = get_valid_mask(labels)
        Xv, lv = X[mask], labels[mask]
        try:
            if sample_size is not None and len(lv) > sample_size:
                return float(silhouette_score(Xv, lv, sample_size=sample_size,
                                              random_state=random_state))
            return float(silhouette_score(Xv, lv))
        except ValueError:  # mẫu con chỉ rơi vào 1 cụm
            return None


def calculate_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Davies-Bouldin Index (loại noise). Càng NHỎ càng tốt.

    Returns:
        float >= 0, hoặc None nếu không tính được.
    """
    ok, _ = is_computable(labels)
    if not ok:
        return None
    X, labels = np.asarray(X), np.asarray(labels)
    mask = get_valid_mask(labels)
    return float(davies_bouldin_score(X[mask], labels[mask]))


def calculate_ari(labels_true: np.ndarray | None, labels_pred: np.ndarray) -> float | None:
    """Adjusted Rand Index so với nhãn tham chiếu.

    Args:
        labels_true (np.ndarray | None): nhãn thật (n,) hoặc None nếu không có.
        labels_pred (np.ndarray): nhãn do thuật toán tìm ra (n,).

    Returns:
        float (1 = khớp hoàn toàn, ~0 = ngẫu nhiên), hoặc None nếu labels_true là None.
    """

    def calculate_ari(labels_true, labels_pred):
        if labels_true is None:
            return None
        return float(adjusted_rand_score(np.asarray(labels_true), np.asarray(labels_pred)))


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
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    mask = get_valid_mask(labels)
    n_noise = int((~mask).sum())
    n_clusters = int(len(np.unique(labels[mask])))
    _, note = is_computable(labels)
    return {
        "silhouette_score": calculate_silhouette_score(X, labels),
        "davies_bouldin_score": calculate_davies_bouldin_score(X, labels),
        "ari_score": calculate_ari(labels_true, labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / n) if n > 0 else 0.0,
        "runtime_seconds": runtime_seconds,
        "note": note,
    }


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
    columns = ["algorithm", "n_clusters", "n_noise", "silhouette_score",
               "davies_bouldin_score", "ari_score", "runtime_seconds"]
    rows = []
    for algo, res in results.items():
        if res.get("feature_space") == "pca":
            if X_pca is None:
                raise ValueError(f"'{algo}' chạy trên PCA nhưng X_pca = None")
            X = X_pca
        else:
            X = X_scaled
        m = evaluate_clustering(X, res["labels"], labels_true, res.get("runtime_seconds"))
        rows.append({"algorithm": algo, **{c: m[c] for c in columns[1:]}})
    return pd.DataFrame(rows, columns=columns)  # results rỗng vẫn giữ đủ cột
