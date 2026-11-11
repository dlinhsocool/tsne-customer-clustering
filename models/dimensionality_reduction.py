"""
models/dimensionality_reduction.py
==================================
Người phụ trách: Huỳnh Lân (toán & cài đặt t-SNE)

NHIỆM VỤ FILE: PCA và t-SNE.

PHÂN BIỆT ĐẦU VÀO (rất dễ nhầm):
    * run_pca  : nhận X_scaled (n, d)  -> trả X_pca (n, k)     [dùng để TÍNH TOÁN, k <= d]
    * run_tsne : nhận ma trận (n, m)   -> trả embedding (n, 2|3) [dùng để NHÌN]
                 ma trận vào nên là X_pca (khuyến nghị) hoặc X_scaled.
    * Phân cụm KHÔNG chạy trên embedding của t-SNE (xem models/clustering.py).

QUY ĐỊNH CHUNG: xem đầu file models/data_processing.py (giữ tên hàm, tham số, key; thứ tự dòng
không đổi; không print/streamlit; dùng random_state).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_pca(
    X_scaled: np.ndarray,
    n_components: int | float = 0.95,
    random_state: int = 42,
) -> dict:
    """PCA trên dữ liệu đã chuẩn hóa.

    Args:
        X_scaled (np.ndarray): (n, d) đã StandardScaler.
        n_components (int | float): int >= 1 = số thành phần; float trong (0, 1) = giữ tối thiểu
            tỷ lệ phương sai đó (ví dụ 0.95 = giữ 95%). Mặc định 0.95.
        random_state (int): mặc định 42.

    Returns:
        dict:
            "pca_model": sklearn PCA đã fit      - phải có .transform() để chiếu khách mới
            "X_pca": np.ndarray (n, k)
            "n_components": int                   - k thực tế sau khi chọn
            "explained_variance_ratio": np.ndarray (k,)
            "cumulative_variance": np.ndarray (k,)
            "components": np.ndarray (k, d)       - ma trận vector riêng Vk (mỗi hàng một thành phần)

    Ví dụ gọi:
        pca = interface.run_pca(sc["X_scaled"], n_components=0.95)
        pca["X_pca"].shape   # (n, k)
    """
    raise NotImplementedError("TODO (Huỳnh Lân): sklearn.decomposition.PCA")


def get_explained_variance(pca_model: Any) -> pd.DataFrame:
    """Bảng phương sai giải thích của từng thành phần (để vẽ scree plot).

    Args:
        pca_model: đối tượng PCA đã fit (lấy từ run_pca()["pca_model"]).

    Returns:
        pd.DataFrame, mỗi dòng một thành phần, ĐÚNG 3 cột:
            "component": int (1, 2, 3, ...)
            "explained_variance_ratio": float
            "cumulative_variance": float

    Ví dụ gọi:
        var_df = interface.get_explained_variance(pca["pca_model"])
    """
    raise NotImplementedError("TODO (Huỳnh Lân): đọc explained_variance_ratio_ + cumsum")


def run_tsne(
    X_input: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = 200.0,
    n_iter: int = 1000,
    random_state: int = 42,
    use_opentsne: bool = False,
) -> dict:
    """t-SNE để trực quan hóa.

    Args:
        X_input (np.ndarray): (n, m) - X_pca (khuyến nghị) hoặc X_scaled.
        n_components (int): 2 hoặc 3. Mặc định 2.
        perplexity (float): 5-50; phải nhỏ hơn n. Mặc định 30.0.
        learning_rate (float | "auto"): mặc định 200.0.
        n_iter (int): số vòng lặp tối ưu. Mặc định 1000. (sklearn mới gọi là max_iter -> tự ánh xạ.)
        random_state (int): mặc định 42.
        use_opentsne (bool): True = dùng openTSNE để sau này có .transform() cho điểm mới.

    Returns:
        dict:
            "embedding": np.ndarray (n, n_components)   - CÙNG THỨ TỰ DÒNG với X_input
            "n_components": int
            "perplexity": float
            "learning_rate": float | str
            "n_iter": int
            "kl_divergence": float | None               - None nếu thư viện không cung cấp
            "tsne_model": object | None                 - openTSNE embedding (có transform) hoặc None
            "backend": str                              - "sklearn" | "opentsne"

    Ví dụ gọi:
        tsne = interface.run_tsne(pca["X_pca"], n_components=2, perplexity=30, random_state=42)
        tsne["embedding"]   # (n, 2)
    """
    raise NotImplementedError("TODO (Huỳnh Lân): sklearn.manifold.TSNE hoặc tự cài đặt theo bài báo 2008")
