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
import torch
from sklearn.decomposition import PCA


def _validate_2d_array(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Kiểm tra tính hợp lệ của ma trận đầu vào 2 chiều."""
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float64)
    else:
        X = X.astype(np.float64, copy=False)

    if X.ndim != 2:
        raise ValueError(f"{name} phải là mảng 2 chiều (n_samples, n_features), nhận được {X.ndim}D.")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(f"{name} phải có ít nhất 2 mẫu và 1 đặc trưng, nhận được shape={X.shape}.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(f"{name} chứa giá trị NaN hoặc Inf. Cần làm sạch dữ liệu trước khi giảm chiều.")
    return X


def run_pca(
    X_scaled: np.ndarray,
    n_components: int | float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Thực hiện giảm chiều tuyến tính bằng PCA trên dữ liệu đã chuẩn hóa.

    Tham số:
        X_scaled: Ma trận dữ liệu đã chuẩn hóa kích thước (n_samples, n_features).
        n_components: Số chiều giữ lại (int) hoặc tỷ lệ phương sai tích lũy mục tiêu (float trong (0, 1)).
        random_state: Hạt giống ngẫu nhiên để đảm bảo tính tái lập.

    Trả về:
        Dictionary chứa kết quả biến đổi và thông tin phương sai giải thích.
    """
    X_arr = _validate_2d_array(X_scaled, name="X_scaled")

    pca_model = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca_model.fit_transform(X_arr)

    explained_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_ratio)

    return {
        "X_pca": X_pca,
        "pca_model": pca_model,
        "components": pca_model.components_,
        "n_components": int(pca_model.n_components_),
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance": cumulative_variance,
    }


def get_explained_variance(pca_input: Any) -> pd.DataFrame:
    """
    Trích xuất bảng thống kê phương sai giải thích từ kết quả hoặc mô hình PCA.
    """
    if isinstance(pca_input, dict):
        if "pca_model" in pca_input and pca_input["pca_model"] is not None:
            pca_obj = pca_input["pca_model"]
            ratio = np.asarray(pca_obj.explained_variance_ratio_, dtype=np.float64)
        elif "explained_variance_ratio" in pca_input:
            ratio = np.asarray(pca_input["explained_variance_ratio"], dtype=np.float64)
        else:
            raise ValueError("Dictionary đầu vào không chứa 'pca_model' hoặc 'explained_variance_ratio'.")
    elif hasattr(pca_input, "explained_variance_ratio_"):
        ratio = np.asarray(pca_input.explained_variance_ratio_, dtype=np.float64)
    else:
        raise TypeError("pca_input phải là dict từ run_pca hoặc mô hình PCA có thuộc tính explained_variance_ratio_.")

    cum_var = np.cumsum(ratio)
    n_comp = len(ratio)

    return pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(n_comp)],
        "explained_variance_ratio": ratio,
        "cumulative_variance": cum_var,
    })

class CustomTSNE:
    """
    Cài đặt thuật toán t-SNE (t-Distributed Stochastic Neighbor Embedding) từ cơ sở lý thuyết
    toán học bằng PyTorch Tensor.

    Các thành phần toán học chính:
    1. Ma trận khoảng cách Euclid bình phương ||x_i - x_j||^2 ở không gian gốc.
    2. Tìm kiếm nhị phân (Binary Search) độ chính xác beta_i = 1 / (2 * sigma_i^2) cho từng điểm
       để phân phối xác suất có điều kiện P_{j|i} đạt Shannon Entropy H(P_i) = log2(perplexity).
    3. Đối xứng hóa phân phối đồng thời P_ij = (P_{j|i} + P_{i|j}) / (2N).
    4. Mô hình hóa phân phối Student-t (1 bậc tự do) Q_ij ở không gian thấp chiều nhằm khắc phục
       vấn đề dồn cục (crowding problem).
    5. Tối ưu hàm mất mát Kullback-Leibler Divergence KL(P || Q) bằng Gradient Descent kết hợp
       Early Exaggeration và Momentum schedule.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        learning_rate: float | str = 200.0,
        momentum: float = 0.8,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.n_iter = int(n_iter)
        self.learning_rate = learning_rate
        self.momentum = float(momentum)
        self.random_state = int(random_state)
        self.kl_divergence_: float | None = None

        # Tự động ưu tiên GPU (CUDA) nếu khả dụng, ngược lại chạy trên CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _pairwise_squared_distances(self, X: torch.Tensor) -> torch.Tensor:
        """
        Tính ma trận khoảng cách Euclid bình phương toàn cặp điểm bằng khai triển ma trận:
        ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * (X @ X^T)
        """
        sum_X = torch.sum(X ** 2, dim=1, keepdim=True)
        D = sum_X + sum_X.T - 2.0 * torch.matmul(X, X.T)
        # Chặn dưới tại 0.0 để tránh sai số dấu phẩy động tạo ra giá trị âm rất nhỏ
        return torch.clamp(D, min=0.0)

    def _compute_joint_probabilities(self, D: torch.Tensor) -> torch.Tensor:
        """
        Tính ma trận xác suất đồng thời P ở không gian dữ liệu gốc:
        - Tìm beta_i = 1 / (2 * sigma_i^2) cho mỗi điểm i sao cho H(P_i) = log2(perplexity).
        - Xác suất có điều kiện: p_{j|i} = exp(-||x_i - x_j||^2 * beta_i) / sum_{k != i} exp(-||x_i - x_k||^2 * beta_i).
        - Xác suất đồng thời đối xứng: p_ij = (p_{j|i} + p_{i|j}) / (2 * N).
        """
        n = D.shape[0]
        P = torch.zeros((n, n), dtype=torch.float64, device=self.device)
        target_H = float(np.log2(self.perplexity))
        log2_e = float(np.log2(np.e))

        for i in range(n):
            beta_min = -float("inf")
            beta_max = float("inf")
            beta_i = 1.0

            # Loại bỏ chính điểm i (j != i) khi tính phân phối lân cận
            mask = torch.ones(n, dtype=torch.bool, device=self.device)
            mask[i] = False
            D_i = D[i, mask]

            # Binary search tối đa 50 vòng lặp để hội tụ beta_i theo sai số 1e-5
            for _ in range(50):
                P_i = torch.exp(-D_i * beta_i)
                sum_P = torch.sum(P_i) + 1e-12

                # Tính Shannon Entropy cơ số 2:
                # H(P_i) = log2(sum_P) + (beta_i * sum(D_i * P_i) / sum_P) * log2(e)
                H_i = torch.log2(sum_P) + (beta_i * torch.sum(D_i * P_i) / sum_P) * log2_e
                diff = float(H_i.item()) - target_H

                if abs(diff) < 1e-5:
                    break

                if diff > 0:
                    # Entropy đang lớn hơn mục tiêu -> cần thu hẹp phương sai sigma_i (tăng beta_i)
                    beta_min = beta_i
                    beta_i = beta_i * 2.0 if beta_max == float("inf") else (beta_i + beta_max) / 2.0
                else:
                    # Entropy đang nhỏ hơn mục tiêu -> cần mở rộng phương sai sigma_i (giảm beta_i)
                    beta_max = beta_i
                    beta_i = beta_i / 2.0 if beta_min == -float("inf") else (beta_i + beta_min) / 2.0

            P[i, mask] = P_i / sum_P

        # Đối xứng hóa ma trận P và chặn dưới 1e-12 để đảm bảo ổn định số học khi tính log
        P = (P + P.T) / (2.0 * n)
        return torch.clamp(P, min=1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Huấn luyện và ánh xạ ma trận X sang không gian n_components chiều.
        """
        # Cố định seed của PyTorch để tái lập kết quả khởi tạo ngẫu nhiên
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        n = X_tensor.shape[0]

        # Quy đổi learning_rate="auto" theo công thức chuẩn max(N / 12, 50)
        if isinstance(self.learning_rate, str) and self.learning_rate.lower() == "auto":
            lr = max(n / 12.0, 50.0)
        else:
            lr = float(self.learning_rate)

        # Bước 1: Tính ma trận xác suất đồng thời P ở không gian cao chiều
        D = self._pairwise_squared_distances(X_tensor)
        P = self._compute_joint_probabilities(D)

        # Early Exaggeration: Nhân P với hệ số 4.0 trong giai đoạn đầu để kéo các cụm tách xa nhau
        exaggeration_iters = 250 if self.n_iter > 250 else max(1, self.n_iter // 2)
        P_opt = P * 4.0

        # Bước 2: Khởi tạo tọa độ thấp chiều Y theo phân phối chuẩn N(0, 10^-4 * I)
        Y = torch.randn((n, self.n_components), dtype=torch.float64, device=self.device) * 1e-4
        dY = torch.zeros_like(Y)

        # Bước 3: Tối ưu hóa KL Divergence bằng Gradient Descent + Momentum
        for it in range(self.n_iter):
            if it == exaggeration_iters:
                # Kết thúc giai đoạn Early Exaggeration, đưa P về giá trị thực
                P_opt = P

            # Tính phân phối Student-t (1 bậc tự do) trong không gian thấp chiều:
            # inv_dist_ij = (1 + ||y_i - y_j||^2)^(-1)
            dist_Y = self._pairwise_squared_distances(Y)
            inv_dist = 1.0 / (1.0 + dist_Y)
            inv_dist.fill_diagonal_(0.0)
            Q = torch.clamp(inv_dist / torch.sum(inv_dist), min=1e-12)

            # Vector hóa công thức đạo hàm hàm mất mát KL(P || Q) theo tọa độ Y:
            # grad_i = 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)
            # Khai triển tổng hiệu thành phép nhân ma trận để tính song song trên toàn bộ N điểm
            PQ_mul_inv = (P_opt - Q) * inv_dist
            row_sums = torch.sum(PQ_mul_inv, dim=1, keepdim=True)
            grad = 4.0 * (row_sums * Y - torch.matmul(PQ_mul_inv, Y))

            # Cập nhật vận tốc dY và tọa độ Y theo lịch trình Momentum (0.5 -> self.momentum)
            current_momentum = 0.5 if it < exaggeration_iters else self.momentum
            dY = current_momentum * dY - lr * grad
            Y = Y + dY

            # Dịch chuyển tâm đám mây điểm về gốc tọa độ sau mỗi bước lặp
            Y = Y - torch.mean(Y, dim=0, keepdim=True)

        # Bước 4: Tính giá trị KL Divergence cuối cùng trên ma trận P gốc (không phóng đại)
        dist_Y = self._pairwise_squared_distances(Y)
        inv_dist = 1.0 / (1.0 + dist_Y)
        inv_dist.fill_diagonal_(0.0)
        Q_final = torch.clamp(inv_dist / torch.sum(inv_dist), min=1e-12)
        kl_loss = torch.sum(P * torch.log(P / Q_final))
        self.kl_divergence_ = float(kl_loss.item())

        return Y.detach().cpu().numpy()


def run_tsne(
    X_input: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = 200.0,
    n_iter: int = 1000,
    random_state: int = 42,
    use_opentsne: bool = False,
) -> dict[str, Any]:
    """
    Chạy thuật toán t-SNE để nhúng dữ liệu nhiều chiều xuống không gian 2D hoặc 3D.

    Tham số:
        X_input: Ma trận dữ liệu đầu vào (X_scaled hoặc X_pca) dạng (n_samples, n_features).
        n_components: Số chiều không gian đích (thường là 2 hoặc 3).
        perplexity: Tham số cân bằng lân cận cục bộ và toàn cục (bắt buộc < n_samples).
        learning_rate: Tốc độ học (số thực dương hoặc chuỗi "auto").
        n_iter: Số vòng lặp tối ưu hóa Gradient Descent.
        random_state: Hạt giống ngẫu nhiên để tái lập kết quả.
        use_opentsne: Nếu True, sử dụng thư viện openTSNE (hỗ trợ transform out-of-sample trực tiếp);
                      nếu False (mặc định), sử dụng CustomTSNE cài đặt bằng PyTorch.

    Trả về:
        Dictionary chuẩn gồm 8 keys:
        - "embedding": Ma trận tọa độ sau giảm chiều (n_samples, n_components).
        - "n_components": Số chiều đầu ra.
        - "perplexity": Giá trị perplexity đã dùng.
        - "learning_rate": Giá trị learning_rate đã truyền.
        - "n_iter": Số vòng lặp tối ưu hóa.
        - "kl_divergence": Giá trị hàm mất mát KL Divergence cuối cùng.
        - "tsne_model": Đối tượng mô hình (nếu dùng openTSNE) hoặc None.
        - "backend": Tên backend thực thi ("custom" hoặc "opentsne").
    """
    X_arr = _validate_2d_array(X_input, name="X_input")
    n_samples = X_arr.shape[0]

    if n_components < 1:
        raise ValueError("n_components phải lớn hơn hoặc bằng 1.")
    if n_iter < 1:
        raise ValueError("n_iter phải lớn hơn hoặc bằng 1.")
    if perplexity <= 0 or perplexity >= n_samples:
        raise ValueError(
            f"perplexity ({perplexity}) phải lớn hơn 0 và nhỏ hơn số mẫu n_samples ({n_samples})."
        )
    if isinstance(learning_rate, str):
        if learning_rate.lower() != "auto":
            raise ValueError("learning_rate dạng chuỗi chỉ chấp nhận giá trị 'auto'.")
    elif float(learning_rate) <= 0:
        raise ValueError("learning_rate phải là số dương.")

    if use_opentsne:
        try:
            from openTSNE import TSNE as OpenTSNE
        except ImportError as exc:
            raise ImportError(
                "Thư viện openTSNE chưa được cài đặt. Hãy cài bằng `pip install openTSNE` "
                "hoặc đặt use_opentsne=False để dùng CustomTSNE."
            ) from exc

        lr_val = "auto" if isinstance(learning_rate, str) else float(learning_rate)
        opentsne_model = OpenTSNE(
            n_components=int(n_components),
            perplexity=float(perplexity),
            learning_rate=lr_val,
            n_iter=int(n_iter),
            random_state=int(random_state),
        )
        embedding_obj = opentsne_model.fit(X_arr)
        embedding = np.asarray(embedding_obj, dtype=np.float64)
        kl_div = float(getattr(embedding_obj, "kl_divergence", np.nan))

        return {
            "embedding": embedding,
            "n_components": int(n_components),
            "perplexity": float(perplexity),
            "learning_rate": learning_rate,
            "n_iter": int(n_iter),
            "kl_divergence": kl_div,
            "tsne_model": embedding_obj,
            "backend": "opentsne",
        }

    # Mặc định: Thực thi CustomTSNE tự cài đặt bằng PyTorch
    tsne_model = CustomTSNE(
        n_components=int(n_components),
        perplexity=float(perplexity),
        n_iter=int(n_iter),
        learning_rate=learning_rate,
        random_state=int(random_state),
    )
    embedding = tsne_model.fit_transform(X_arr)

    return {
        "embedding": embedding,
        "n_components": int(n_components),
        "perplexity": float(perplexity),
        "learning_rate": learning_rate,
        "n_iter": int(n_iter),
        "kl_divergence": tsne_model.kl_divergence_,
        "tsne_model": None,
        "backend": "custom",
    }