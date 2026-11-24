"""
models/dimensionality_reduction.py
==================================
Người phụ trách: Huỳnh Lân (toán & cài đặt giảm chiều)

NHIỆM VỤ FILE: PCA, UMAP và t-SNE.
  * run_pca  : nhận X_scaled (n, d) -> trả X_pca (n, k)
  * run_umap : nhận X_scaled (n, d) -> trả embedding (n, k)
  * run_tsne : nhận ma trận (n, m) -> trả embedding (n, 2|3)
"""
from __future__ import annotations

import time
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
    """Chạy Principal Component Analysis (PCA) trên dữ liệu đã chuẩn hóa."""
    X_arr = _validate_2d_array(X_scaled, name="X_scaled")
    n_samples, n_features = X_arr.shape

    if isinstance(n_components, float) and not isinstance(n_components, bool):
        if not (0.0 < n_components <= 1.0):
            raise ValueError("n_components dạng tỷ lệ phương sai phải nằm trong (0, 1].")
        target_comp: int | float = float(n_components)
    else:
        k = int(n_components)
        max_k = min(n_samples, n_features)
        if k < 1 or k > max_k:
            raise ValueError(f"n_components phải từ 1 đến {max_k}, nhận được {k}.")
        target_comp = k

    t0 = time.perf_counter()
    model = PCA(n_components=target_comp, random_state=int(random_state))
    X_pca = np.asarray(model.fit_transform(X_arr), dtype=np.float64)
    runtime = time.perf_counter() - t0

    ratio = np.asarray(model.explained_variance_ratio_, dtype=np.float64)
    cum_var = np.cumsum(ratio)

    return {
        "pca_model": model,
        "X_pca": X_pca,
        "n_components": int(X_pca.shape[1]),
        "explained_variance_ratio": ratio,
        "cumulative_variance": cum_var,
        "components": np.asarray(model.components_, dtype=np.float64),
        "runtime_seconds": float(runtime),
    }


def get_explained_variance(pca_input: Any) -> pd.DataFrame:
    """Trích xuất bảng thống kê phương sai giải thích từ kết quả hoặc mô hình PCA."""
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


def run_umap(
    X_input: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> dict[str, Any]:
    """Chạy UMAP (Uniform Manifold Approximation and Projection)."""
    X_arr = _validate_2d_array(X_input, name="X_input")
    n_samples = X_arr.shape[0]
    k = int(n_components)
    if k < 1 or k >= n_samples:
        raise ValueError(f"n_components ({k}) phải >= 1 và < n_samples ({n_samples}).")

    eff_neighbors = min(int(n_neighbors), n_samples - 1)
    if eff_neighbors < 2:
        raise ValueError("Số mẫu quá ít để chạy UMAP (cần ít nhất 3 mẫu).")

    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "Thư viện umap-learn chưa được cài đặt. Hãy cài bằng `pip install umap-learn`."
        ) from exc

    params = {
        "n_components": k,
        "n_neighbors": eff_neighbors,
        "min_dist": float(min_dist),
        "metric": str(metric),
        "random_state": int(random_state),
    }

    t0 = time.perf_counter()
    model = UMAP(**params)
    embedding = np.asarray(model.fit_transform(X_arr), dtype=np.float64)
    runtime = time.perf_counter() - t0

    return {
        "model": model,
        "embedding": embedding,
        "n_components": int(embedding.shape[1]),
        "params": params,
        "runtime_seconds": float(runtime),
        "backend": "umap-learn",
    }


class CustomTSNE:
    """
    Cài đặt thuật toán t-SNE (t-Distributed Stochastic Neighbor Embedding) từ cơ sở lý thuyết
    toán học bằng PyTorch Tensor (tự động chọn CUDA nếu khả dụng).
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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _pairwise_squared_distances(self, X: torch.Tensor) -> torch.Tensor:
        sum_X = torch.sum(X ** 2, dim=1, keepdim=True)
        D = sum_X + sum_X.T - 2.0 * torch.matmul(X, X.T)
        return torch.clamp(D, min=0.0)

    def _compute_joint_probabilities(self, D: torch.Tensor) -> torch.Tensor:
        n = D.shape[0]
        P = torch.zeros((n, n), dtype=torch.float64, device=self.device)
        target_entropy = np.log2(self.perplexity)

        for i in range(n):
            beta_min = -float("inf")
            beta_max = float("inf")
            beta_i = 1.0

            mask = torch.ones(n, dtype=torch.bool, device=self.device)
            mask[i] = False
            D_i = D[i, mask]

            for _ in range(50):
                P_i_unnorm = torch.exp(-D_i * beta_i)
                sum_P = torch.sum(P_i_unnorm)

                if sum_P < 1e-12:
                    P_i = torch.full_like(D_i, 1.0 / (n - 1))
                else:
                    P_i = P_i_unnorm / sum_P

                H = -torch.sum(P_i * torch.log2(torch.clamp(P_i, min=1e-12))).item()
                H_diff = H - target_entropy

                if abs(H_diff) < 1e-5:
                    break

                if H_diff > 0:
                    beta_min = beta_i
                    beta_i = beta_i * 2.0 if beta_max == float("inf") else (beta_i + beta_max) / 2.0
                else:
                    beta_max = beta_i
                    beta_i = beta_i / 2.0 if beta_min == -float("inf") else (beta_i + beta_min) / 2.0

            P[i, mask] = P_i

        P = (P + P.T) / (2.0 * n)
        return torch.clamp(P, min=1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        n = X_tensor.shape[0]

        if isinstance(self.learning_rate, str) and self.learning_rate.lower() == "auto":
            lr = max(n / 12.0, 50.0)
        else:
            lr = float(self.learning_rate)

        D = self._pairwise_squared_distances(X_tensor)
        P = self._compute_joint_probabilities(D)

        exaggeration_iters = 250 if self.n_iter > 250 else max(1, self.n_iter // 2)
        P_opt = P * 4.0

        Y = torch.randn((n, self.n_components), dtype=torch.float64, device=self.device) * 1e-4
        dY = torch.zeros_like(Y)

        for it in range(self.n_iter):
            if it == exaggeration_iters:
                P_opt = P

            dist_Y = self._pairwise_squared_distances(Y)
            inv_dist = 1.0 / (1.0 + dist_Y)
            inv_dist.fill_diagonal_(0.0)
            Q = torch.clamp(inv_dist / torch.sum(inv_dist), min=1e-12)

            PQ_mul_inv = (P_opt - Q) * inv_dist
            row_sums = torch.sum(PQ_mul_inv, dim=1, keepdim=True)
            grad = 4.0 * (row_sums * Y - torch.matmul(PQ_mul_inv, Y))

            current_momentum = 0.5 if it < exaggeration_iters else self.momentum
            dY = current_momentum * dY - lr * grad
            Y = Y + dY
            Y = Y - torch.mean(Y, dim=0, keepdim=True)

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
    """Chạy thuật toán t-SNE để nhúng dữ liệu nhiều chiều xuống không gian 2D hoặc 3D."""
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

    t0 = time.perf_counter()
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
        runtime = time.perf_counter() - t0
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
            "device": "cpu",
            "runtime_seconds": float(runtime),
        }

    tsne_model = CustomTSNE(
        n_components=int(n_components),
        perplexity=float(perplexity),
        n_iter=int(n_iter),
        learning_rate=learning_rate,
        random_state=int(random_state),
    )
    embedding = tsne_model.fit_transform(X_arr)
    runtime = time.perf_counter() - t0

    return {
        "embedding": embedding,
        "n_components": int(n_components),
        "perplexity": float(perplexity),
        "learning_rate": learning_rate,
        "n_iter": int(n_iter),
        "kl_divergence": tsne_model.kl_divergence_,
        "tsne_model": None,
        "backend": "custom",
        "device": str(tsne_model.device),
        "runtime_seconds": float(runtime),
    }