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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def run_pca(X_scaled, n_components = 0.95, random_state: int = 42):
    pca=PCA(n_components=n_components, random_state=random_state)
    X_pca=pca.fit_transform(X_scaled)

    k=pca.n_components_

    return{
        "pca_model": pca,
        "X_pca": X_pca,
        "n_components": k,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "components": pca.components_
    }


def get_explained_variance(pca_model):
    ratio = pca_model.explained_variance_ratio_
    cum_var=np.cumsum(ratio)
    components = np.arange(1, len(ratio)+1)

    df = pd.DataFrame({
        "component": components,
        "explained_variance_ratio": ratio,
        "cumulative_variance": cum_var
    })
    return df

def run_tsne(X_input, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=42, use_opentsne=False):
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=random_state
    )

    embedding = tsne_model.fit_transform(X_input)
    kl_divergence= getattr(tsne_model, "kl_divergence_", None)
    return {
        "embedding": embedding,
        "n_components": n_components,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "n_iter": n_iter,
        "kl_divergence": kl_divergence,
        "tsne_model": None,
        "backend": "sklearn" 
    }

