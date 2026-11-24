"""
utils/contract.py
=================
Kiểm tra output của các hàm backend có đúng HỢP ĐỒNG v2.0 hay không.
File này KHÔNG import streamlit, dùng được cả trong check_backend.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.constants import (
    CONTRACT_COLUMNS,
    CONTRACT_KEYS,
    CONTRACT_TYPES,
    FEATURE_SPACES,
    NOISE_LABEL,
)

BENCHMARK_REQUIRED_COLS = [
    "space", "algorithm", "n_samples", "n_features",
    "n_clusters", "n_noise", "noise_ratio",
    "reduction_runtime", "clustering_runtime", "total_runtime",
    "silhouette_space", "davies_bouldin_space",
    "silhouette_original", "davies_bouldin_original",
    "ari_score", "note",
]


def check_contract(name: str, result: Any) -> list[str]:
    """Trả về danh sách lỗi (rỗng nếu hợp lệ)."""
    problems: list[str] = []

    if name in CONTRACT_KEYS:
        if not isinstance(result, dict):
            return [f"{name}: phải trả về dict, nhận được {type(result).__name__}"]
        missing = [k for k in CONTRACT_KEYS[name] if k not in result]
        if missing:
            problems.append(f"{name}: thiếu key {missing} (không được xóa key; không có dữ liệu thì để None)")
        problems += _extra_checks(name, result)

    elif name in CONTRACT_COLUMNS:
        if not isinstance(result, pd.DataFrame):
            return [f"{name}: phải trả về DataFrame, nhận được {type(result).__name__}"]
        missing = [c for c in CONTRACT_COLUMNS[name] if c not in result.columns]
        if missing:
            problems.append(f"{name}: DataFrame thiếu cột {missing}")

    elif name in CONTRACT_TYPES:
        if not isinstance(result, pd.DataFrame):
            problems.append(f"{name}: phải trả về DataFrame, nhận được {type(result).__name__}")

    return problems


def _extra_checks(name: str, r: dict) -> list[str]:
    """Các kiểm tra sâu hơn: kiểu dữ liệu, shape, thứ tự dòng, thời gian benchmark."""
    p: list[str] = []
    try:
        if name == "preprocess_data":
            n = len(r["df_clean"])
            if len(r["X"]) != n:
                p.append("preprocess_data: len(X) phải bằng len(df_clean)")
            if not r["df_clean"].index.equals(pd.RangeIndex(n)):
                p.append("preprocess_data: df_clean phải reset_index(drop=True) (index 0..n-1)")
            if list(r["X"].columns) != list(r["feature_columns"]):
                p.append("preprocess_data: cột của X phải đúng thứ tự feature_columns")

        elif name == "scale_data":
            X = np.asarray(r["X_scaled"])
            if X.ndim != 2:
                p.append("scale_data: X_scaled phải là ndarray 2 chiều (n, d)")

        elif name == "run_pca":
            X = np.asarray(r["X_pca"])
            if X.ndim != 2 or X.shape[1] != r["n_components"]:
                p.append("run_pca: X_pca phải có shape (n, n_components)")
            if r["runtime_seconds"] is not None and float(r["runtime_seconds"]) < 0:
                p.append("run_pca: runtime_seconds không được âm")

        elif name == "run_umap":
            E = np.asarray(r["embedding"])
            if E.ndim != 2 or E.shape[1] != r["n_components"]:
                p.append("run_umap: embedding phải có shape (n, n_components)")
            if r["runtime_seconds"] is not None and float(r["runtime_seconds"]) < 0:
                p.append("run_umap: runtime_seconds không được âm")

        elif name == "run_tsne":
            E = np.asarray(r["embedding"])
            if E.ndim != 2 or E.shape[1] != r["n_components"]:
                p.append("run_tsne: embedding phải có shape (n, n_components)")
            if r["runtime_seconds"] is not None and float(r["runtime_seconds"]) < 0:
                p.append("run_tsne: runtime_seconds không được âm")

        elif name == "run_clustering":
            lab = np.asarray(r["labels"])
            if lab.ndim != 1:
                p.append("run_clustering: labels phải là ndarray 1 chiều (n,)")
            n_clusters = len(set(lab.tolist()) - {NOISE_LABEL})
            if n_clusters != r["n_clusters"]:
                p.append(
                    f"run_clustering: n_clusters={r['n_clusters']} nhưng labels có {n_clusters} cụm (không tính noise)"
                )
            if int((lab == NOISE_LABEL).sum()) != r["n_noise"]:
                p.append("run_clustering: n_noise không khớp số nhãn -1 trong labels")
            if r["algorithm"] not in ("kmeans", "gmm", "dbscan"):
                p.append("run_clustering: algorithm phải là 'kmeans' | 'gmm' | 'dbscan'")
            if r["feature_space"] not in FEATURE_SPACES:
                p.append(
                    f"run_clustering: feature_space phải thuộc {list(FEATURE_SPACES)}, nhận '{r['feature_space']}'"
                )

        elif name == "benchmark_clustering_spaces":
            tbl = r["table"]
            if not isinstance(tbl, pd.DataFrame):
                p.append("benchmark_clustering_spaces: 'table' phải là DataFrame")
            else:
                missing_cols = [c for c in BENCHMARK_REQUIRED_COLS if c not in tbl.columns]
                if missing_cols:
                    p.append(f"benchmark_clustering_spaces: table thiếu cột {missing_cols}")
                else:
                    if (tbl["n_features"] <= 0).any():
                        p.append("benchmark_clustering_spaces: n_features phải > 0")
                    if not tbl["algorithm"].isin(["kmeans", "gmm", "dbscan"]).all():
                        p.append("benchmark_clustering_spaces: algorithm phải là kmeans | gmm | dbscan")
                    diff = np.abs(
                        tbl["total_runtime"] - (tbl["reduction_runtime"] + tbl["clustering_runtime"])
                    )
                    if (diff > 1e-3).any():
                        p.append("benchmark_clustering_spaces: total_runtime phải bằng reduction_runtime + clustering_runtime")

        elif name == "predict_customer_cluster":
            if r["method_used"] not in ("model_predict", "knn"):
                p.append("predict_customer_cluster: method_used phải là 'model_predict' hoặc 'knn'")

        elif name == "transform_new_customer":
            c = np.asarray(r["coordinates"])
            if c.ndim != 2 or c.shape[0] != 1:
                p.append("transform_new_customer: coordinates phải có shape (1, n_components)")
    except Exception as exc:
        p.append(f"{name}: không kiểm tra được cấu trúc output ({exc})")
    return p