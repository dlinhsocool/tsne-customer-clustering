"""
utils/interface.py
==================
LỚP TRUNG GIAN giữa giao diện (pages/) và backend (models/).
Phiên bản hợp đồng: 2.0
"""
from __future__ import annotations

import copy
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from models import clustering as m_clustering
from models import customer_profiling as m_profiling
from models import data_processing as m_data
from models import dimensionality_reduction as m_dr
from models import evaluation as m_eval
from models import prediction as m_pred
from utils import mock_backend as mock
from utils.constants import (
    ALGORITHMS,
    FEATURE_DISPLAY,
    NON_FEATURE_HINTS,
    RANDOM_STATE,
    SK,
    USE_MOCK_WHEN_NOT_IMPLEMENTED,
)
from utils.contract import check_contract


def _real_or_todo(module: Any, name: str) -> Callable:
    fn = getattr(module, name, None)
    if fn is not None:
        return fn

    def _todo(*a: Any, **k: Any) -> Any:
        raise NotImplementedError(f"{module.__name__}.{name} chưa được viết")
    return _todo


_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "load_dataset": (m_data.load_dataset, mock.load_dataset),
    "validate_dataset": (m_data.validate_dataset, mock.validate_dataset),
    "preprocess_data": (m_data.preprocess_data, mock.preprocess_data),
    "scale_data": (m_data.scale_data, mock.scale_data),
    "run_pca": (_real_or_todo(m_dr, "run_pca"), mock.run_pca),
    "get_explained_variance": (m_dr.get_explained_variance, mock.get_explained_variance),
    "run_umap": (_real_or_todo(m_dr, "run_umap"), mock.run_umap),
    "run_tsne": (m_dr.run_tsne, mock.run_tsne),
    "run_clustering": (m_clustering.run_clustering, mock.run_clustering),
    "evaluate_clustering": (m_eval.evaluate_clustering, mock.evaluate_clustering),
    "compare_models": (m_eval.compare_models, mock.compare_models),
    "compare_dimensionality_reduction_method": (
        _real_or_todo(m_eval, "compare_dimensionality_reduction_method"),
        mock.compare_dimensionality_reduction_method,
    ),
    "benchmark_clustering_spaces": (
        _real_or_todo(m_eval, "benchmark_clustering_spaces"),
        mock.benchmark_clustering_spaces,
    ),
    "calculate_cluster_profiles": (m_profiling.calculate_cluster_profiles, mock.calculate_cluster_profiles),
    "generate_cluster_description": (m_profiling.generate_cluster_description, mock.generate_cluster_description),
    "preprocess_new_customer": (m_pred.preprocess_new_customer, mock.preprocess_new_customer),
    "predict_customer_cluster": (m_pred.predict_customer_cluster, mock.predict_customer_cluster),
    "transform_new_customer": (m_pred.transform_new_customer, mock.transform_new_customer),
}


def _fail(title: str, exc: BaseException) -> None:
    st.error(f"{title}: {exc}")
    with st.expander("Chi tiết lỗi (gửi cho người phụ trách hàm này)"):
        st.code("".join(traceback.format_exception(exc)))
    st.stop()


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    """Gọi hàm thật; chưa viết thì dùng mock; kiểm tra hợp đồng; ghi trạng thái."""
    real, fake = _REGISTRY[name]
    used = "real"
    result = None
    error: BaseException | None = None
    try:
        result = real(*args, **kwargs)
    except NotImplementedError as exc:
        if USE_MOCK_WHEN_NOT_IMPLEMENTED:
            used = "mock"
            try:
                result = fake(*args, **kwargs)
            except Exception as exc2:  # noqa: BLE001
                error = exc2
        else:
            error = exc
    except Exception as exc:  # noqa: BLE001
        error = exc
    if error is not None:
        _fail(f"Hàm {name}() bị lỗi" if used == "real" else f"Hàm giả {name}() bị lỗi", error)

    problems = check_contract(name, result)
    if problems:
        st.error(f"Output của {name}() không đúng hợp đồng ({used}):")
        for p in problems:
            st.markdown(f"- {p}")
        st.stop()

    status = st.session_state.setdefault(SK.BACKEND_STATUS, {})
    status[name] = used
    return result


# ---------------------------------------------------------------------------
# 2. Session state
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    SK.RAW_DF: None,
    SK.IS_SAMPLE: False,
    SK.UPLOAD_SIG: None,
    SK.VALIDATION: None,
    SK.RAW_FEATURE_COLUMNS: None,
    SK.FEATURE_COLUMNS: None,
    SK.PREP: None,
    SK.SCALE: None,
    SK.PCA: None,
    SK.MAIN_TSNE_2D: None,
    SK.MAIN_TSNE_3D: None,
    SK.TSNE_INPUT_SPACE: "original",
    SK.CLUSTER_RESULTS: {},
    SK.ACTIVE_ALGO: None,
    SK.EVAL_TABLE: None,
    SK.PROFILES: {},
    SK.CLUSTER_NAMES: {},
    SK.NEW_CUSTOMER: None,
    # Benchmark state
    SK.BENCHMARK_SAMPLE: None,
    SK.BENCHMARK_REPRESENTATIONS: {},
    SK.BENCHMARK_REDUCTION_TABLE: None,
    SK.BENCHMARK_CLUSTER_TABLE: None,
    SK.BENCHMARK_CLUSTER_RESULTS: {},
    # System
    SK.RANDOM_STATE: RANDOM_STATE,
    SK.BACKEND_STATUS: {},
}

_RESET_GROUPS: dict[str, list[str]] = {
    "data": [
        SK.VALIDATION, SK.RAW_FEATURE_COLUMNS, SK.FEATURE_COLUMNS, SK.PREP, SK.SCALE,
        SK.PCA, SK.MAIN_TSNE_2D, SK.MAIN_TSNE_3D, SK.TSNE_INPUT_SPACE,
        SK.CLUSTER_RESULTS, SK.ACTIVE_ALGO, SK.EVAL_TABLE, SK.PROFILES,
        SK.CLUSTER_NAMES, SK.NEW_CUSTOMER,
        SK.BENCHMARK_SAMPLE, SK.BENCHMARK_REPRESENTATIONS,
        SK.BENCHMARK_REDUCTION_TABLE, SK.BENCHMARK_CLUSTER_TABLE, SK.BENCHMARK_CLUSTER_RESULTS,
    ],
    "prep": [
        SK.SCALE, SK.PCA, SK.MAIN_TSNE_2D, SK.MAIN_TSNE_3D, SK.TSNE_INPUT_SPACE,
        SK.CLUSTER_RESULTS, SK.ACTIVE_ALGO, SK.EVAL_TABLE, SK.PROFILES,
        SK.CLUSTER_NAMES, SK.NEW_CUSTOMER,
        SK.BENCHMARK_SAMPLE, SK.BENCHMARK_REPRESENTATIONS,
        SK.BENCHMARK_REDUCTION_TABLE, SK.BENCHMARK_CLUSTER_TABLE, SK.BENCHMARK_CLUSTER_RESULTS,
    ],
    "pca": [SK.MAIN_TSNE_2D, SK.MAIN_TSNE_3D, SK.NEW_CUSTOMER],
    "main_tsne": [SK.NEW_CUSTOMER],
    "tsne": [SK.NEW_CUSTOMER],
    "main_cluster": [SK.EVAL_TABLE, SK.NEW_CUSTOMER],
    "cluster": [SK.EVAL_TABLE, SK.NEW_CUSTOMER],
    "benchmark": [
        SK.BENCHMARK_SAMPLE, SK.BENCHMARK_REPRESENTATIONS,
        SK.BENCHMARK_REDUCTION_TABLE, SK.BENCHMARK_CLUSTER_TABLE, SK.BENCHMARK_CLUSTER_RESULTS,
    ],
}


def init_state() -> None:
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default)


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def put(key: str, value: Any) -> None:
    st.session_state[key] = value


def reset_from(step: str) -> None:
    """Xóa kết quả phía sau `step`."""
    for key in _RESET_GROUPS.get(step, []):
        st.session_state[key] = copy.deepcopy(_DEFAULTS[key])


def require(*keys: str, message: str) -> bool:
    """Trả False và hiện hướng dẫn nếu thiếu bước trước."""
    if any(get(k) is None for k in keys):
        st.info(message)
        return False
    return True


# ---------------------------------------------------------------------------
# 3. Hàm bọc backend
# ---------------------------------------------------------------------------
def load_dataset(*a: Any, **k: Any) -> pd.DataFrame:
    return _call("load_dataset", *a, **k)


def validate_dataset(*a: Any, **k: Any) -> dict:
    return _call("validate_dataset", *a, **k)


def preprocess_data(*a: Any, **k: Any) -> dict:
    return _call("preprocess_data", *a, **k)


def scale_data(*a: Any, **k: Any) -> dict:
    return _call("scale_data", *a, **k)


def run_pca(*a: Any, **k: Any) -> dict:
    return _call("run_pca", *a, **k)


def get_explained_variance(*a: Any, **k: Any) -> pd.DataFrame:
    return _call("get_explained_variance", *a, **k)


def run_umap(*a: Any, **k: Any) -> dict:
    return _call("run_umap", *a, **k)


def run_tsne(*a: Any, **k: Any) -> dict:
    return _call("run_tsne", *a, **k)


def run_clustering(*a: Any, **k: Any) -> dict:
    return _call("run_clustering", *a, **k)


def evaluate_clustering(*a: Any, **k: Any) -> dict:
    return _call("evaluate_clustering", *a, **k)


def compare_models(*a: Any, **k: Any) -> pd.DataFrame:
    return _call("compare_models", *a, **k)


def compare_dimensionality_reduction_method(*a: Any, **k: Any) -> pd.DataFrame:
    return _call("compare_dimensionality_reduction_method", *a, **k)


def benchmark_clustering_spaces(*a: Any, **k: Any) -> dict:
    return _call("benchmark_clustering_spaces", *a, **k)


def calculate_cluster_profiles(*a: Any, **k: Any) -> dict:
    return _call("calculate_cluster_profiles", *a, **k)


def generate_cluster_description(*a: Any, **k: Any) -> dict:
    return _call("generate_cluster_description", *a, **k)


def preprocess_new_customer(*a: Any, **k: Any) -> dict:
    return _call("preprocess_new_customer", *a, **k)


def predict_customer_cluster(*a: Any, **k: Any) -> dict:
    return _call("predict_customer_cluster", *a, **k)


def transform_new_customer(*a: Any, **k: Any) -> dict:
    return _call("transform_new_customer", *a, **k)


def load_sample_dataset() -> pd.DataFrame:
    """Dữ liệu mẫu GIẢ LẬP để thử giao diện (không phải dữ liệu của đồ án)."""
    return mock.generate_sample_dataset(random_state=get(SK.RANDOM_STATE, RANDOM_STATE))


# ---------------------------------------------------------------------------
# 4. Hàm tiện ích cho trang
# ---------------------------------------------------------------------------
def display_name(column: str) -> str:
    if column.startswith("favorite_category_"):
        cat = column.replace("favorite_category_", "", 1)
        return f"Danh mục: {cat}"
    return FEATURE_DISPLAY.get(column, column)


def default_feature_columns(candidate_columns: list[str]) -> list[str]:
    picked = [c for c in candidate_columns if not any(h in c.lower() for h in NON_FEATURE_HINTS)]
    return picked or candidate_columns


def matrix_for_space(space: str) -> np.ndarray | None:
    """Trả về ma trận đặc trưng ứng với không gian ('original' | 'pca' | 'umap' | 'tsne')."""
    norm = "original" if space == "scaled" else space
    if norm == "original":
        scale = get(SK.SCALE)
        return None if scale is None else np.asarray(scale["X_scaled"])
    if norm == "pca":
        pca = get(SK.PCA)
        if pca is not None:
            return np.asarray(pca["X_pca"])
        rep = get(SK.BENCHMARK_REPRESENTATIONS, {}).get("pca")
        return None if rep is None else np.asarray(rep["X"])
    if norm == "umap":
        rep = get(SK.BENCHMARK_REPRESENTATIONS, {}).get("umap")
        return None if rep is None else np.asarray(rep["X"])
    if norm == "tsne":
        t2 = get(SK.MAIN_TSNE_2D)
        if t2 is not None:
            return np.asarray(t2["embedding"])
        rep = get(SK.BENCHMARK_REPRESENTATIONS, {}).get("tsne")
        return None if rep is None else np.asarray(rep["X"])
    return None


def get_view_coordinates(n_components: int, method: str = "t-SNE") -> tuple[np.ndarray | None, str | None]:
    """Tọa độ 2D/3D để vẽ bản đồ trên luồng chính hoặc benchmark."""
    res = None
    if method == "t-SNE":
        res = get(SK.MAIN_TSNE_2D if n_components == 2 else SK.MAIN_TSNE_3D)
        if res is not None:
            return np.asarray(res["embedding"]), "t-SNE"
    elif method == "UMAP":
        rep = get(SK.BENCHMARK_REPRESENTATIONS, {}).get("umap")
        if rep is not None and np.asarray(rep["X"]).shape[1] >= n_components:
            return np.asarray(rep["X"])[:, :n_components], "UMAP"
    elif method == "PCA":
        pca = get(SK.PCA)
        if pca is not None and np.asarray(pca["X_pca"]).shape[1] >= n_components:
            return np.asarray(pca["X_pca"])[:, :n_components], "PCA"
        rep = get(SK.BENCHMARK_REPRESENTATIONS, {}).get("pca")
        if rep is not None and np.asarray(rep["X"]).shape[1] >= n_components:
            return np.asarray(rep["X"])[:, :n_components], "PCA"
    return None, None


def hover_dataframe(df_subset: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Bảng thông tin hiện khi rê chuột (thang đo gốc)."""
    prep = get(SK.PREP)
    if prep is None and df_subset is None:
        return None
    df = df_subset if df_subset is not None else prep["df_clean"]
    id_cols = [c for c in df.columns if c.lower() in ("customer_id", "name", "id")][:2]
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_cols][:8]
    cols = id_cols + num_cols
    return df[cols].round(3).reset_index(drop=True)


def cluster_label_for(algo: str, cluster_id: int) -> str:
    custom = get(SK.CLUSTER_NAMES, {}).get(algo, {}).get(cluster_id)
    if custom:
        return custom
    desc = get(SK.PROFILES, {}).get(algo, {}).get("descriptions", {}).get(cluster_id)
    if desc:
        return desc["name"]
    return "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"


# ---------------------------------------------------------------------------
# 5. Cảnh báo trạng thái backend
# ---------------------------------------------------------------------------
def render_backend_banner() -> None:
    status: dict = get(SK.BACKEND_STATUS, {})
    mocked = [n for n, s in status.items() if s == "mock"]
    st.markdown("**Trạng thái backend**")
    if get(SK.IS_SAMPLE):
        st.warning("Đang dùng dữ liệu mẫu giả lập, không phải dữ liệu đồ án.")
    if not status:
        st.caption("Chưa gọi hàm nào.")
    elif mocked:
        st.warning(f"{len(mocked)} hàm đang dùng bản MOCK. Số liệu hiển thị KHÔNG phải kết quả thực nghiệm.")
    else:
        st.success("Toàn bộ hàm đã gọi đều là bản thật.")
    with st.expander("Chi tiết từng hàm"):
        rows = [
            {"Hàm": n, "Trạng thái": {"real": "Thật", "mock": "MOCK"}.get(status.get(n), "Chưa gọi")}
            for n in _REGISTRY
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def algo_name(key: str) -> str:
    return ALGORITHMS.get(key, key)