"""
utils/interface.py
==================
LỚP TRUNG GIAN giữa giao diện (pages/) và backend (models/).

  * Trang (pages/*.py) CHỈ gọi hàm trong file này, không import models/ trực tiếp.
  * Mỗi hàm ở đây gọi hàm thật trong models/. Nếu hàm thật chưa viết (NotImplementedError)
    thì dùng bản giả trong utils/mock_backend.py và ghi trạng thái "mock".
  * Sau khi hàm thật chạy xong, output được kiểm tra theo hợp đồng (utils/contract.py).
    Sai hợp đồng => hiện lỗi rõ ràng chứ không để giao diện vỡ khó hiểu.
  * Quản lý st.session_state ở một chỗ duy nhất (init_state / get / put / reset_from).

Tên và tham số của các hàm bọc dưới đây GIỐNG HỆT hàm trong models/, nên ví dụ gọi trong
docstring của models/ dùng được nguyên văn: interface.run_pca(...), interface.run_tsne(...).
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
    ALGORITHMS, FEATURE_DISPLAY, NON_FEATURE_HINTS, RANDOM_STATE, SK,
    USE_MOCK_WHEN_NOT_IMPLEMENTED,
)
from utils.contract import check_contract

# ---------------------------------------------------------------------------
# 1. Bảng đăng ký: tên hàm -> (hàm thật, hàm giả)
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "load_dataset": (m_data.load_dataset, mock.load_dataset),
    "validate_dataset": (m_data.validate_dataset, mock.validate_dataset),
    "preprocess_data": (m_data.preprocess_data, mock.preprocess_data),
    "scale_data": (m_data.scale_data, mock.scale_data),
    "run_pca": (m_dr.run_pca, mock.run_pca),
    "get_explained_variance": (m_dr.get_explained_variance, mock.get_explained_variance),
    "run_tsne": (m_dr.run_tsne, mock.run_tsne),
    "run_clustering": (m_clustering.run_clustering, mock.run_clustering),
    "evaluate_clustering": (m_eval.evaluate_clustering, mock.evaluate_clustering),
    "compare_models": (m_eval.compare_models, mock.compare_models),
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
    SK.RAW_DF: None, SK.IS_SAMPLE: False, SK.UPLOAD_SIG: None, SK.VALIDATION: None,
    SK.FEATURE_COLUMNS: None, SK.PREP: None, SK.SCALE: None, SK.PCA: None,
    SK.TSNE_2D: None, SK.TSNE_3D: None, SK.TSNE_INPUT_SPACE: None,
    SK.CLUSTER_RESULTS: {}, SK.ACTIVE_ALGO: None, SK.EVAL_TABLE: None,
    SK.PROFILES: {}, SK.CLUSTER_NAMES: {}, SK.NEW_CUSTOMER: None,
    SK.RANDOM_STATE: RANDOM_STATE, SK.BACKEND_STATUS: {},
}

# Khi bước ở trái thay đổi thì các kết quả ở phải bị xóa (tránh dùng dữ liệu cũ)
_RESET_GROUPS: dict[str, list[str]] = {
    "data": [SK.VALIDATION, SK.FEATURE_COLUMNS, SK.PREP, SK.SCALE, SK.PCA, SK.TSNE_2D, SK.TSNE_3D,
             SK.TSNE_INPUT_SPACE, SK.CLUSTER_RESULTS, SK.ACTIVE_ALGO, SK.EVAL_TABLE, SK.PROFILES,
             SK.CLUSTER_NAMES, SK.NEW_CUSTOMER],
    "prep": [SK.SCALE, SK.PCA, SK.TSNE_2D, SK.TSNE_3D, SK.TSNE_INPUT_SPACE, SK.CLUSTER_RESULTS,
             SK.ACTIVE_ALGO, SK.EVAL_TABLE, SK.PROFILES, SK.CLUSTER_NAMES, SK.NEW_CUSTOMER],
    "pca": [SK.TSNE_2D, SK.TSNE_3D, SK.TSNE_INPUT_SPACE, SK.CLUSTER_RESULTS, SK.ACTIVE_ALGO,
            SK.EVAL_TABLE, SK.PROFILES, SK.CLUSTER_NAMES, SK.NEW_CUSTOMER],
    "tsne": [SK.NEW_CUSTOMER],
    "cluster": [SK.EVAL_TABLE, SK.NEW_CUSTOMER],
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
    """Xóa kết quả phía sau `step` ("data" | "prep" | "pca" | "tsne" | "cluster")."""
    for key in _RESET_GROUPS[step]:
        st.session_state[key] = copy.deepcopy(_DEFAULTS[key])


def require(*keys: str, message: str) -> bool:
    """Trả False và hiện hướng dẫn nếu thiếu bước trước. Dùng: if not require(...): return"""
    if any(get(k) is None for k in keys):
        st.info(message)
        return False
    return True


# ---------------------------------------------------------------------------
# 3. Hàm bọc backend (giữ nguyên chữ ký như models/)
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


def run_tsne(*a: Any, **k: Any) -> dict:
    return _call("run_tsne", *a, **k)


def run_clustering(*a: Any, **k: Any) -> dict:
    return _call("run_clustering", *a, **k)


def evaluate_clustering(*a: Any, **k: Any) -> dict:
    return _call("evaluate_clustering", *a, **k)


def compare_models(*a: Any, **k: Any) -> pd.DataFrame:
    return _call("compare_models", *a, **k)


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
    return FEATURE_DISPLAY.get(column, column)


def default_feature_columns(numeric_columns: list[str]) -> list[str]:
    picked = [c for c in numeric_columns if not any(h in c.lower() for h in NON_FEATURE_HINTS)]
    return picked or numeric_columns


def matrix_for_space(space: str) -> np.ndarray | None:
    """Ma trận theo không gian: "scaled" -> X_scaled, "pca" -> X_pca. None nếu chưa có."""
    if space == "scaled":
        scale = get(SK.SCALE)
        return None if scale is None else np.asarray(scale["X_scaled"])
    if space == "pca":
        pca = get(SK.PCA)
        return None if pca is None else np.asarray(pca["X_pca"])
    return None


def get_view_coordinates(n_components: int) -> tuple[np.ndarray | None, str | None]:
    """Tọa độ để vẽ bản đồ: ưu tiên t-SNE; chưa chạy thì dùng PCA. Trả (coords, nguồn)."""
    tsne = get(SK.TSNE_2D if n_components == 2 else SK.TSNE_3D)
    if tsne is not None:
        return np.asarray(tsne["embedding"]), "t-SNE"
    pca = get(SK.PCA)
    if pca is not None and np.asarray(pca["X_pca"]).shape[1] >= n_components:
        return np.asarray(pca["X_pca"])[:, :n_components], "PCA"
    return None, None


def hover_dataframe() -> pd.DataFrame | None:
    """Bảng thông tin hiện khi rê chuột: cột id (nếu có) + tối đa 8 đặc trưng, thang đo gốc."""
    prep = get(SK.PREP)
    if prep is None:
        return None
    df = prep["df_clean"]
    id_cols = [c for c in df.columns if c.lower() in ("customer_id", "id")][:1]
    cols = id_cols + [c for c in prep["feature_columns"][:8] if c not in id_cols]
    return df[cols].round(3).reset_index(drop=True)


def cluster_label_for(algo: str, cluster_id: int) -> str:
    """Tên cụm để hiển thị: tên người dùng đặt > mô tả sinh từ dữ liệu > "Cluster k"."""
    custom = get(SK.CLUSTER_NAMES, {}).get(algo, {}).get(cluster_id)
    if custom:
        return custom
    desc = get(SK.PROFILES, {}).get(algo, {}).get("descriptions", {}).get(cluster_id)
    if desc:
        return desc["name"]
    return "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"


# ---------------------------------------------------------------------------
# 5. Cảnh báo trạng thái backend (hiện ở thanh bên)
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
        rows = [{"Hàm": n, "Trạng thái": {"real": "Thật", "mock": "MOCK"}.get(status.get(n), "Chưa gọi")}
                for n in _REGISTRY]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def algo_name(key: str) -> str:
    return ALGORITHMS.get(key, key)
