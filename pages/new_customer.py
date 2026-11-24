"""pages/new_customer.py - Trang 7: Dự đoán phân khúc & định vị khách hàng mới."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, SK, SUPPORTS_PREDICT


def _to_1d_int_array(values: Any) -> np.ndarray:
    """Chuyển neighbor_indices về ndarray int một chiều; xử lý None và mảng lồng."""
    if values is None:
        return np.array([], dtype=int)

    try:
        arr = np.asarray(values)
        if arr.size == 0:
            return np.array([], dtype=int)

        # Chuyển qua số thực để loại NaN/Inf trước khi ép kiểu int.
        flat = arr.astype(float).reshape(-1)
        flat = flat[np.isfinite(flat)]
        return flat.astype(int)
    except (TypeError, ValueError, OverflowError):
        return np.array([], dtype=int)


def _to_1d_float_array(values: Any) -> np.ndarray:
    """Chuyển neighbor_distances về ndarray float một chiều."""
    if values is None:
        return np.array([], dtype=float)

    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    except (TypeError, ValueError, OverflowError):
        return np.array([], dtype=float)


def _form(prep: dict[str, Any]) -> dict[str, Any] | None:
    """Tạo form nhập khách hàng mới theo schema dữ liệu huấn luyện."""
    st.markdown("**Thông tin khách hàng mới**")

    df = prep["df_clean"]
    cfg = prep["preprocess_config"]
    raw_cols = cfg.get("raw_feature_columns") or prep["feature_columns"]
    classified = cfg.get("column_roles") or {}
    dt_set = set(classified.get("datetime_columns", []))
    cat_set = set(classified.get("categorical_columns", []))
    raw_ranges = cfg.get("raw_feature_ranges") or cfg.get("feature_ranges", {})
    category_levels = cfg.get("category_levels", {})
    raw_fill_values = cfg.get("raw_fill_values", {})

    values: dict[str, Any] = {}

    with st.form("new_customer_form"):
        cols = st.columns(2)

        for i, col_name in enumerate(raw_cols):
            box = cols[i % 2]
            label = interface.display_name(col_name)

            if col_name in dt_set:
                default_str = str(raw_fill_values.get(col_name, "2024-06-01"))
                default_dt = pd.to_datetime(default_str, errors="coerce")
                if pd.isna(default_dt):
                    default_dt = pd.Timestamp("2024-06-01")

                picked = box.date_input(label, value=default_dt.date())
                values[col_name] = picked.strftime("%Y-%m-%d")

            elif col_name in cat_set:
                options = category_levels.get(col_name)
                if options:
                    values[col_name] = box.selectbox(label, options)
                else:
                    default_bool = bool(raw_fill_values.get(col_name, False))
                    values[col_name] = box.checkbox(label, value=default_bool)

            else:
                lo, hi = raw_ranges.get(col_name, (0.0, 10000.0))
                if col_name in df.columns:
                    median_value = pd.to_numeric(df[col_name], errors="coerce").median()
                else:
                    median_value = np.nan

                if pd.isna(median_value):
                    median_value = (float(lo) + float(hi)) / 2.0

                values[col_name] = box.number_input(
                    label,
                    value=float(median_value),
                    format="%.3f",
                    help=f"Trong dữ liệu huấn luyện: từ {lo:.3g} đến {hi:.3g}.",
                )

        submitted = st.form_submit_button(
            "Phân tích khách hàng",
            type="primary",
            width="stretch",
        )

    return values if submitted else None


def _build_neighbor_table(
    prep: dict[str, Any],
    algorithm: str,
    results: dict[str, Any],
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> pd.DataFrame | None:
    """Tạo bảng hàng xóm gần nhất, bảo đảm mọi chỉ số đều là mảng 1D hợp lệ."""
    df = prep.get("df_clean")
    if not isinstance(df, pd.DataFrame):
        return None

    df = df.reset_index(drop=True)
    if neighbor_indices.size == 0:
        return None

    valid_mask = (neighbor_indices >= 0) & (neighbor_indices < len(df))
    valid_indices = neighbor_indices[valid_mask]
    if valid_indices.size == 0:
        return None

    cfg = prep.get("preprocess_config", {})
    candidate_cols = (
        cfg.get("raw_feature_columns")
        or prep.get("feature_columns", [])
    )
    show_cols = [col for col in candidate_cols if col in df.columns]
    if not show_cols:
        show_cols = df.columns.tolist()

    # valid_indices luôn có shape (k,), tránh PyArrow nhận chỉ số 2 chiều.
    neighbor_df = df.iloc[valid_indices][show_cols].copy()

    labels = np.asarray(results[algorithm]["labels"]).reshape(-1)
    if len(labels) == len(df):
        neighbor_df.insert(0, "Cụm", labels[valid_indices])

    # Chỉ thêm khoảng cách khi số khoảng cách khớp số chỉ số ban đầu.
    if neighbor_distances.size == neighbor_indices.size:
        valid_distances = neighbor_distances[valid_mask]
        insert_position = 1 if "Cụm" in neighbor_df.columns else 0
        neighbor_df.insert(insert_position, "Khoảng cách", valid_distances)

    return neighbor_df


def render() -> None:
    """Hiển thị form, dự đoán cụm và vị trí khách hàng mới trên t-SNE."""
    st.title("New Customer")

    if not interface.require(
        SK.PREP,
        SK.SCALE,
        message="Chưa có pipeline. Hãy chạy tiền xử lý và chuẩn hóa trước.",
    ):
        return

    results = interface.get(SK.CLUSTER_RESULTS)
    if not results:
        st.info("Chưa có kết quả phân cụm. Hãy chạy một thuật toán ở trang Clustering.")
        return

    st.caption(
        "Hai bước độc lập trên X_scaled: (1) gán khách mới vào cụm bằng mô hình phân cụm chính, "
        "(2) chiếu khách mới lên bản đồ t-SNE 2D để quan sát vị trí."
    )

    prep = interface.get(SK.PREP)
    scale = interface.get(SK.SCALE)
    algos = list(results)

    active_algo = interface.get(SK.ACTIVE_ALGO)
    default_index = algos.index(active_algo) if active_algo in algos else 0

    left, right = st.columns([1, 1.5], gap="large")

    with left:
        selected_algo = st.selectbox(
            "Thuật toán phân cụm đã huấn luyện",
            algos,
            index=default_index,
            format_func=lambda key: ALGORITHMS.get(key, key),
        )
        selected_result = results[selected_algo]

        st.markdown("**1. Phương pháp gán cụm (trên X_scaled)**")
        options = ["auto", "knn"]
        if SUPPORTS_PREDICT.get(selected_algo, False):
            options.append("model_predict")

        labels_map = {
            "auto": "Tự động",
            "knn": "kNN",
            "model_predict": "predict() của mô hình",
        }
        method = st.radio(
            "Cách gán",
            options,
            horizontal=True,
            format_func=lambda value: labels_map[value],
        )

        supports_predict = SUPPORTS_PREDICT.get(selected_algo, False)
        if not supports_predict:
            st.warning(
                f"{ALGORITHMS.get(selected_algo, selected_algo)} không có predict() trực tiếp: "
                "khách mới được gán bằng kNN, và có thể được nhận diện là noise nếu nằm xa mọi điểm lõi."
            )
        elif method == "auto":
            st.caption("Tự động: dùng predict() của mô hình.")

        k = st.slider("Số hàng xóm k", 3, 15, 5)

        st.markdown("**2. Vị trí trên bản đồ t-SNE**")
        has_map = interface.get(SK.MAIN_TSNE_2D) is not None
        show_map = st.checkbox(
            "Hiển thị vị trí trên bản đồ 2D",
            value=has_map,
            disabled=not has_map,
        )

        map_method = "knn"
        if not has_map:
            st.caption("Chưa chạy t-SNE nên chưa có bản đồ để chiếu khách mới lên.")
        elif show_map:
            map_method = st.radio(
                "Cách chiếu",
                ["knn", "opentsne"],
                horizontal=True,
                format_func=lambda value: {
                    "knn": "kNN (trung bình tọa độ hàng xóm)",
                    "opentsne": "openTSNE transform",
                }[value],
            )

        values = _form(prep)

    # Chạy pipeline khi người dùng bấm nút trong form.
    if values is not None:
        X_original = interface.matrix_for_space("original")
        if X_original is None:
            st.error("Không tìm thấy X_scaled. Hãy chạy lại bước chuẩn hóa dữ liệu.")
            return

        new_customer_result = interface.preprocess_new_customer(
            values,
            prep["preprocess_config"],
            scale["scaler"],
            pca_model=None,
        )

        assignment = interface.predict_customer_cluster(
            new_customer_result["X_new_scaled"],
            selected_result,
            X_original,
            method=method,
            k=k,
        )

        position = None
        if has_map and show_map:
            tsne_result = interface.get(SK.MAIN_TSNE_2D)
            position = interface.transform_new_customer(
                new_customer_result["X_new_scaled"],
                tsne_result,
                X_original,
                method=map_method,
                k=k,
            )

        interface.put(
            SK.NEW_CUSTOMER,
            {
                "algo": selected_algo,
                "prep": new_customer_result,
                "assignment": assignment,
                "position": position,
                "k": k,
            },
        )

    out = interface.get(SK.NEW_CUSTOMER)

    with right:
        # Không hiển thị kết quả cũ của thuật toán khác sau khi người dùng đổi lựa chọn.
        if out is None or out.get("algo") != selected_algo:
            st.info("Nhập thông tin ở bên trái rồi bấm Phân tích khách hàng.")
            return

        assignment = out["assignment"]
        new_customer_result = out["prep"]
        position = out.get("position")

        # Chuẩn hóa hai trường neighbor_* ngay khi đọc kết quả backend.
        neighbor_indices = _to_1d_int_array(assignment.get("neighbor_indices"))
        neighbor_distances = _to_1d_float_array(assignment.get("neighbor_distances"))

        st.markdown("**Kết quả gán cụm**")
        if assignment.get("is_noise", False):
            st.warning(
                "Khách hàng này nằm xa mọi cụm (noise / bất thường). "
                "Không ép vào cụm nào; nên xem riêng."
            )
        else:
            st.success(interface.cluster_label_for(selected_algo, int(assignment["cluster"])))

        col1, col2, col3 = st.columns(3)
        method_labels = {
            "model_predict": "predict()",
            "knn": "kNN",
        }
        col1.metric(
            "Phương pháp",
            method_labels.get(assignment.get("method_used"), "Không xác định"),
        )

        confidence = assignment.get("confidence")
        col2.metric(
            "Độ tin cậy",
            "N/A" if confidence is None else f"{float(confidence):.0%}",
        )

        mean_distance = assignment.get("mean_neighbor_distance")
        col3.metric(
            "KC TB tới hàng xóm",
            "N/A" if mean_distance is None else f"{float(mean_distance):.2f}",
        )

        for warning in assignment.get("warnings", []) + new_customer_result.get("warnings", []):
            st.warning(warning)

        out_of_range = new_customer_result.get("out_of_range_features", [])
        if out_of_range:
            feature_names = ", ".join(interface.display_name(name) for name in out_of_range)
            st.warning(
                "Giá trị ngoài khoảng đã thấy trong dữ liệu huấn luyện: "
                f"{feature_names}. Kết quả kém chắc chắn hơn."
            )

        if assignment.get("is_outlier", False) and not assignment.get("is_noise", False):
            st.warning(
                "Khách này xa các hàng xóm hơn thông thường "
                "(dấu hiệu dữ liệu mới lệch khỏi dữ liệu huấn luyện)."
            )

        st.markdown("**Vị trí trên bản đồ t-SNE**")
        if position is None:
            st.caption("Không hiển thị (chưa chạy t-SNE hoặc đang tắt).")
        else:
            for warning in position.get("warnings", []):
                st.warning(warning)

            tsne_result = interface.get(SK.MAIN_TSNE_2D)
            coords = np.asarray(tsne_result["embedding"], dtype=float)
            if coords.ndim != 2 or coords.shape[1] < 2:
                st.warning("Embedding t-SNE không có đủ hai chiều để vẽ.")
            else:
                labels = np.asarray(results[selected_algo]["labels"]).reshape(-1)
                if len(labels) != len(coords):
                    st.warning(
                        "Số nhãn phân cụm không khớp số điểm trên embedding t-SNE; "
                        "biểu đồ sẽ không hiển thị màu cụm."
                    )
                    labels_for_plot = None
                else:
                    labels_for_plot = labels

                hover_df = interface.hover_dataframe()
                if hover_df is not None and len(hover_df) != len(coords):
                    hover_df = None

                # Chỉ số hàng xóm phải hợp lệ đối với tọa độ t-SNE.
                plot_neighbor_indices = neighbor_indices[
                    (neighbor_indices >= 0) & (neighbor_indices < len(coords))
                ]
                if plot_neighbor_indices.size == 0:
                    plot_neighbor_indices = None

                new_point = np.asarray(position["coordinates"], dtype=float).reshape(-1)
                if new_point.size >= 2:
                    fig = viz.plot_scatter_2d(
                        coords,
                        labels=labels_for_plot,
                        hover_df=hover_df,
                        axis_prefix="t-SNE",
                        new_point=new_point[:2],
                        neighbor_idx=plot_neighbor_indices,
                    )
                    st.plotly_chart(fig, width="stretch")

                    if position.get("method_used") == "knn":
                        st.caption(
                            "Cách chiếu: kNN. Vòng đen biểu thị các hàng xóm gần nhất "
                            "được dùng để nội suy vị trí trên bản đồ."
                        )
                    else:
                        st.caption("Cách chiếu: openTSNE transform.")
                else:
                    st.warning("Không nhận được tọa độ hợp lệ cho khách hàng mới.")

        with st.expander("Hàng xóm gần nhất"):
            if neighbor_indices.size == 0:
                st.info(
                    "Không có danh sách hàng xóm cho phương pháp predict() trực tiếp "
                    "hoặc backend không trả về neighbor_indices."
                )
            else:
                # Dùng PREP của tập huấn luyện, không dùng kết quả preprocess khách mới.
                neighbor_table = _build_neighbor_table(
                    prep,
                    selected_algo,
                    results,
                    neighbor_indices,
                    neighbor_distances,
                )
                if neighbor_table is None or neighbor_table.empty:
                    st.warning("Không tìm thấy hàng xóm hợp lệ trong dữ liệu huấn luyện.")
                else:
                    st.dataframe(
                        neighbor_table.round(3),
                        width="stretch",
                        hide_index=True,
                    )
