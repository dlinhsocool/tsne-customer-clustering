"""pages/new_customer.py - Trang 7: Dự đoán phân khúc & định vị khách hàng mới (out-of-sample)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, SK, SUPPORTS_PREDICT


def _form(prep: dict) -> dict | None:
    st.markdown("**Thông tin khách hàng mới**")
    df = prep["df_clean"]
    cfg = prep["preprocess_config"]
    raw_cols = cfg.get("raw_feature_columns") or prep["feature_columns"]
    classified = cfg.get("column_roles") or {}
    dt_set = set(classified.get("datetime_columns", []))
    cat_set = set(classified.get("categorical_columns", []))
    raw_ranges = cfg.get("raw_feature_ranges") or cfg.get("feature_ranges", {})
    category_levels = cfg.get("category_levels", {})

    values: dict = {}
    with st.form("new_customer_form"):
        cols = st.columns(2)
        for i, col_name in enumerate(raw_cols):
            box = cols[i % 2]
            label = interface.display_name(col_name)

            if col_name in dt_set:
                default_str = str(cfg.get("raw_fill_values", {}).get(col_name, "2024-06-01"))
                default_dt = pd.to_datetime(default_str, errors="coerce")
                default_date = default_dt.date() if pd.notna(default_dt) else pd.Timestamp("2024-06-01").date()
                picked = box.date_input(label, value=default_date)
                values[col_name] = picked.strftime("%Y-%m-%d")
            elif col_name in cat_set:
                if col_name in category_levels:
                    opts = category_levels[col_name]
                    values[col_name] = box.selectbox(label, opts)
                else:
                    default_bool = bool(cfg.get("raw_fill_values", {}).get(col_name, False))
                    values[col_name] = box.checkbox(label, value=default_bool)
            else:
                lo, hi = raw_ranges.get(col_name, (0.0, 10000.0))
                med = float(df[col_name].median()) if col_name in df.columns else float((lo + hi) / 2.0)
                values[col_name] = box.number_input(
                    label,
                    value=med,
                    format="%.3f",
                    help=f"Trong dữ liệu huấn luyện: từ {lo:.3g} đến {hi:.3g}.",
                )
        submitted = st.form_submit_button("Phân tích khách hàng", type="primary", width="stretch")
    return values if submitted else None


def render() -> None:
    st.title("New Customer")
    if not interface.require(SK.PREP, SK.SCALE, message="Chưa có pipeline. Hãy chạy tiền xử lý và chuẩn hóa trước."):
        return
    results = interface.get(SK.CLUSTER_RESULTS)
    if not results:
        st.info("Chưa có kết quả phân cụm. Hãy chạy một thuật toán ở trang Clustering.")
        return

    st.caption(
        "Hai bước độc lập trên X_scaled: (1) gán khách mới vào cụm bằng mô hình phân cụm chính, "
        "(2) chiếu khách mới lên bản đồ t-SNE 2D để quan sát vị trí."
    )

    prep, scale = interface.get(SK.PREP), interface.get(SK.SCALE)
    algos = list(results)
    default = algos.index(interface.get(SK.ACTIVE_ALGO)) if interface.get(SK.ACTIVE_ALGO) in algos else 0

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        algo = st.selectbox(
            "Thuật toán phân cụm đã huấn luyện",
            algos,
            index=default,
            format_func=lambda k: ALGORITHMS[k],
        )
        res = results[algo]

        st.markdown("**1. Phương pháp gán cụm (trên X_scaled)**")
        options = ["auto", "knn"] + (["model_predict"] if SUPPORTS_PREDICT[algo] else [])
        labels_map = {"auto": "Tự động", "knn": "kNN", "model_predict": "predict() của mô hình"}
        method = st.radio("Cách gán", options, horizontal=True, format_func=lambda m: labels_map[m])
        if not SUPPORTS_PREDICT[algo]:
            st.warning(
                f"{ALGORITHMS[algo]} không có predict() trực tiếp: khách mới được gán bằng kNN, "
                "và bị coi là noise nếu nằm xa mọi điểm lõi."
            )
        elif method == "auto":
            st.caption("Tự động: dùng predict() của mô hình.")
        k = st.slider("Số hàng xóm k", 3, 15, 5)

        st.markdown("**2. Vị trí trên bản đồ t-SNE**")
        has_map = interface.get(SK.MAIN_TSNE_2D) is not None
        show_map = st.checkbox("Hiển thị vị trí trên bản đồ 2D", value=has_map, disabled=not has_map)
        map_method = "knn"
        if not has_map:
            st.caption("Chưa chạy t-SNE nên chưa có bản đồ để chiếu khách mới lên.")
        elif show_map:
            map_method = st.radio(
                "Cách chiếu",
                ["knn", "opentsne"],
                horizontal=True,
                format_func=lambda m: {
                    "knn": "kNN (trung bình tọa độ hàng xóm)",
                    "opentsne": "openTSNE transform",
                }[m],
            )
        values = _form(prep)

    if values is not None:
        X_orig = interface.matrix_for_space("original")
        pn = interface.preprocess_new_customer(values, prep["preprocess_config"], scale["scaler"], pca_model=None)
        assignment = interface.predict_customer_cluster(
            pn["X_new_scaled"], res, X_orig, method=method, k=k
        )
        position = None
        if has_map and show_map:
            position = interface.transform_new_customer(
                pn["X_new_scaled"], interface.get(SK.MAIN_TSNE_2D), X_orig, method=map_method, k=k
            )
        interface.put(
            SK.NEW_CUSTOMER,
            {"algo": algo, "prep": pn, "assignment": assignment, "position": position, "k": k},
        )

    out = interface.get(SK.NEW_CUSTOMER)
    with right:
        if out is None or out["algo"] not in results:
            st.info("Nhập thông tin ở bên trái rồi bấm Phân tích khách hàng.")
            return
        a, pn, pos = out["assignment"], out["prep"], out["position"]
        algo = out["algo"]

        st.markdown("**Kết quả gán cụm**")
        if a["is_noise"]:
            st.warning("Khách hàng này nằm xa mọi cụm (noise / bất thường). Không ép vào cụm nào; nên xem riêng.")
        else:
            st.success(interface.cluster_label_for(algo, a["cluster"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Phương pháp", {"model_predict": "predict()", "knn": "kNN"}[a["method_used"]])
        c2.metric("Độ tin cậy", "N/A" if a["confidence"] is None else f"{a['confidence']:.0%}")
        c3.metric("KC TB tới hàng xóm", f"{a['mean_neighbor_distance']:.2f}")
        for w in a["warnings"] + pn["warnings"]:
            st.warning(w)
        if pn["out_of_range_features"]:
            names = ", ".join(interface.display_name(f) for f in pn["out_of_range_features"])
            st.warning(f"Giá trị ngoài khoảng đã thấy trong dữ liệu huấn luyện: {names}. Kết quả kém chắc chắn hơn.")
        if a["is_outlier"] and not a["is_noise"]:
            st.warning("Khách này xa các hàng xóm hơn thông thường (dấu hiệu dữ liệu mới lệch khỏi dữ liệu huấn luyện).")

        st.markdown("**Vị trí trên bản đồ t-SNE**")
        if pos is None:
            st.caption("Không hiển thị (chưa chạy t-SNE hoặc đang tắt).")
        else:
            for w in pos["warnings"]:
                st.warning(w)
            coords = np.asarray(interface.get(SK.MAIN_TSNE_2D)["embedding"])
            labels = results[algo]["labels"]
            fig = viz.plot_scatter_2d(
                coords,
                labels=labels,
                hover_df=interface.hover_dataframe(),
                axis_prefix="t-SNE",
                new_point=np.asarray(pos["coordinates"])[0],
                neighbor_idx=a["neighbor_indices"],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Cách chiếu: {'kNN' if pos['method_used'] == 'knn' else 'openTSNE'}. "
                "Vòng đen là các hàng xóm gần nhất dùng để quyết định."
            )

        with st.expander("Hàng xóm gần nhất"):
            df = prep["df_clean"]
            idx = np.asarray(a["neighbor_indices"]).astype(int)
            show_cols = [c for c in (prep["preprocess_config"].get("raw_feature_columns") or prep["feature_columns"]) if c in df.columns]
            nb = df.iloc[idx][show_cols].copy()
            nb.insert(0, "Cụm", results[algo]["labels"][idx])
            nb.insert(1, "Khoảng cách", np.asarray(a["neighbor_distances"]))
            st.dataframe(nb.round(3), width="stretch")