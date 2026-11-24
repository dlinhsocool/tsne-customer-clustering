"""pages/new_customer.py - Trang 7: khách hàng mới (out-of-sample)."""
from __future__ import annotations

import numpy as np
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, FEATURE_SPACES, NOISE_LABEL, SK, SUPPORTS_PREDICT


def _x_in_space(prep_new: dict, space: str):
    return prep_new["X_new_pca"] if space == "pca" else prep_new["X_new_scaled"]


def _form(prep) -> dict | None:
    st.markdown("**Thông tin khách hàng mới**")
    df = prep["df_clean"]
    ranges = prep["preprocess_config"]["feature_ranges"]
    values: dict = {}
    with st.form("new_customer_form"):
        cols = st.columns(2)
        for i, f in enumerate(prep["feature_columns"]):
            lo, hi = ranges[f]
            med = float(df[f].median())
            values[f] = cols[i % 2].number_input(
                interface.display_name(f), value=med, format="%.3f",
                help=f"Trong dữ liệu huấn luyện: từ {lo:.3g} đến {hi:.3g}.")
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

    st.caption("Hai việc khác nhau: (1) gán khách mới vào cụm, (2) chiếu khách mới lên bản đồ t-SNE hiện tại. "
               "Vị trí trên bản đồ chỉ để hiển thị, không dùng để quyết định cụm.")

    prep, scale, pca = interface.get(SK.PREP), interface.get(SK.SCALE), interface.get(SK.PCA)
    algos = list(results)
    default = algos.index(interface.get(SK.ACTIVE_ALGO)) if interface.get(SK.ACTIVE_ALGO) in algos else 0

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        algo = st.selectbox("Thuật toán phân cụm đã huấn luyện", algos, index=default,
                            format_func=lambda k: ALGORITHMS[k])
        res = results[algo]

        st.markdown("**1. Phương pháp gán cụm**")
        options = ["auto", "knn"] + (["model_predict"] if SUPPORTS_PREDICT[algo] else [])
        labels_map = {"auto": "Tự động", "knn": "kNN", "model_predict": "predict() của mô hình"}
        method = st.radio("Cách gán", options, horizontal=True, format_func=lambda m: labels_map[m])
        if not SUPPORTS_PREDICT[algo]:
            st.warning(f"{ALGORITHMS[algo]} không có predict() trực tiếp: khách mới được gán bằng kNN, "
                       "và bị coi là noise nếu nằm xa mọi điểm lõi.")
        elif method == "auto":
            st.caption("Tự động: dùng predict() của mô hình.")
        k = st.slider("Số hàng xóm k", 3, 15, 5)

        st.markdown("**2. Vị trí trên bản đồ t-SNE**")
        has_map = interface.get(SK.TSNE_2D) is not None
        show_map = st.checkbox("Hiển thị vị trí trên bản đồ 2D", value=has_map, disabled=not has_map)
        map_method = "knn"
        if not has_map:
            st.caption("Chưa chạy t-SNE nên chưa có bản đồ để chiếu khách mới lên.")
        elif show_map:
            map_method = st.radio("Cách chiếu", ["knn", "opentsne"], horizontal=True,
                                  format_func=lambda m: {"knn": "kNN (trung bình tọa độ hàng xóm)",
                                                         "opentsne": "openTSNE transform"}[m])
        values = _form(prep)

    if values is not None:
        space = res["feature_space"]
        pn = interface.preprocess_new_customer(
            values, prep["preprocess_config"], scale["scaler"],
            pca["pca_model"] if pca is not None else None)
        assignment = interface.predict_customer_cluster(
            _x_in_space(pn, space), res, interface.matrix_for_space(space), method=method, k=k)
        position = None
        if has_map and show_map:
            t_space = interface.get(SK.TSNE_INPUT_SPACE) or "scaled"
            position = interface.transform_new_customer(
                _x_in_space(pn, t_space), interface.get(SK.TSNE_2D),
                interface.matrix_for_space(t_space), method=map_method, k=k)
        interface.put(SK.NEW_CUSTOMER, {"algo": algo, "prep": pn, "assignment": assignment,
                                        "position": position, "k": k})

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
            coords = np.asarray(interface.get(SK.TSNE_2D)["embedding"])
            labels = results[algo]["labels"]
            fig = viz.plot_scatter_2d(
                coords, labels=labels, hover_df=interface.hover_dataframe(), axis_prefix="t-SNE",
                new_point=np.asarray(pos["coordinates"])[0], neighbor_idx=a["neighbor_indices"])
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Cách chiếu: {'kNN' if pos['method_used'] == 'knn' else 'openTSNE'}. "
                       "Vòng đen là các hàng xóm gần nhất dùng để quyết định.")

        with st.expander("Hàng xóm gần nhất"):
            df = prep["df_clean"]
            idx = np.asarray(a["neighbor_indices"]).astype(int)
            nb = df.iloc[idx][prep["feature_columns"]].copy()
            nb.insert(0, "Cụm", results[algo]["labels"][idx])
            nb.insert(1, "Khoảng cách", np.asarray(a["neighbor_distances"]))
            st.dataframe(nb.round(3), width="stretch")
