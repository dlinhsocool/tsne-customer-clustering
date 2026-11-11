"""pages/clustering.py - Trang 4: chạy K-Means / GMM / DBSCAN."""
from __future__ import annotations

import numpy as np
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, DEFAULT_PARAMS, FEATURE_SPACES, NOISE_LABEL, SK, SUPPORTS_PREDICT

_ALGO_NOTES = {
    "kmeans": "Chia thành K cụm hình cầu bằng cách cực tiểu tổng bình phương khoảng cách tới tâm. Phải chọn trước K.",
    "gmm": "Mỗi cụm là một phân phối Gauss; cho phép cụm hình elip và trả về xác suất thuộc cụm.",
    "dbscan": "Cụm là vùng mật độ cao liên thông; không cần chọn K; điểm lẻ loi có nhãn Noise (-1). "
              "Không có predict() nên khách mới được gán bằng kNN.",
}


def _params_form(algo: str) -> dict:
    d = DEFAULT_PARAMS[algo]
    if algo == "kmeans":
        a, b, c = st.columns(3)
        return {"n_clusters": a.slider("Số cụm K", 2, 10, d["n_clusters"]),
                "n_init": b.number_input("n_init", 1, 50, d["n_init"]),
                "max_iter": c.number_input("max_iter", 50, 1000, d["max_iter"], step=50)}
    if algo == "gmm":
        a, b, c = st.columns(3)
        return {"n_components": a.slider("Số thành phần", 2, 10, d["n_components"]),
                "covariance_type": b.selectbox("Covariance", ["full", "tied", "diag", "spherical"]),
                "max_iter": c.number_input("max_iter", 20, 1000, d["max_iter"], step=20)}
    a, b = st.columns(2)
    return {"eps": a.slider("eps", 0.1, 5.0, float(d["eps"]), step=0.1),
            "min_samples": b.slider("min_samples", 2, 50, d["min_samples"])}


def render() -> None:
    st.title("Clustering")
    if not interface.require(SK.SCALE, message="Chưa có dữ liệu chuẩn hóa. Hãy chạy bước tiền xử lý ở trang Preprocessing & Dimensionality Reduction."):
        return

    st.caption("Phân cụm chạy trên dữ liệu đa chiều đã chuẩn hóa (hoặc sau PCA), không chạy trên tọa độ t-SNE. "
               "Bản đồ bên dưới chỉ dùng để quan sát.")

    algo = st.radio("Thuật toán", list(ALGORITHMS), format_func=lambda k: ALGORITHMS[k], horizontal=True)
    st.info(_ALGO_NOTES[algo])

    spaces = ["scaled"] + (["pca"] if interface.get(SK.PCA) is not None else [])
    space = st.radio("Không gian dữ liệu để phân cụm", spaces, horizontal=True,
                     format_func=lambda s: FEATURE_SPACES[s],
                     help="Chọn PCA sau khi đã chạy PCA ở trang trước.")
    params = _params_form(algo)

    if st.button("Run Clustering", type="primary"):
        interface.reset_from("cluster")
        with st.spinner(f"Đang chạy {ALGORITHMS[algo]}..."):
            res = interface.run_clustering(
                interface.matrix_for_space(space), algo, params,
                feature_space=space, random_state=interface.get(SK.RANDOM_STATE))
        interface.get(SK.CLUSTER_RESULTS)[algo] = res
        interface.get(SK.PROFILES).pop(algo, None)
        interface.put(SK.ACTIVE_ALGO, algo)

    results = interface.get(SK.CLUSTER_RESULTS)
    if algo not in results:
        st.caption("Chưa chạy thuật toán này.")
        return
    res = results[algo]
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số cụm", res["n_clusters"])
    m2.metric("Điểm noise", f"{res['n_noise']:,}")
    rt = res["runtime_seconds"]
    m3.metric("Thời gian chạy", "N/A" if rt is None else f"{rt:.2f} s")
    m4.metric("Có predict()", "Có" if SUPPORTS_PREDICT[algo] else "Không")
    if res["n_clusters"] < 2:
        st.warning("Chỉ có dưới 2 cụm: các chỉ số đánh giá sẽ không tính được. Thử đổi tham số.")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Số khách hàng từng cụm**")
        st.plotly_chart(viz.plot_cluster_sizes(res["cluster_sizes"]), width="stretch")
        st.caption(f"Phân cụm chạy trên: {FEATURE_SPACES[res['feature_space']].lower()}.")
    with right:
        hover = interface.hover_dataframe()
        tab2, tab3 = st.tabs(["Bản đồ 2D", "Bản đồ 3D"])
        for tab, dims in ((tab2, 2), (tab3, 3)):
            with tab:
                coords, source = interface.get_view_coordinates(dims)
                if coords is None:
                    st.info("Chưa có tọa độ. Chạy PCA hoặc t-SNE ở trang Preprocessing & Dimensionality Reduction.")
                    continue
                if source == "PCA":
                    st.caption("Chưa chạy t-SNE nên đang vẽ trên 2-3 thành phần PCA đầu tiên.")
                fn = viz.plot_scatter_2d if dims == 2 else viz.plot_scatter_3d
                st.plotly_chart(fn(coords, labels=res["labels"], hover_df=hover, axis_prefix=source),
                                width="stretch")
