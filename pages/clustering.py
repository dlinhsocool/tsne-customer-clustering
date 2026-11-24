"""pages/clustering.py - Trang 4: Phân cụm trên dữ liệu gốc đã chuẩn hóa (X_scaled)."""
from __future__ import annotations

import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, DEFAULT_PARAMS, SK, SUPPORTS_PREDICT

_ALGO_NOTES = {
    "kmeans": "Chia thành K cụm hình cầu trên X_scaled bằng cách cực tiểu tổng bình phương khoảng cách tới tâm.",
    "gmm": "Mô hình hỗn hợp Gauss (GMM) trên X_scaled; cho phép cụm hình elip và trả về xác suất thuộc cụm.",
    "dbscan": "Phân cụm theo mật độ trên X_scaled; không cần chọn trước K; điểm lẻ loi có nhãn Noise (-1).",
}


def _params_form(algo: str) -> dict:
    d = DEFAULT_PARAMS[algo]
    if algo == "kmeans":
        a, b, c = st.columns(3)
        return {
            "n_clusters": a.slider("Số cụm K", 2, 10, d["n_clusters"]),
            "n_init": b.number_input("n_init", 1, 50, d["n_init"]),
            "max_iter": c.number_input("max_iter", 50, 1000, d["max_iter"], step=50),
        }
    if algo == "gmm":
        a, b, c = st.columns(3)
        return {
            "n_components": a.slider("Số thành phần", 2, 10, d["n_components"]),
            "covariance_type": b.selectbox(
                "Covariance",
                ["full", "tied", "diag", "spherical"],
                index=["full", "tied", "diag", "spherical"].index(d["covariance_type"]),
            ),
            "max_iter": c.number_input("max_iter", 20, 1000, d["max_iter"], step=20),
        }
    a, b, c = st.columns(3)
    auto_eps = a.checkbox("Tự động ước lượng eps (k-NN)", value=True)
    eps_val = b.slider("eps thủ công", 0.1, 8.0, 2.5, step=0.1, disabled=auto_eps)
    min_s = c.slider("min_samples", 2, 50, d["min_samples"])
    return {"eps": None if auto_eps else float(eps_val), "min_samples": int(min_s)}


def render() -> None:
    st.title("Clustering")
    if not interface.require(
        SK.SCALE,
        message="Chưa có dữ liệu chuẩn hóa X_scaled. Hãy chạy bước 1 ở trang Preprocessing & t-SNE.",
    ):
        return

    st.caption(
        "Luồng ứng dụng chính luôn phân cụm trên dữ liệu gốc đã chuẩn hóa (X_scaled) và chiếu kết quả "
        "lên bản đồ t-SNE để quan sát."
    )

    algo = st.radio("Thuật toán", list(ALGORITHMS), format_func=lambda k: ALGORITHMS[k], horizontal=True)
    st.info(_ALGO_NOTES[algo])
    params = _params_form(algo)

    if st.button("Run Clustering", type="primary"):
        interface.reset_from("main_cluster")
        X_orig = interface.matrix_for_space("original")
        with st.spinner(f"Đang chạy {ALGORITHMS[algo]} trên X_scaled..."):
            res = interface.run_clustering(
                X_orig,
                algo,
                params,
                feature_space="original",
                random_state=interface.get(SK.RANDOM_STATE),
            )
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
    m3.metric("Thời gian chạy", "N/A" if rt is None else f"{rt:.3f} s")
    m4.metric("Có predict()", "Có" if SUPPORTS_PREDICT[algo] else "Không")
    st.caption(f"Tham số đã dùng: `{res['params']}`")

    if res["n_clusters"] < 2:
        st.warning("Chỉ có dưới 2 cụm: các chỉ số đánh giá sẽ không tính được. Hãy điều chỉnh tham số.")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Số khách hàng từng cụm**")
        st.plotly_chart(viz.plot_cluster_sizes(res["cluster_sizes"]), width="stretch")
    with right:
        hover = interface.hover_dataframe()
        tab2, tab3 = st.tabs(["Bản đồ t-SNE 2D", "Bản đồ t-SNE 3D"])
        for tab, dims in ((tab2, 2), (tab3, 3)):
            with tab:
                coords, source = interface.get_view_coordinates(dims, "t-SNE")
                if coords is None:
                    st.info("Chưa có tọa độ t-SNE. Hãy chạy t-SNE ở trang Preprocessing & t-SNE.")
                    continue
                fn = viz.plot_scatter_2d if dims == 2 else viz.plot_scatter_3d
                st.plotly_chart(
                    fn(coords, labels=res["labels"], hover_df=hover, axis_prefix=source),
                    width="stretch",
                )