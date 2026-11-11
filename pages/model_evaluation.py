"""pages/model_evaluation.py - Trang 5: so sánh các thuật toán."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, DEFAULT_PARAMS, SK


def _run_missing_with_defaults() -> None:
    results = interface.get(SK.CLUSTER_RESULTS)
    X = interface.matrix_for_space("scaled")
    for algo in ALGORITHMS:
        if algo not in results:
            with st.spinner(f"Đang chạy {ALGORITHMS[algo]} (tham số mặc định)..."):
                results[algo] = interface.run_clustering(
                    X, algo, DEFAULT_PARAMS[algo], feature_space="scaled",
                    random_state=interface.get(SK.RANDOM_STATE))
    interface.reset_from("cluster")


def render() -> None:
    st.title("Model Evaluation & Comparison")
    if not interface.require(SK.SCALE, message="Chưa có dữ liệu chuẩn hóa. Hãy chạy bước tiền xử lý trước."):
        return
    results = interface.get(SK.CLUSTER_RESULTS)
    prep = interface.get(SK.PREP)

    trained = [ALGORITHMS[a] for a in results]
    st.write("Đã huấn luyện: " + (", ".join(trained) if trained else "chưa có thuật toán nào."))
    if len(results) < len(ALGORITHMS):
        if st.button("Chạy các thuật toán còn thiếu với tham số mặc định"):
            _run_missing_with_defaults()
            st.rerun()
    if not results:
        return

    st.subheader("Nhãn tham chiếu (cho ARI)")
    df_clean = prep["df_clean"]
    candidates = [c for c in df_clean.columns if c not in prep["feature_columns"]]
    ref = st.selectbox("Cột nhãn thật", ["Không có"] + candidates,
                       help="Chỉ chọn nếu cột này là nhãn đúng và KHÔNG nằm trong các đặc trưng. "
                            "Dữ liệu thật không có nhãn thì để 'Không có'.")
    labels_true = None if ref == "Không có" else df_clean[ref].to_numpy()

    if st.button("Tính chỉ số", type="primary"):
        table = interface.compare_models(
            results, interface.matrix_for_space("scaled"),
            X_pca=interface.matrix_for_space("pca"), labels_true=labels_true)
        interface.put(SK.EVAL_TABLE, table)

    table = interface.get(SK.EVAL_TABLE)
    if table is None:
        st.info("Bấm 'Tính chỉ số' để so sánh.")
        return

    st.subheader("Bảng so sánh")
    show = table.copy()
    show["algorithm"] = show["algorithm"].map(ALGORITHMS).fillna(show["algorithm"])
    show = show.rename(columns={
        "algorithm": "Thuật toán", "n_clusters": "Số cụm", "n_noise": "Noise",
        "silhouette_score": "Silhouette", "davies_bouldin_score": "Davies-Bouldin",
        "ari_score": "ARI", "runtime_seconds": "Thời gian (s)"})
    num_cols = ["Silhouette", "Davies-Bouldin", "ARI", "Thời gian (s)"]
    for c in num_cols:
        show[c] = pd.to_numeric(show[c], errors="coerce")
    st.dataframe(
        show.style.format({c: "{:.3f}" for c in num_cols}, na_rep="N/A"),
        hide_index=True, width="stretch")
    st.caption("N/A: chỉ số không tính được (dưới 2 cụm, toàn bộ là noise, hoặc không có nhãn tham chiếu). "
               "Silhouette và Davies-Bouldin đã loại điểm noise.")

    st.subheader("Biểu đồ so sánh")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(viz.plot_metric_comparison(table, "silhouette_score", "Silhouette"), width="stretch")
        st.plotly_chart(viz.plot_metric_comparison(table, "ari_score", "ARI"), width="stretch")
    with b:
        st.plotly_chart(viz.plot_metric_comparison(table, "davies_bouldin_score", "Davies-Bouldin", True),
                        width="stretch")
        st.plotly_chart(viz.plot_metric_comparison(table, "runtime_seconds", "Thời gian chạy (s)", True),
                        width="stretch")
