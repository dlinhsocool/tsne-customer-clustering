"""pages/dimensionality_reduction.py - Trang 3: Tiền xử lý, chuẩn hóa & trực quan hóa t-SNE."""
from __future__ import annotations

import numpy as np
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import (
    ALGORITHMS,
    MISSING_STRATEGIES,
    OUTLIER_METHODS,
    SK,
    TSNE_DEFAULTS,
)


def _section_preprocess(df, val) -> None:
    st.subheader("1. Chọn biến, tiền xử lý và chuẩn hóa")
    candidate_cols = (
        val.get("numeric_columns", [])
        + val.get("datetime_columns", [])
        + val.get("categorical_columns", [])
    )
    if len(candidate_cols) < 2:
        st.error("Dữ liệu cần ít nhất 2 cột đặc trưng hợp lệ.")
        return

    saved = interface.get(SK.RAW_FEATURE_COLUMNS)
    default = [c for c in (saved or interface.default_feature_columns(candidate_cols)) if c in candidate_cols]
    features = st.multiselect(
        "Các cột đưa vào pipeline (tự động trích xuất ngày tháng & One-Hot biến phân loại)",
        candidate_cols,
        default=default,
        format_func=interface.display_name,
    )

    a, b, c = st.columns(3)
    missing = a.selectbox("Giá trị thiếu", MISSING_STRATEGIES, help="Trung vị bền với ngoại lai hơn trung bình.")
    outlier = b.selectbox(
        "Ngoại lai (trên biến số)",
        OUTLIER_METHODS,
        format_func=lambda v: {None: "Không loại", "iqr": "IQR", "zscore": "Z-score"}[v],
    )
    drop_dup = c.checkbox("Bỏ dòng trùng", value=True)

    num_selected = [f for f in features if f in val.get("numeric_columns", [])]
    skewed = [f for f in num_selected if abs(float(df[f].skew())) > 1]
    log_cols = st.multiselect(
        "Log-transform (log1p cho cột lệch phải)",
        num_selected,
        default=skewed,
        format_func=interface.display_name,
        help="Mặc định chọn sẵn các cột số có độ lệch mạnh (|skew| > 1).",
    )
    d, e = st.columns(2)
    sample = d.number_input(
        "Giới hạn số dòng cho luồng chính (0 = giữ hết)",
        min_value=0,
        value=3000,
        step=500,
        help="CustomTSNE tạo ma trận n×n nên khuyến nghị 2.000–5.000 dòng để chạy mượt.",
    )
    seed = e.number_input("Random state", min_value=0, value=int(interface.get(SK.RANDOM_STATE)), step=1)
    interface.put(SK.RANDOM_STATE, int(seed))

    if st.button("Chạy tiền xử lý và chuẩn hóa (StandardScaler)", type="primary", disabled=len(features) < 2):
        interface.reset_from("prep")
        prep = interface.preprocess_data(
            df,
            features,
            missing_strategy=missing,
            drop_duplicates=drop_dup,
            outlier_method=outlier,
            log_transform_columns=log_cols,
            sample_size=int(sample) or None,
            random_state=int(seed),
        )
        scale = interface.scale_data(prep["X"])
        interface.put(SK.RAW_FEATURE_COLUMNS, features)
        interface.put(SK.FEATURE_COLUMNS, prep["feature_columns"])
        interface.put(SK.PREP, prep)
        interface.put(SK.SCALE, scale)

    prep, scale = interface.get(SK.PREP), interface.get(SK.SCALE)
    if prep is None or scale is None:
        return
    Xs = np.asarray(scale["X_scaled"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số dòng trước", f"{prep['n_rows_before']:,}")
    m2.metric("Số dòng sau", f"{prep['n_rows_after']:,}")
    m3.metric("Số chiều X_scaled", Xs.shape[1])
    m4.metric("Đã chuẩn hóa", "Có")
    st.caption(
        f"Kiểm tra StandardScaler: |trung bình| lớn nhất = {np.abs(Xs.mean(axis=0)).max():.3f}, "
        f"độ lệch chuẩn từ {Xs.std(axis=0).min():.2f} đến {Xs.std(axis=0).max():.2f}."
    )
    with st.expander("Nhật ký tiền xử lý & danh sách đặc trưng sau mã hóa"):
        for line in prep["steps_log"] or ["Không có thay đổi."]:
            st.markdown(f"- {line}")
        st.write("**Danh sách đặc trưng đầu vào mô hình:**", prep["feature_columns"])


def _section_tsne() -> None:
    st.subheader("2. Trực quan hóa t-SNE trên dữ liệu gốc đã chuẩn hóa (X_scaled)")
    st.caption(
        "Trong luồng ứng dụng chính, t-SNE nhận trực tiếp X_scaled để dựng bản đồ 2D/3D phục vụ quan sát. "
        "Để so sánh PCA, UMAP và t-SNE, hãy mở trang Benchmark & Evaluation."
    )

    X = interface.matrix_for_space("original")
    n = len(X)

    a, b, c = st.columns(3)
    perplexity = a.slider("Perplexity", 5, 50, int(TSNE_DEFAULTS["perplexity"]), step=1)
    lr = b.number_input("Learning rate", min_value=10, max_value=1000, value=int(TSNE_DEFAULTS["learning_rate"]), step=10)
    iters = c.number_input("Number of iterations", min_value=250, max_value=5000, value=int(TSNE_DEFAULTS["n_iter"]), step=250)

    d, e = st.columns(2)
    want_3d = d.checkbox("Chạy thêm bản 3D", value=False)
    use_open = e.checkbox("Dùng openTSNE (hỗ trợ transform out-of-sample)", value=False)

    if perplexity >= n:
        st.error(f"Perplexity ({perplexity}) phải nhỏ hơn số dòng ({n}).")
        return
    if n > 8000 and not use_open:
        st.warning(f"{n:,} dòng: CustomTSNE tạo ma trận {n}×{n} có thể tốn RAM/thời gian. Nên chọn ≤ 5.000 dòng hoặc bật openTSNE.")

    if st.button("Chạy t-SNE"):
        interface.reset_from("main_tsne")
        seed = interface.get(SK.RANDOM_STATE)
        with st.spinner("Đang chạy t-SNE 2D trên X_scaled..."):
            interface.put(
                SK.MAIN_TSNE_2D,
                interface.run_tsne(
                    X,
                    n_components=2,
                    perplexity=float(perplexity),
                    learning_rate=float(lr),
                    n_iter=int(iters),
                    random_state=seed,
                    use_opentsne=use_open,
                ),
            )
        interface.put(SK.MAIN_TSNE_3D, None)
        if want_3d:
            with st.spinner("Đang chạy t-SNE 3D trên X_scaled..."):
                interface.put(
                    SK.MAIN_TSNE_3D,
                    interface.run_tsne(
                        X,
                        n_components=3,
                        perplexity=float(perplexity),
                        learning_rate=float(lr),
                        n_iter=int(iters),
                        random_state=seed,
                        use_opentsne=use_open,
                    ),
                )
        interface.put(SK.TSNE_INPUT_SPACE, "original")

    t2 = interface.get(SK.MAIN_TSNE_2D)
    if t2 is not None:
        kl = t2.get("kl_divergence")
        rt = t2.get("runtime_seconds")
        st.caption(
            f"t-SNE ({t2['backend']} · {t2.get('device', 'cpu')}) trên X_scaled, "
            f"perplexity = {t2['perplexity']:g}"
            + (f", KL = {kl:.3f}" if kl is not None else "")
            + (f", thời gian = {rt:.2f}s" if rt is not None else "")
            + "."
        )


def _section_visual() -> None:
    st.subheader("3. Bản đồ t-SNE 2D / 3D")
    if interface.get(SK.MAIN_TSNE_2D) is None:
        st.info("Bấm 'Chạy t-SNE' ở bước 2 để hiển thị bản đồ.")
        return

    clusters = interface.get(SK.CLUSTER_RESULTS)
    color_options = ["Không tô màu"] + [ALGORITHMS[a] for a in clusters]
    choice = st.selectbox("Tô màu theo kết quả phân cụm", color_options)
    labels = None
    if choice != "Không tô màu":
        key = next(k for k, v in ALGORITHMS.items() if v == choice)
        labels = clusters[key]["labels"]

    hover = interface.hover_dataframe()
    tab2, tab3 = st.tabs(["Bản đồ 2D", "Bản đồ 3D"])
    for tab, dims in ((tab2, 2), (tab3, 3)):
        with tab:
            t = interface.get(SK.MAIN_TSNE_2D if dims == 2 else SK.MAIN_TSNE_3D)
            if t is None:
                st.info("Chưa có dữ liệu 3D (bật ô 'Chạy thêm bản 3D' ở bước 2 rồi chạy lại t-SNE).")
                continue
            coords = np.asarray(t["embedding"])
            fn = viz.plot_scatter_2d if dims == 2 else viz.plot_scatter_3d
            st.plotly_chart(fn(coords, labels=labels, hover_df=hover, axis_prefix="t-SNE"), width="stretch")


def render() -> None:
    st.title("Preprocessing & t-SNE Visualization")
    df = interface.get(SK.RAW_DF)
    if not interface.require(SK.RAW_DF, message="Chưa có dữ liệu. Hãy sang trang Data Exploration để tải CSV."):
        return
    _section_preprocess(df, interface.get(SK.VALIDATION))
    if not interface.require(SK.SCALE, message="Chạy bước 1 để có dữ liệu chuẩn hóa X_scaled trước khi chạy t-SNE."):
        return
    st.divider()
    _section_tsne()
    st.divider()
    _section_visual()