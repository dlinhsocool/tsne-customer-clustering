"""pages/dimensionality_reduction.py - Trang 3: tiền xử lý, chuẩn hóa, PCA, t-SNE."""
from __future__ import annotations

import numpy as np
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import (
    FEATURE_SPACES, MISSING_STRATEGIES, OUTLIER_METHODS, PCA_DEFAULT_VARIANCE,
    SK, TSNE_DEFAULTS, ALGORITHMS,
)


def _section_preprocess(df, val) -> None:
    st.subheader("1. Chọn biến và tiền xử lý")
    numeric = val["numeric_columns"]
    if len(numeric) < 2:
        st.error("Dữ liệu cần ít nhất 2 cột số.")
        return

    saved = interface.get(SK.FEATURE_COLUMNS)
    default = [c for c in (saved or interface.default_feature_columns(numeric)) if c in numeric]
    features = st.multiselect("Biến đầu vào", numeric, default=default, format_func=interface.display_name)

    a, b, c = st.columns(3)
    missing = a.selectbox("Giá trị thiếu", MISSING_STRATEGIES,
                          help="Trung vị bền với ngoại lai hơn trung bình.")
    outlier = b.selectbox("Ngoại lai", OUTLIER_METHODS, format_func=lambda v: {None: "Không loại", "iqr": "IQR", "zscore": "Z-score"}[v])
    drop_dup = c.checkbox("Bỏ dòng trùng", value=True)

    skewed = [f for f in features if abs(float(df[f].skew())) > 1]
    log_cols = st.multiselect("Log-transform (log1p)", features, default=skewed,
                              format_func=interface.display_name,
                              help="Mặc định chọn sẵn các cột lệch mạnh (|skew| > 1).")
    d, e = st.columns(2)
    sample = d.number_input("Giới hạn số dòng (0 = giữ hết)", min_value=0, value=5000, step=500,
                            help="t-SNE chạy chậm khi n lớn. Lấy mẫu ở bước này để mọi bước sau cùng số dòng.")
    seed = e.number_input("Random state", min_value=0, value=int(interface.get(SK.RANDOM_STATE)), step=1)
    interface.put(SK.RANDOM_STATE, int(seed))

    if st.button("Chạy tiền xử lý và chuẩn hóa", type="primary", disabled=len(features) < 2):
        interface.reset_from("prep")
        prep = interface.preprocess_data(
            df, features, missing_strategy=missing, drop_duplicates=drop_dup,
            outlier_method=outlier, log_transform_columns=log_cols,
            sample_size=int(sample) or None, random_state=int(seed))
        scale = interface.scale_data(prep["X"])
        interface.put(SK.FEATURE_COLUMNS, features)
        interface.put(SK.PREP, prep)
        interface.put(SK.SCALE, scale)

    prep, scale = interface.get(SK.PREP), interface.get(SK.SCALE)
    if prep is None or scale is None:
        return
    Xs = np.asarray(scale["X_scaled"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số dòng trước", f"{prep['n_rows_before']:,}")
    m2.metric("Số dòng sau", f"{prep['n_rows_after']:,}")
    m3.metric("Số đặc trưng", Xs.shape[1])
    m4.metric("Đã chuẩn hóa", "Có")
    st.caption(f"Kiểm tra StandardScaler: |trung bình| lớn nhất = {np.abs(Xs.mean(axis=0)).max():.3f}, "
               f"độ lệch chuẩn từ {Xs.std(axis=0).min():.2f} đến {Xs.std(axis=0).max():.2f} (mong đợi 0 và 1).")
    with st.expander("Nhật ký tiền xử lý"):
        for line in prep["steps_log"] or ["Không có thay đổi."]:
            st.markdown(f"- {line}")


def _section_pca() -> None:
    st.subheader("2. PCA")
    mode = st.radio("Cách chọn số thành phần", ["Theo tỷ lệ phương sai", "Theo số thành phần"], horizontal=True)
    n_features = np.asarray(interface.get(SK.SCALE)["X_scaled"]).shape[1]
    if mode == "Theo tỷ lệ phương sai":
        n_comp = st.slider("Giữ tối thiểu phương sai", 0.70, 0.99, PCA_DEFAULT_VARIANCE, step=0.01, format="%.2f")
    else:
        n_comp = st.slider("Số thành phần", 2, max(2, n_features), min(3, n_features))
    if st.button("Chạy PCA"):
        interface.reset_from("pca")
        interface.put(SK.PCA, interface.run_pca(
            interface.matrix_for_space("scaled"), n_components=n_comp,
            random_state=interface.get(SK.RANDOM_STATE)))

    pca = interface.get(SK.PCA)
    if pca is None:
        return
    st.caption(f"Giữ {pca['n_components']} thành phần, tổng phương sai giải thích "
               f"{float(np.asarray(pca['cumulative_variance'])[-1]):.1%}.")
    var_df = interface.get_explained_variance(pca["pca_model"])
    st.plotly_chart(viz.plot_scree(var_df, target=n_comp if isinstance(n_comp, float) else None), width="stretch")


def _section_tsne() -> None:
    st.subheader("3. t-SNE")
    st.caption("t-SNE chỉ để quan sát bản đồ. Khoảng cách giữa các cụm xa nhau trên bản đồ không có ý nghĩa định lượng.")
    spaces = ["scaled"] + (["pca"] if interface.get(SK.PCA) is not None else [])
    default_idx = spaces.index("pca") if "pca" in spaces else 0
    space = st.radio("Dữ liệu đưa vào t-SNE", spaces, index=default_idx, horizontal=True,
                     format_func=lambda s: FEATURE_SPACES[s])
    n = len(interface.matrix_for_space(space))

    a, b, c = st.columns(3)
    perplexity = a.slider("Perplexity", 5, 50, int(TSNE_DEFAULTS["perplexity"]), step=1,
                          help="Số hàng xóm hiệu dụng của mỗi điểm.")
    lr = b.number_input("Learning rate", min_value=10, max_value=1000, value=int(TSNE_DEFAULTS["learning_rate"]), step=10)
    iters = c.number_input("Number of iterations", min_value=250, max_value=5000,
                           value=int(TSNE_DEFAULTS["n_iter"]), step=250)
    d, e = st.columns(2)
    want_3d = d.checkbox("Chạy thêm bản 3D", value=True)
    use_open = e.checkbox("Dùng openTSNE (hỗ trợ chiếu khách mới)", value=False)

    if perplexity >= n:
        st.error(f"Perplexity ({perplexity}) phải nhỏ hơn số dòng ({n}).")
        return
    if n > 10000:
        st.warning(f"{n:,} dòng: t-SNE sẽ rất chậm. Nên giảm 'Giới hạn số dòng' ở bước 1.")

    if st.button("Chạy t-SNE"):
        interface.reset_from("tsne")
        X = interface.matrix_for_space(space)
        seed = interface.get(SK.RANDOM_STATE)
        with st.spinner("Đang chạy t-SNE 2D..."):
            interface.put(SK.TSNE_2D, interface.run_tsne(
                X, n_components=2, perplexity=float(perplexity), learning_rate=float(lr),
                n_iter=int(iters), random_state=seed, use_opentsne=use_open))
        interface.put(SK.TSNE_3D, None)
        if want_3d:
            with st.spinner("Đang chạy t-SNE 3D..."):
                interface.put(SK.TSNE_3D, interface.run_tsne(
                    X, n_components=3, perplexity=float(perplexity), learning_rate=float(lr),
                    n_iter=int(iters), random_state=seed, use_opentsne=use_open))
        interface.put(SK.TSNE_INPUT_SPACE, space)

    t2 = interface.get(SK.TSNE_2D)
    if t2 is not None:
        kl = t2.get("kl_divergence")
        st.caption(f"t-SNE ({t2['backend']}) trên {FEATURE_SPACES[interface.get(SK.TSNE_INPUT_SPACE)].lower()}, "
                   f"perplexity {t2['perplexity']:g}"
                   + (f", KL = {kl:.3f}" if kl is not None else "") + ".")


def _section_visual() -> None:
    st.subheader("4. Biểu đồ giảm chiều")
    options = []
    if interface.get(SK.PCA) is not None:
        options.append("PCA")
    if interface.get(SK.TSNE_2D) is not None:
        options.append("t-SNE")
    if not options:
        st.info("Chạy PCA hoặc t-SNE ở trên để xem biểu đồ.")
        return
    method = st.radio("Phương pháp", options, horizontal=True)

    clusters = interface.get(SK.CLUSTER_RESULTS)
    color_options = ["Không tô màu"] + [ALGORITHMS[a] for a in clusters]
    choice = st.selectbox("Tô màu theo", color_options)
    labels = None
    if choice != "Không tô màu":
        key = next(k for k, v in ALGORITHMS.items() if v == choice)
        labels = clusters[key]["labels"]

    hover = interface.hover_dataframe()
    tab2, tab3 = st.tabs(["2D", "3D"])
    for tab, dims in ((tab2, 2), (tab3, 3)):
        with tab:
            if method == "t-SNE":
                t = interface.get(SK.TSNE_2D if dims == 2 else SK.TSNE_3D)
                coords = None if t is None else np.asarray(t["embedding"])
            else:
                x_pca = np.asarray(interface.get(SK.PCA)["X_pca"])
                coords = x_pca[:, :dims] if x_pca.shape[1] >= dims else None
            if coords is None:
                st.info("Chưa có dữ liệu cho chế độ này (PCA cần đủ thành phần; t-SNE 3D cần bật ô chạy 3D).")
                continue
            fn = viz.plot_scatter_2d if dims == 2 else viz.plot_scatter_3d
            st.plotly_chart(fn(coords, labels=labels, hover_df=hover, axis_prefix=method), width="stretch")


def render() -> None:
    st.title("Preprocessing & Dimensionality Reduction")
    df = interface.get(SK.RAW_DF)
    if not interface.require(SK.RAW_DF, message="Chưa có dữ liệu. Hãy sang trang Data Exploration để tải CSV."):
        return
    _section_preprocess(df, interface.get(SK.VALIDATION))
    if not interface.require(SK.SCALE, message="Chạy bước 1 để có dữ liệu chuẩn hóa trước khi dùng PCA và t-SNE."):
        return
    st.divider()
    _section_pca()
    st.divider()
    _section_tsne()
    st.divider()
    _section_visual()
