"""pages/tsne_scaling.py - Mở rộng: Barnes-Hut t-SNE và FIt-SNE (FFT) viết tay, kèm số liệu thực nghiệm."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from models import tsne_scalable as T

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tsne_scaling"


def _csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def render() -> None:
    st.title("Mở rộng: Barnes-Hut t-SNE và FIt-SNE")
    st.markdown(
        "t-SNE chính xác tốn **O(n²)** ở bước tính lực đẩy. Hai cách xấp xỉ, đều **viết tay** trong "
        "`models/tsne_scalable.py`, dùng chung ma trận P thưa, khởi tạo PCA và bộ tối ưu:\n"
        "- **Barnes-Hut**: quadtree, gộp các ô xa thành khối tâm theo tiêu chí θ → O(n log n).\n"
        "- **FIt-SNE**: nội suy Lagrange lên lưới đều + tích chập bằng FFT → O(n)."
    )
    rep, th, ip, bm, full = (_csv(f) for f in (
        "repulsion_scaling.csv", "theta_sweep.csv", "interpolation_sweep.csv", "benchmark.csv", "full_run_fft.csv"))
    if rep is None:
        st.warning("Chưa có số liệu. Chạy `python run_tsne_scaling_experiment.py` rồi tải lại trang.")
    else:
        st.subheader("1. Chi phí một lần tính lực đẩy theo n")
        fig = px.line(rep, x="n", y="seconds", color="method", markers=True, log_x=True, log_y=True)
        st.plotly_chart(fig, use_container_width=True)
        piv = rep.pivot(index="n", columns="method", values="seconds")
        if {"exact", "fft"} <= set(piv.columns):
            sp = (piv["exact"] / piv["fft"]).dropna()
            st.caption(f"Tại n = {int(sp.index[-1]):,}: FIt-SNE nhanh hơn exact {sp.iloc[-1]:.0f} lần.")
        if full is not None:
            st.success(f"FIt-SNE tự viết chạy trọn {int(full.n[0]):,} khách hàng trong {full.seconds[0]:.0f} giây.")
        c1, c2 = st.columns(2)
        if th is not None:
            c1.subheader("2. Đánh đổi θ (Barnes-Hut)")
            c1.plotly_chart(px.line(th.assign(rel_grad_error=th.rel_grad_error.clip(lower=1e-12)), x="theta",
                                    y="rel_grad_error", markers=True, log_y=True), use_container_width=True)
        if ip is not None:
            c2.subheader("3. Số điểm nội suy (FIt-SNE)")
            c2.plotly_chart(px.line(ip, x="n_interpolation_points", y="rel_grad_error", markers=True, log_y=True),
                            use_container_width=True)
        if bm is not None:
            st.subheader("4. Chất lượng bản đồ và thời gian toàn bộ quá trình")
            st.dataframe(bm.round(4), use_container_width=True)
            m = st.selectbox("Thước đo", ["runtime_s", "kl", "trustworthiness", "continuity", "knn_preservation",
                                          "kmeans_ari_on_map"])
            st.plotly_chart(px.line(bm.dropna(subset=[m]), x="n", y=m, color="method", markers=True,
                                    log_x=True, log_y=(m == "runtime_s")), use_container_width=True)
        for f, cap in (("maps.png", "Bản đồ cạnh nhau"), ("map_full.png", "Bản đồ toàn bộ dữ liệu")):
            if (RESULTS / f).exists():
                st.image(str(RESULTS / f), caption=cap)

    st.subheader("5. Chạy thử trực tiếp")
    n = st.slider("Số điểm (dữ liệu giả lập 5 cụm)", 500, 5000, 1500, 500)
    method = st.selectbox("Phương pháp", list(T.METHODS), index=1)
    if st.button("Chạy t-SNE"):
        rng = np.random.default_rng(0)
        k = np.arange(n) % 5
        X = rng.normal(0, 1, (n, 13)) + k[:, None] * 3.0
        with st.spinner("Đang chạy..."):
            r = T.fit_tsne(X, method, n_iter=300)
        st.write(f"Thời gian {r['timing']['total']:.1f}s, KL = {r['kl_divergence']:.3f}")
        st.plotly_chart(px.scatter(x=r["embedding"][:, 0], y=r["embedding"][:, 1], color=k.astype(str)),
                        use_container_width=True)
