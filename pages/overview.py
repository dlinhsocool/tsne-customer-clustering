"""pages/overview.py - Trang 1: Tổng quan kiến trúc hai luồng của đồ án."""
from __future__ import annotations

import streamlit as st

from utils import interface
from utils.constants import CONTRACT_VERSION, SK


def render() -> None:
    st.title("Customer Segmentation & Dimensionality Reduction Benchmark")
    st.caption(f"Kiến trúc hai luồng độc lập · Contract v{CONTRACT_VERSION}")

    st.markdown(
        """
        **Câu hỏi nghiên cứu.** Khi dữ liệu khách hàng có nhiều chiều (trong đó có chiều nhiễu), giảm chiều bằng
        **t-SNE** (so với **PCA**, **UMAP** và dữ liệu gốc) có giúp K-Means, GMM, DBSCAN tìm đúng nhóm khách hàng hơn không?
        Đo bằng **ARI so với nhóm thật** trên bộ mô phỏng 5 nhóm, và silhouette đo lại trên không gian gốc để so sánh công bằng.
        """
    )
    with st.expander("Giới hạn cần biết khi đọc kết quả", expanded=False):
        st.markdown(
            """
            - **t-SNE chỉ để quan sát.** Khoảng cách giữa các cụm và kích thước cụm trên bản đồ không có nghĩa định lượng;
              kết quả phụ thuộc perplexity và seed. t-SNE không có phép chiếu cố định cho điểm mới.
            - **File gốc 100.000 dòng là dữ liệu ngẫu nhiên độc lập** (Hopkins ≈ 0,50 trên bộ RFM): không có cụm thật.
              Silhouette trong không gian 2D vẫn cao trên dữ liệu như vậy, nên không dùng nó làm bằng chứng.
            - **Bộ mô phỏng 5 nhóm** có cụm thật; kết quả giảm chiều có lợi hay không phụ thuộc vào số chiều nhiễu
              và thuật toán, không phải lúc nào t-SNE cũng thắng.
            """
        )

    st.markdown(
        """
        Hệ thống được thiết kế thành **hai luồng xử lý tách biệt** nhằm giải quyết đồng thời bài toán
        **ứng dụng phân khúc khách hàng thực tế** và bài toán **thực nghiệm so sánh khoa học** giữa các không gian giảm chiều:
        """
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("1. Luồng ứng dụng chính (Production Pipeline)")
        st.info(
            "Phân cụm trực tiếp trên **dữ liệu gốc đã chuẩn hóa (`X_scaled`)** để bảo toàn toàn bộ "
            "thông tin đặc trưng; **t-SNE** chỉ đóng vai trò ánh xạ 2D/3D phục vụ trực quan hóa."
        )
        st.code(
            """
CSV thô (20 cột hoặc bộ mô phỏng 5 nhóm)
   │
   ▼
Preprocessing (ngày -> recency_days với ngày tham chiếu cố định,
               n_orders = chi tiêu / giá trị đơn, subscription 0/1, One-Hot danh mục)
   │
   ▼
StandardScaler  ──►  X_scaled ("original")
   │
   ├──► Clustering (K-Means / GMM / DBSCAN) ──► Nhãn cụm chính
   │                                                 │
   └──► t-SNE (2D / 3D Embedding) ──► Trực quan hóa ◄┘
                                           │
                                           ▼
                         Customer Profiling & New Customer
            """.strip(),
            language="text",
        )

    with col2:
        st.subheader("2. Luồng thực nghiệm (Benchmark Pipeline)")
        st.warning(
            "So sánh công bằng giữa **Original (`X_scaled`)**, **PCA**, **UMAP** và **t-SNE** "
            "trên cùng tập mẫu, tách bạch thời gian giảm chiều và thời gian phân cụm."
        )
        st.code(
            """
X_scaled (cùng tập mẫu thực nghiệm)
   │
   ├──► Original (d chiều)
   ├──► PCA (giữ 95% phương sai hoặc 2D)
   ├──► UMAP (5D hoặc 2D)
   └──► t-SNE (2D)
           │
           ▼
   K-Means / GMM / DBSCAN (4 × 3 = 12 cấu hình)
           │
           ▼
   Đánh giá kép:
   • Trên không gian chạy (silhouette_space, DB_space)
   • Trên dữ liệu gốc chuẩn hóa (silhouette_original, DB_original)
   • Phân tách thời gian: T_total = T_reduction + T_clustering
            """.strip(),
            language="text",
        )

    st.divider()
    st.subheader("Trạng thái phiên làm việc hiện tại")
    raw_df = interface.get(SK.RAW_DF)
    prep = interface.get(SK.PREP)
    scale = interface.get(SK.SCALE)
    clusters = interface.get(SK.CLUSTER_RESULTS, {})
    bench_tbl = interface.get(SK.BENCHMARK_CLUSTER_TABLE)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dữ liệu gốc", "Chưa tải" if raw_df is None else f"{len(raw_df):,} dòng")
    m2.metric("Sau tiền xử lý", "Chưa chạy" if prep is None else f"{prep['n_rows_after']:,} × {len(prep['feature_columns'])}")
    m3.metric("Phân cụm chính", f"{len(clusters)} / 3 thuật toán")
    m4.metric("Benchmark 4×3", "Chưa chạy" if bench_tbl is None else f"{len(bench_tbl)} cấu hình")

    if scale is None:
        st.caption("Bắt đầu bằng cách mở trang **Data Exploration** để tải file `synthetic_customers_11cols.csv`.")