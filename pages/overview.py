"""pages/overview.py - Trang 1: giới thiệu bài toán và pipeline."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.constants import APP_TITLE

_PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded", fontname="Helvetica", fontsize=11, color="#555555", margin="0.15,0.08"];
  edge [color="#888888"];
  a [label="Data\\nLoading"]; b [label="Preprocessing"]; c [label="Standard\\nScaler"];
  d [label="PCA"]; e [label="t-SNE\\n(2D / 3D)"]; f [label="K-Means\\nGMM\\nDBSCAN"];
  g [label="Evaluation"]; h [label="Profiling"];
  a -> b -> c -> d -> e; d -> f; c -> f [style=dashed];
  f -> g; f -> h; e -> h [style=dotted, arrowhead=none];
}
"""


def render() -> None:
    st.title(APP_TITLE)
    st.caption("Phân khúc hành vi khách hàng trong thương mại điện tử")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Bài toán")
        st.write(
            "Doanh nghiệp có hàng nghìn khách hàng, mỗi người để lại dấu vết hành vi: chi bao nhiêu, "
            "mua thường xuyên không, online bao lâu, hủy đơn nhiều hay ít. Không thể chăm sóc từng người, "
            "nhưng cũng không nên đối xử với tất cả như nhau. Phân khúc khách hàng chia tập khách thành "
            "vài nhóm giống nhau bên trong, khác nhau giữa các nhóm, để mỗi nhóm có chiến lược riêng."
        )
        st.write(
            "Đây là bài toán học không giám sát: dữ liệu không có nhãn đúng, thuật toán phải tự tìm "
            "cấu trúc. Mỗi khách hàng là một vector đặc trưng trong không gian nhiều chiều."
        )
    with right:
        st.subheader("Mục tiêu")
        st.markdown(
            "- Gán mỗi khách hàng vào một cụm.\n"
            "- Tạo bản đồ 2D/3D để quan sát và kiểm chứng các cụm.\n"
            "- Mô tả từng cụm bằng ngôn ngữ nghiệp vụ.\n"
            "- Gán cụm cho khách hàng mới chưa từng thấy."
        )

    st.subheader("Quy trình")
    st.graphviz_chart(_PIPELINE_DOT, width="stretch")
    st.caption("Phân cụm chạy trên dữ liệu đã chuẩn hóa hoặc sau PCA. t-SNE chỉ dùng để quan sát, không dùng để phân cụm.")

    steps = pd.DataFrame(
        [
            ("1", "Data Loading", "Đọc và kiểm tra dữ liệu CSV", "Data Exploration"),
            ("2", "Preprocessing", "Điền khuyết, loại ngoại lai, log-transform", "Preprocessing & Reduction"),
            ("3", "Standardization", "StandardScaler (lưu lại để dùng cho khách mới)", "Preprocessing & Reduction"),
            ("4", "Dimensionality Reduction", "PCA để tính toán, t-SNE để nhìn", "Preprocessing & Reduction"),
            ("5", "Clustering", "K-Means, GMM, DBSCAN", "Clustering"),
            ("6", "Evaluation", "Silhouette, Davies-Bouldin, ARI", "Model Evaluation"),
            ("7", "Visualization", "Plotly 2D và 3D, hover xem khách hàng", "Clustering"),
            ("8", "Customer Profiling", "Đặc trưng và mô tả từng cụm", "Customer Profiling"),
            ("9", "Out-of-sample", "Gán cụm và chiếu khách mới lên bản đồ", "New Customer"),
        ],
        columns=["Bước", "Tên", "Nội dung", "Trang"],
    )
    st.dataframe(steps, hide_index=True, width="stretch")
