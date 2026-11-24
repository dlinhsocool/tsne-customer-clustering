"""
app.py - điểm vào của ứng dụng.  Chạy:  streamlit run app.py

File này chỉ cấu hình trang, khởi tạo session state và điều hướng. Mọi xử lý nằm ở
pages/ (giao diện), utils/interface.py (lớp trung gian) và models/ (backend).
"""
import streamlit as st

from pages import (
    clustering, customer_profiling, data_exploration, dimensionality_reduction,
    model_evaluation, new_customer, overview,
)
from utils import interface
from utils.constants import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
interface.init_state()

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1280px;}
    h1 {font-weight: 800; letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# LƯU Ý: url_path KHÔNG được trùng tên file trong pages/ (ví dụ "overview", "clustering").
# Nếu trùng, mở thẳng URL đó Streamlit sẽ chạy file trang một mình và ra trang trắng.
navigation = st.navigation(
    [
        st.Page(overview.render, title="Overview", url_path="home", default=True),
        st.Page(data_exploration.render, title="Data Exploration", url_path="data"),
        st.Page(dimensionality_reduction.render, title="Preprocessing & Reduction", url_path="reduction"),
        st.Page(clustering.render, title="Clustering", url_path="cluster"),
        st.Page(model_evaluation.render, title="Model Evaluation", url_path="evaluation"),
        st.Page(customer_profiling.render, title="Customer Profiling", url_path="profiling"),
        st.Page(new_customer.render, title="New Customer", url_path="new-customer"),
    ]
)

with st.sidebar:
    banner = st.empty()  # điền sau khi trang chạy xong để trạng thái luôn mới nhất

try:
    navigation.run()
finally:
    with banner.container():
        interface.render_backend_banner()
