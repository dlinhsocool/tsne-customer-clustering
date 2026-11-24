"""
app.py - điểm vào của ứng dụng. Chạy: streamlit run app.py
"""
import streamlit as st

from pages import (
    model_evaluation,
    clustering,
    customer_profiling,
    data_exploration,
    dimensionality_reduction,
    new_customer,
    overview,
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

navigation = st.navigation(
    [
        st.Page(overview.render, title="Overview", url_path="home", default=True),
        st.Page(data_exploration.render, title="Data Exploration", url_path="data"),
        st.Page(dimensionality_reduction.render, title="Preprocessing & t-SNE", url_path="reduction"),
        st.Page(clustering.render, title="Clustering", url_path="cluster"),
        st.Page(model_evaluation.render, title="Model Evaluation", url_path="evaluation"),
        st.Page(customer_profiling.render, title="Customer Profiling", url_path="profiling"),
        st.Page(new_customer.render, title="New Customer", url_path="new-customer"),
    ]
)

with st.sidebar:
    banner = st.empty()

try:
    navigation.run()
finally:
    with banner.container():
        interface.render_backend_banner()