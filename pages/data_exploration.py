"""pages/data_exploration.py - Trang 2: tải CSV và khám phá dữ liệu."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import SK


def _load_uploaded(upload, sep: str) -> None:
    try:
        df = interface.load_dataset(upload, sep=sep)
    except UnicodeDecodeError:
        upload.seek(0)
        try:
            df = interface.load_dataset(upload, sep=sep, encoding="utf-8-sig")
        except Exception:
            st.error("Không đọc được bảng mã của file. Hãy lưu file CSV dạng UTF-8 rồi tải lại.")
            return
    interface.reset_from("data")
    interface.put(SK.RAW_DF, df)
    interface.put(SK.IS_SAMPLE, False)
    interface.put(SK.VALIDATION, interface.validate_dataset(df))


def render() -> None:
    st.title("Data Exploration")

    c1, c2, c3 = st.columns([3, 1, 2])
    with c1:
        upload = st.file_uploader("Tải file CSV", type=["csv"])
    with c2:
        sep = st.selectbox("Ký tự phân tách", [",", ";", "\t", "|"], format_func=lambda s: {"\t": "Tab"}.get(s, s))
    with c3:
        st.write("")
        st.write("")
        use_sample = st.button("Dùng dữ liệu mẫu giả lập", width="stretch",
                               help="Chỉ để thử giao diện khi chưa có dữ liệu chuẩn.")

    if upload is not None:
        sig = (upload.name, upload.size, sep)
        if interface.get(SK.UPLOAD_SIG) != sig:
            try:
                _load_uploaded(upload, sep)
                interface.put(SK.UPLOAD_SIG, sig)
            except Exception as exc:  # file hỏng, sai định dạng...
                st.error(f"Không đọc được file: {exc}")
                return
    elif use_sample:
        interface.reset_from("data")
        df = interface.load_sample_dataset()
        interface.put(SK.RAW_DF, df)
        interface.put(SK.IS_SAMPLE, True)
        interface.put(SK.UPLOAD_SIG, None)
        interface.put(SK.VALIDATION, interface.validate_dataset(df))

    df = interface.get(SK.RAW_DF)
    if df is None:
        st.info("Tải file CSV hoặc dùng dữ liệu mẫu để bắt đầu.")
        return

    val = interface.get(SK.VALIDATION)
    for e in val["errors"]:
        st.error(e)
    for w in val["warnings"]:
        st.warning(w)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số dòng", f"{val['n_rows']:,}")
    m2.metric("Số cột", val["n_cols"])
    m3.metric("Dòng trùng", f"{val['n_duplicates']:,}")
    m4.metric("Giá trị thiếu", f"{sum(val['missing_per_column'].values()):,}")

    tab_prev, tab_stat, tab_miss, tab_dist, tab_corr = st.tabs(
        ["Xem trước", "Thống kê", "Giá trị thiếu", "Phân phối", "Tương quan"])

    with tab_prev:
        st.dataframe(df.head(200), width="stretch")
        st.caption(f"Hiển thị 200 dòng đầu trong {val['n_rows']:,} dòng.")

    with tab_stat:
        st.dataframe(df.describe(include="all").T, width="stretch")
        dtypes = pd.DataFrame({"Cột": list(val["dtypes"]), "Kiểu": list(val["dtypes"].values())})
        st.dataframe(dtypes, hide_index=True, width="stretch")

    numeric = val["numeric_columns"]
    with tab_miss:
        miss = pd.Series(val["missing_per_column"])
        if miss.sum() == 0:
            st.success("Không có giá trị thiếu.")
        else:
            st.plotly_chart(viz.plot_missing_bar(miss[miss > 0]), width="stretch")
        st.dataframe(miss.rename("Số giá trị thiếu").to_frame(), width="stretch")

    with tab_dist:
        if not numeric:
            st.info("Không có cột số.")
        else:
            a, b = st.columns([2, 1])
            col = a.selectbox("Đặc trưng", numeric, format_func=interface.display_name)
            bins = b.slider("Số cột histogram", 10, 80, 30)
            st.plotly_chart(viz.plot_histogram(df, col, bins), width="stretch")
            skew = float(df[col].skew())
            if abs(skew) > 1:
                st.caption(f"Độ lệch (skew) = {skew:.2f}: phân phối lệch mạnh, cân nhắc log-transform ở bước tiền xử lý.")
            else:
                st.caption(f"Độ lệch (skew) = {skew:.2f}.")

    with tab_corr:
        if len(numeric) < 2:
            st.info("Cần ít nhất 2 cột số.")
        else:
            chosen = st.multiselect("Cột tính tương quan", numeric, default=numeric[:12])
            if len(chosen) >= 2:
                st.plotly_chart(viz.plot_correlation_heatmap(df[chosen].corr()), width="stretch")
            else:
                st.info("Chọn ít nhất 2 cột.")
