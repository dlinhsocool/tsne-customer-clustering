"""pages/customer_profiling.py - Trang 6: đặc trưng và mô tả từng cụm."""
from __future__ import annotations

import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import ALGORITHMS, NOISE_LABEL, SK


def render() -> None:
    st.title("Customer Profiling")
    results = interface.get(SK.CLUSTER_RESULTS)
    if not interface.require(SK.PREP, message="Chưa có dữ liệu. Hãy chạy tiền xử lý trước."):
        return
    if not results:
        st.info("Chưa có kết quả phân cụm. Hãy chạy một thuật toán ở trang Clustering.")
        return

    algos = list(results)
    default = algos.index(interface.get(SK.ACTIVE_ALGO)) if interface.get(SK.ACTIVE_ALGO) in algos else 0
    algo = st.radio("Kết quả của thuật toán", algos, index=default, horizontal=True,
                    format_func=lambda k: ALGORITHMS[k])
    res = results[algo]
    prep = interface.get(SK.PREP)
    feats = prep["feature_columns"]
    include_noise = st.checkbox("Hiển thị cụm noise", value=res["n_noise"] > 0, disabled=res["n_noise"] == 0)

    store = interface.get(SK.PROFILES)
    if algo not in store or store[algo].get("include_noise") != include_noise:
        profiles = interface.calculate_cluster_profiles(prep["df_clean"], res["labels"], feats,
                                                        include_noise=include_noise)
        disp = {f: interface.display_name(f) for f in feats}
        descs = {cid: interface.generate_cluster_description(cid, profiles, feature_labels=disp)
                 for cid in profiles["cluster_ids"]}
        store[algo] = {"profiles": profiles, "descriptions": descs, "include_noise": include_noise}

    profiles = store[algo]["profiles"]
    descs = store[algo]["descriptions"]
    names = {cid: interface.cluster_label_for(algo, cid) for cid in profiles["cluster_ids"]}

    st.subheader("Mô tả từng cụm")
    st.caption("Mô tả được sinh từ số liệu của lần chạy này (so với trung bình toàn bộ), không phải nhãn gán sẵn. "
               "Bạn có thể đổi tên để dùng ở trang New Customer.")
    custom = interface.get(SK.CLUSTER_NAMES).setdefault(algo, {})
    for cid in profiles["cluster_ids"]:
        d = descs[cid]
        row = profiles["profile_table"].loc[cid]
        with st.expander(f"{d['name']}  ·  {int(row['size']):,} khách ({row['ratio']:.1%})", expanded=cid == profiles["cluster_ids"][0]):
            st.write(d["description"])
            new_name = st.text_input("Tên nghiệp vụ", value=custom.get(cid, ""), key=f"name_{algo}_{cid}",
                                     placeholder="Ví dụ: Khách VIP chi tiêu cao")
            if new_name.strip():
                custom[cid] = new_name.strip()
            else:
                custom.pop(cid, None)

    st.subheader("Giá trị trung bình theo cụm")
    table = profiles["profile_table"].copy()
    show = table.rename(columns={"size": "Số khách", "ratio": "Tỷ lệ", **{f: interface.display_name(f) for f in feats}})
    show.index = [names[i] for i in table.index]
    st.dataframe(show.style.format({"Tỷ lệ": "{:.1%}", **{interface.display_name(f): "{:,.2f}" for f in feats}}),
                 width="stretch")

    st.subheader("So sánh đặc trưng")
    kind = st.radio("Dạng biểu đồ", ["Radar", "Bar chart"], horizontal=True)
    z = profiles["z_scores"].rename(columns={f: interface.display_name(f) for f in feats})
    if kind == "Radar":
        lo, hi = z.min(), z.max()
        norm = (z - lo) / (hi - lo).replace(0, 1)
        st.plotly_chart(viz.plot_radar(norm, names), width="stretch")
        st.caption("Mỗi trục được co về [0, 1] giữa các cụm để so sánh; đây là vị trí tương đối, không phải giá trị tuyệt đối.")
    else:
        st.plotly_chart(viz.plot_profile_bars(z, names), width="stretch")
        st.caption("Trục tung là z-score: trung bình cụm lệch bao nhiêu độ lệch chuẩn so với trung bình toàn bộ.")
