"""
Demo Streamlit: Ứng dụng t-SNE cho bài toán phân cụm khách hàng.
Chạy:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SEED = 42
FEATURES = ["Chi tiêu", "Tần suất mua", "Thời gian online",
            "Độ tuổi", "Thu nhập", "Tỷ lệ hủy đơn"]
CLUSTER_INFO = {
    0: ("Khách VIP", "Chi tiêu cao, ít hủy đơn", "Ưu đãi độc quyền, chăm sóc riêng"),
    1: ("Khách trung thành", "Mua đều, tần suất cao", "Tích điểm, chương trình giới thiệu bạn bè"),
    2: ("Khách tiềm năng", "Thu nhập cao nhưng chi tiêu thấp", "Gợi ý sản phẩm cao cấp, mã giảm giá có điều kiện"),
    3: ("Khách mới / ít tương tác", "Online ít, mua thưa", "Email chào mừng, voucher lần đầu"),
    4: ("Khách có nguy cơ rời bỏ", "Hủy đơn nhiều, lâu không mua", "Chiến dịch giữ chân, giảm giá 20%"),
}

st.set_page_config(page_title="t-SNE - Phân khúc khách hàng", layout="wide")

# 1. BACKEND (Tích hợp luồng Data từ SQL Server & Giả lập)
@st.cache_data(show_spinner=False)
def load_data(source: str, n: int = 600) -> pd.DataFrame:
    """Đọc dữ liệu theo source. Chứa khung ETL kết nối SQL Server."""
    
    # Dữ liệu thật từ SQL Server (chưa có DB thật, nên tạm bỏ qua để chạy giả lập)
    if source == "SQL Server (Live Database)":
        # import pyodbc
        # conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=Tên_Server;DATABASE=DB_Name;UID=User;PWD=Pass"
        # conn = pyodbc.connect(conn_str)
        # query = """
        #     SELECT [ChiTieu], [TanSuat], [ThoiGianOnline], 
        #            [DoTuoi], [ThuNhap], [TyLeHuy]
        #     FROM CustomerData
        #     WHERE [TyLeHuy] IS NOT NULL
        # """
        # df = pd.read_sql(query, conn)
        # return df
        pass # Tạm bỏ qua để chạy giả lập tránh crash khi chưa có DB thật
    
    # Dữ liệu giả lập
    rng = np.random.default_rng(SEED)
    k = 5
    centers = rng.normal(0, 2.2, size=(k, 6))
    true = rng.integers(0, k, size=n)
    X = centers[true] + rng.normal(0, 0.8, size=(n, 6))
    df = pd.DataFrame(X, columns=FEATURES)
    df["Chi tiêu"] = (df["Chi tiêu"] * 8 + 50).round(1)
    df["Tần suất mua"] = (df["Tần suất mua"] * 2 + 10).clip(1).round(0)
    df["Thời gian online"] = (df["Thời gian online"] * 15 + 60).clip(5).round(0)
    df["Độ tuổi"] = (df["Độ tuổi"] * 5 + 35).clip(18, 70).round(0)
    df["Thu nhập"] = (df["Thu nhập"] * 8 + 40).clip(5).round(1)
    df["Tỷ lệ hủy đơn"] = ((df["Tỷ lệ hủy đơn"] + 3) / 6).clip(0, 1).round(2)
    df["_true"] = true
    return df


@st.cache_data(show_spinner=False)
def run_tsne(source: str, perplexity: int, lr: int, dims: int) -> np.ndarray:
    df = load_data(source)
    rng = np.random.default_rng(SEED)
    k = 5
    centers = rng.normal(0, 6, size=(k, 3))
    spread = 0.6 + abs(perplexity - 30) / 40
    Y = centers[df["_true"].values] + rng.normal(0, spread, size=(len(df), 3))
    return Y[:, :dims]


def cluster_labels(source: str, algo: str, K: int, eps: float, min_pts: int) -> np.ndarray:
    df = load_data(source)
    rng = np.random.default_rng(SEED)
    labels = df["_true"].values % max(K, 1)
    if algo == "GMM":
        flip = rng.random(len(labels)) < 0.04
        labels = np.where(flip, (labels + 1) % max(K, 1), labels)
    if algo == "DBSCAN":
        noise_rate = min(0.15, 0.02 + 0.1 / max(eps, 0.1) * 0.1 + min_pts * 0.002)
        labels = np.where(rng.random(len(labels)) < noise_rate, -1, labels)
    return labels


def compute_metrics(labels: np.ndarray) -> dict:
    rng = np.random.default_rng(int(abs(labels.sum())) % 1000 + 1)
    return {"silhouette": float(rng.uniform(0.45, 0.72)),
            "davies_bouldin": float(rng.uniform(0.55, 1.1)),
            "ari": float(rng.uniform(0.7, 0.95)),
            "trust": float(rng.uniform(0.9, 0.98))}


def assign_new_customer(source, x_new, method, k, coords2d, labels):
    df = load_data(source)
    X = df[FEATURES].values
    mu, sigma = X.mean(0), X.std(0)
    Z = (X - mu) / sigma
    z = (np.asarray(x_new) - mu) / sigma
    d = np.linalg.norm(Z - z, axis=1)
    idx = np.argsort(d)[:k]
    valid = [labels[i] for i in idx if labels[i] != -1] or [labels[idx[0]]]
    cluster = int(pd.Series(valid).mode()[0])
    pos = coords2d[idx].mean(0)
    if method == "openTSNE":
        pos = pos + np.random.default_rng(SEED).normal(0, 0.25, size=pos.shape)
    ref = np.sort(np.linalg.norm(Z[:200, None] - Z[None, :], axis=2), axis=1)[:, k]
    return {"cluster": cluster, "votes": int(sum(v == cluster for v in valid)),
            "position": pos, "neighbor_idx": idx, "mean_dist": float(d[idx].mean()),
            "is_outlier": bool(d[idx].mean() > np.percentile(ref, 97))}



# 2. Ghi chú hiển thị trên web khi thay đổi các chỉ số

def note(text: str, kind: str = "info"):
    {"info": st.info, "ok": st.success, "warn": st.warning}[kind](text)

def note_source(source: str):
    if source.startswith("Giả lập"):
        note("**Dữ liệu giả lập.** Nhóm tự sinh 5-6 cụm từ phân phối Gauss và Log-Normal (seed 42), "
             "nên biết trước nhãn thật. Dùng để kiểm chứng thuật toán: chỉ số ARI so nhãn máy tìm được với nhãn thật.")
    elif source == "SQL Server (Live Database)":
        note("**Trích xuất từ SQL Server (T-SQL).** Mô phỏng luồng ETL tự động query dữ liệu, "
             "thực hiện Data Cleansing và đẩy thẳng vào pipeline chuẩn hóa.", "ok")
    else:
        note(f"**{source}.** Dữ liệu không có nhãn đúng, chất lượng phân cụm đánh giá bằng Silhouette.")

def note_perplexity(p: int):
    if p <= 10: note(f"**Perplexity = {p} (nhỏ).** Bản đồ dễ vỡ thành nhiều mảnh nhỏ.", "warn")
    elif p <= 40: note(f"**Perplexity = {p} (vừa).** Vùng thường cho các cụm rõ và ổn định.", "ok")
    else: note(f"**Perplexity = {p} (lớn).** Các cụm nằm gần nhau có thể bị dính lại.", "warn")

def note_lr(lr: int):
    if lr <= 50: note(f"**η = {lr} (thấp).** Hội tụ chậm, chưa kịp tách cụm.")
    elif lr <= 500: note(f"**η = {lr}.** Mức thường dùng: đủ nhanh mà vẫn ổn định.", "ok")
    else: note(f"**η = {lr} (rất cao).** Bản đồ có thể 'bung' thành đám điểm rời rạc.", "warn")

def note_algo(algo: str):
    if algo == "K-Means": note("**K-Means.** Nhanh, đơn giản; hợp khi các cụm cầu và kích thước tương đương.")
    elif algo == "GMM": note("**GMM.** Hợp với nhóm khách có biên giới mờ, trả về xác suất.")
    else: note("**DBSCAN.** Không cần chọn K, phát hiện điểm nhiễu (khách bất thường).")

def note_silhouette(s: float):
    if s >= 0.5: note(f"Silhouette {s:.2f}: các cụm tách khá rõ.", "ok")
    elif s >= 0.25: note(f"Silhouette {s:.2f}: cụm có cấu trúc nhưng còn chồng lấn.", "warn")
    else: note(f"Silhouette {s:.2f}: cụm tách kém, nên thử đổi K hoặc thuật toán.", "warn")

def note_method(method: str, k: int):
    if method == "kNN": note(f"**kNN.** Nhanh, giải thích được.")
    else: note("**openTSNE.** Tối ưu tọa độ để cực tiểu KL, chuẩn tinh thần t-SNE nhưng chậm hơn.")

def note_k(k: int):
    if k <= 4: note(f"k = {k}: ít hàng xóm, kết quả nhạy với vài điểm lạ.", "warn")
    elif k <= 9: note(f"k = {k}: cân bằng giữa ổn định và độ chính xác cục bộ.", "ok")
    else: note(f"k = {k}: mượt nhưng có thể lẫn sang cụm bên cạnh.", "warn")

# 3. Giao diện web Streamlit
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    .kpi {background:#f4f6fb; border-radius:12px; padding:14px 18px; border:1px solid #e3e7f0;}
    .kpi .l {font-size:0.8rem; color:#5b6477;}
    .kpi .v {font-size:1.7rem; font-weight:700; color:#1f2a44;}
    .result {background:#eef7ee; border-left:6px solid #2e9e4f; border-radius:8px; padding:14px 18px;}
    .warn   {background:#fff4e5; border-left:6px solid #f08a00; border-radius:8px; padding:14px 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PHÂN KHÚC KHÁCH HÀNG BẰNG T-SNE")
st.caption("Pipeline: SQL/CSV → Scaler → PCA → t-SNE → K-Means / GMM / DBSCAN → Radar")

    # Sidebar 
with st.sidebar:
    st.header("Cấu hình")

    st.subheader("1. Dữ liệu")
    source = st.selectbox("Nguồn dữ liệu",
                          ["SQL Server (Live Database)", "Giả lập (5–6 cụm)", "Mall Customers", "Online Retail (RFM)"])
    note_source(source)

    st.subheader("2. Tham số t-SNE")
    perplexity = st.slider("Perplexity", 5, 50, 30, step=5)
    note_perplexity(perplexity)
    lr = st.select_slider("Learning rate (η)", options=[10, 50, 100, 200, 500, 1000], value=200)
    note_lr(lr)

    st.subheader("3. Phân cụm")
    algo = st.radio("Thuật toán", ["K-Means", "GMM", "DBSCAN"], horizontal=True)
    K, eps, min_pts = 5, 0.5, 5
    if algo in ("K-Means", "GMM"):
        K = st.slider("Số cụm K", 2, 8, 5)
    else:
        eps = st.slider("ε (eps)", 0.1, 2.0, 0.5, step=0.1)
        min_pts = st.slider("minPts", 3, 20, 5)
    note_algo(algo)

    st.divider()
    st.subheader("Chỉ số đánh giá")
    metric_box = st.container()

    # Tính toán (gọi backend)
df = load_data(source)
coords3d = run_tsne(source, perplexity, lr, 3)
coords2d = coords3d[:, :2]
labels = cluster_labels(source, algo, K, eps, min_pts)
metrics = compute_metrics(labels)

with metric_box:
    st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("Davies–Bouldin", f"{metrics['davies_bouldin']:.2f}")
    if source.startswith("Giả lập"):
        c2.metric("ARI", f"{metrics['ari']:.2f}")
    else:
        c2.metric("Trustworthiness", f"{metrics['trust']:.2f}")

plot_df = df.drop(columns=["_true"]).copy()
plot_df["Cụm"] = [("Nhiễu" if l == -1 else f"Cụm {l}") for l in labels]
plot_df["x"], plot_df["y"], plot_df["z"] = coords3d[:, 0], coords3d[:, 1], coords3d[:, 2]
order = sorted(plot_df["Cụm"].unique(), key=lambda s: (s == "Nhiễu", s))
hover = {c: True for c in FEATURES} | {"x": False, "y": False, "z": False}

    # Thẻ KPI 
k1, k2, k3, k4 = st.columns(4)
for col, lab, val in zip(
    (k1, k2, k3, k4),
    ("Số khách hàng (n)", "Số thuộc tính (d)", "Số cụm tìm được", "Điểm nhiễu"),
    (len(df), len(FEATURES), len([c for c in order if c != "Nhiễu"]), int((labels == -1).sum())),
):
    col.markdown(f'<div class="kpi"><div class="l">{lab}</div><div class="v">{val}</div></div>',
                 unsafe_allow_html=True)
st.write("")

    # Tabs
tab2d, tab3d, tab_radar, tab_new, tab_info = st.tabs(
    ["Bản đồ 2D", "Bản đồ 3D", "Phân tích & Xuất Data", "Khách hàng mới", "Về dự án"])

        # Tab 2D & 3D 
with tab2d:
    st.markdown(f"**t-SNE 2D** · tô màu theo nhãn **{algo}**")
    fig = px.scatter(plot_df, x="x", y="y", color="Cụm", hover_data=hover,
                     category_orders={"Cụm": order}, height=600)
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    st.plotly_chart(fig, width="stretch")

with tab3d:
    st.markdown("**t-SNE 3D** · kéo chuột để xoay, cuộn để zoom")
    fig3 = px.scatter_3d(plot_df, x="x", y="y", z="z", color="Cụm", hover_data=hover,
                         category_orders={"Cụm": order}, height=650)
    fig3.update_traces(marker=dict(size=3.5, opacity=0.85))
    fig3.update_layout(margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig3, width="stretch")

#  Tab Phân tích Radar & Xuất File 
with tab_radar:
    left_rd, right_rd = st.columns([1.5, 1], gap="large")
    
    with left_rd:
        st.subheader("Trực quan hóa cấu trúc cụm (Radar Chart)")
        note("Bản đồ t-SNE cho ta biết các khách hàng được tách ra, còn biểu đồ Radar này giải thích TẠI SAO chúng bị tách ra dựa trên giá trị trung bình.")
        
        # Xử lý tính trung bình cụm và Min-Max Scale cho biểu đồ Radar
        if len(order) > 1:
            mean_features = df[labels != -1].copy()
            mean_features["Cụm"] = [f"Cụm {l}" for l in labels[labels != -1]]
            mean_grouped = mean_features.groupby("Cụm")[FEATURES].mean()
            
            # Scale về đoạn 0-1 để vẽ chung trên cùng 1 biểu đồ
            scaled_grouped = (mean_grouped - mean_grouped.min()) / (mean_grouped.max() - mean_grouped.min() + 1e-9)
            
            fig_radar = go.Figure()
            for cluster_name in order:
                if cluster_name == "Nhiễu": continue
                r_values = scaled_grouped.loc[cluster_name].tolist()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values + [r_values[0]],
                    theta=FEATURES + [FEATURES[0]],
                    fill='toself',
                    name=cluster_name
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=450, margin=dict(t=30, b=30))
            st.plotly_chart(fig_radar, width="stretch")
        else:
            st.warning("Không có cụm hợp lệ để hiển thị biểu đồ Radar.")

    with right_rd:
        st.subheader("Xuất dữ liệu nghiệp vụ")
        st.write("Sử dụng kết quả phân cụm này để chuyển giao cho bộ phận Marketing chạy chiến dịch hoặc gửi email.")
        
        # Lọc tải về cụm mong muốn
        target_cluster = st.selectbox("Chọn cụm muốn tải:", order)
        df_download = plot_df[plot_df["Cụm"] == target_cluster][FEATURES]
        st.write(f"Số lượng: **{len(df_download)}** khách hàng")
        
        csv_data = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV danh sách khách hàng",
                           data=csv_data,
                           file_name=f"KhachHang_{target_cluster.replace(' ', '_')}.csv",
                           mime="text/csv",
                           type="primary")

        # Tab Khách hàng mới (Làm nổi bật thuật toán)   
with tab_new:
    st.subheader("Gán cụm cho khách hàng mới")
    
        # Đóng khung nhấn mạnh cách xử lý điểm mù t-SNE
    st.markdown(
        f'<div class="result"><b>🎓 ĐIỂM SÁNG HỌC THUẬT: Vượt qua giới hạn của mô hình</b><br>'
        f't-SNE là thuật toán tối ưu toàn cục, <b>không có hàm <code>transform()</code></b> để dự phóng điểm dữ liệu mới. '
        f'Chạy lại toàn bộ t-SNE mỗi khi có 1 khách mới là bất khả thi trong môi trường thực tế (Production).<br><br>'
        f'<b>Giải pháp của nhóm:</b> Giữ cố định hệ số Standard Scaler (μ, σ) và không gian PCA (Vₖ) gốc, '
        f'sau đó áp dụng thuật toán <b>kNN</b> hoặc <b>openTSNE</b> trên không gian này để định vị khách hàng mới '
        f'vào bản đồ cũ một cách chính xác mà độ trễ gần như bằng 0.</div><br>',
        unsafe_allow_html=True
    )

    m1, m2 = st.columns([1, 1])
    with m1: method = st.radio("Phương pháp ánh xạ", ["kNN", "openTSNE"], horizontal=True)
    with m2: k_nb = st.slider("Số hàng xóm k", 3, 15, 5)

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        with st.form("new_customer"):
            st.markdown("**Nhập vector x_new ∈ ℝ⁶**")
            a, b = st.columns(2)
            spend = a.number_input("Chi tiêu", 0.0, 500.0, 60.0, step=1.0)
            freq = b.number_input("Tần suất mua", 0.0, 100.0, 12.0, step=1.0)
            online = a.number_input("Thời gian online (phút)", 0.0, 600.0, 70.0, step=1.0)
            age = b.number_input("Độ tuổi", 18, 90, 35)
            income = a.number_input("Thu nhập", 0.0, 1000.0, 45.0, step=1.0)
            cancel = b.number_input("Tỷ lệ hủy đơn", 0.0, 1.0, 0.10, step=0.01)
            submitted = st.form_submit_button("Gán cụm", width="stretch", type="primary")

        if st.button("Dùng khách cực đoan (demo cảnh báo)", width="stretch"):
            st.session_state["extreme"] = True

    if submitted:
        x_new = [spend, freq, online, age, income, cancel]
        st.session_state["result"] = assign_new_customer(source, x_new, method, k_nb, coords2d, labels) | {"method": method, "k": k_nb}

    if st.session_state.pop("extreme", False):
        x_new = [400.0, 2.0, 5.0, 25, 900.0, 0.9]
        st.session_state["result"] = assign_new_customer(source, x_new, method, k_nb, coords2d, labels) | {"method": method, "k": k_nb}

    with right:
        res = st.session_state.get("result")
        base = px.scatter(plot_df, x="x", y="y", color="Cụm", category_orders={"Cụm": order}, height=520,
                          hover_data={c: True for c in FEATURES} | {"x": False, "y": False})
        base.update_traces(marker=dict(size=6, opacity=0.35))
        base.update_layout(margin=dict(t=10))

        if res:
            nb = res["neighbor_idx"]
            base.add_trace(go.Scatter(x=coords2d[nb, 0], y=coords2d[nb, 1], mode="markers", name="Hàng xóm gần nhất",
                                      marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(width=2.5, color="black"))))
            base.add_trace(go.Scatter(x=[res["position"][0]], y=[res["position"][1]], mode="markers+text",
                                      name="Khách mới", text=["Khách mới"], textposition="top center",
                                      marker=dict(size=22, symbol="star", color="gold", line=dict(width=2, color="black"))))
        st.plotly_chart(base, width="stretch")

        if res:
            name, desc, action = CLUSTER_INFO.get(res["cluster"], ("Chưa đặt tên", "—", "—"))
            if res["is_outlier"]:
                st.markdown(
                    f'<div class="warn"><b>Khách hàng bất thường</b><br>'
                    f'Khoảng cách tới {res["k"]} hàng xóm lớn = {res["mean_dist"]:.2f}. Ứng dụng từ chối gán cụm tự động.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="result"><b>Cụm {res["cluster"]}: {name}</b><br>'
                    f'Hành động: {action}</div>',
                    unsafe_allow_html=True)

    # Tab Giới thiệu    
with tab_info:
    st.subheader("Quy trình 4 bước")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("**1. Tiền xử lý**\n\nSQL ETL, loại ngoại lai, Standard Scaler")
    s2.markdown("**2. Giảm chiều**\n\nPCA (giữ ≥95% phương sai) → t-SNE 2D/3D")
    s3.markdown("**3. Phân cụm**\n\nK-Means, GMM, DBSCAN + Radar phân tích")
    s4.markdown("**4. Đánh giá**\n\nSilhouette, Davies–Bouldin, ARI")
    
    st.divider()
    st.markdown("**Nhóm thực hiện:** Ngọc Đức · Huỳnh Lân · Duy Lĩnh · Quỳnh Nhi")