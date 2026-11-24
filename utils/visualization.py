"""
utils/visualization.py
======================
Các hàm vẽ biểu đồ Plotly. Mỗi hàm nhận dữ liệu và TRẢ VỀ plotly Figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.constants import ALGORITHMS, NEW_POINT_COLOR, NOISE_COLOR, NOISE_LABEL, PALETTE

_FONT = dict(family="sans-serif", size=14, color="#1F1F1F")
_SPACE_NAMES = {
    "original": "Original (X_scaled)",
    "pca": "PCA",
    "umap": "UMAP",
    "tsne": "t-SNE",
}


def _style(fig: go.Figure, height: int = 480, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=_FONT,
        title=dict(text=title, x=0, font=dict(size=18)) if title else None,
        margin=dict(l=12, r=12, t=44 if title else 16, b=12),
        legend=dict(title=None),
    )
    return fig


def cluster_names(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    return np.array(["Noise" if v == NOISE_LABEL else f"Cluster {v}" for v in labels])


def color_map(labels: np.ndarray) -> dict[str, str]:
    ids = sorted(set(np.asarray(labels).astype(int).tolist()))
    cmap = {}
    for i in [v for v in ids if v != NOISE_LABEL]:
        cmap[f"Cluster {i}"] = PALETTE[i % len(PALETTE)]
    if NOISE_LABEL in ids:
        cmap["Noise"] = NOISE_COLOR
    return cmap


def _order(labels: np.ndarray) -> list[str]:
    ids = sorted(set(np.asarray(labels).astype(int).tolist()), key=lambda v: (v == NOISE_LABEL, v))
    return ["Noise" if v == NOISE_LABEL else f"Cluster {v}" for v in ids]


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    fig = px.histogram(df, x=column, nbins=bins, color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(yaxis_title="Số khách", bargap=0.04)
    return _style(fig, 320)


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    return _style(fig, max(380, 55 * len(corr)))


def plot_missing_bar(missing: pd.Series) -> go.Figure:
    fig = px.bar(x=missing.index, y=missing.values, color_discrete_sequence=[PALETTE[1]])
    fig.update_layout(xaxis_title=None, yaxis_title="Số giá trị thiếu")
    return _style(fig, 300)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------
def plot_scree(var_df: pd.DataFrame, target: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=var_df["component"], y=var_df["explained_variance_ratio"],
        name="Từng thành phần", marker_color=PALETTE[0],
    )
    fig.add_scatter(
        x=var_df["component"], y=var_df["cumulative_variance"],
        name="Tích lũy", mode="lines+markers", line=dict(color="#1F1F1F"),
    )
    if target:
        fig.add_hline(
            y=target, line_dash="dot", line_color="#888",
            annotation_text=f"{target:.0%}", annotation_position="bottom right",
        )
    fig.update_layout(
        xaxis_title="Thành phần chính", yaxis_title="Tỷ lệ phương sai",
        yaxis_tickformat=".0%", xaxis=dict(dtick=1),
    )
    return _style(fig, 340)


def plot_scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray | None = None,
    hover_df: pd.DataFrame | None = None,
    title: str | None = None,
    axis_prefix: str = "Dim",
    new_point: np.ndarray | None = None,
    neighbor_idx: np.ndarray | None = None,
    new_point_label: str = "Khách mới",
) -> go.Figure:
    df = pd.DataFrame(coords[:, :2], columns=["x", "y"])
    hover_cols: list[str] = []
    if hover_df is not None:
        hover_df = hover_df.reset_index(drop=True)
        df = pd.concat([df, hover_df], axis=1)
        hover_cols = list(hover_df.columns)
    if labels is not None:
        df["Cụm"] = cluster_names(labels)
        fig = px.scatter(
            df, x="x", y="y", color="Cụm", hover_data=hover_cols,
            color_discrete_map=color_map(labels),
            category_orders={"Cụm": _order(labels)},
        )
    else:
        fig = px.scatter(df, x="x", y="y", hover_data=hover_cols, color_discrete_sequence=[PALETTE[0]])
    has_new = new_point is not None
    fig.update_traces(marker=dict(size=6, opacity=0.35 if has_new else 0.8), selector=dict(mode="markers"))
    if has_new and neighbor_idx is not None:
        nb = np.asarray(neighbor_idx).astype(int)
        fig.add_trace(go.Scatter(
            x=coords[nb, 0], y=coords[nb, 1], mode="markers", name="Hàng xóm gần nhất",
            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=2, color=NEW_POINT_COLOR)),
            hoverinfo="skip",
        ))
    if has_new:
        p = np.asarray(new_point).reshape(-1)
        fig.add_trace(go.Scatter(
            x=[p[0]], y=[p[1]], mode="markers+text", name=new_point_label,
            text=[new_point_label], textposition="top center",
            marker=dict(size=18, symbol="star", color="#FFD400", line=dict(width=2, color=NEW_POINT_COLOR)),
        ))
    fig.update_layout(xaxis_title=f"{axis_prefix} 1", yaxis_title=f"{axis_prefix} 2")
    return _style(fig, 540, title)


def plot_scatter_3d(
    coords: np.ndarray,
    labels: np.ndarray | None = None,
    hover_df: pd.DataFrame | None = None,
    title: str | None = None,
    axis_prefix: str = "Dim",
) -> go.Figure:
    df = pd.DataFrame(coords[:, :3], columns=["x", "y", "z"])
    hover_cols: list[str] = []
    if hover_df is not None:
        hover_df = hover_df.reset_index(drop=True)
        df = pd.concat([df, hover_df], axis=1)
        hover_cols = list(hover_df.columns)
    if labels is not None:
        df["Cụm"] = cluster_names(labels)
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color="Cụm", hover_data=hover_cols,
            color_discrete_map=color_map(labels),
            category_orders={"Cụm": _order(labels)},
        )
    else:
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", hover_data=hover_cols,
            color_discrete_sequence=[PALETTE[0]],
        )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(scene=dict(
        xaxis_title=f"{axis_prefix} 1",
        yaxis_title=f"{axis_prefix} 2",
        zaxis_title=f"{axis_prefix} 3",
    ))
    return _style(fig, 600, title)


def plot_reduction_comparison(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray | None = None,
) -> go.Figure:
    """Vẽ song song 2D embedding của PCA, UMAP, t-SNE trên cùng tập mẫu."""
    items = [(k, np.asarray(v)) for k, v in embeddings.items() if v is not None and np.asarray(v).shape[1] >= 2]
    if not items:
        return _style(go.Figure(), 300, "Chưa có dữ liệu embedding để so sánh")

    fig = make_subplots(
        rows=1, cols=len(items),
        subplot_titles=[_SPACE_NAMES.get(k, k.upper()) for k, _ in items],
    )
    c_names = cluster_names(labels) if labels is not None else None
    cmap = color_map(labels) if labels is not None else {}

    for col_idx, (space, emb) in enumerate(items, start=1):
        if c_names is None:
            fig.add_trace(
                go.Scatter(
                    x=emb[:, 0], y=emb[:, 1], mode="markers",
                    marker=dict(size=5, opacity=0.75, color=PALETTE[0]),
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )
        else:
            for grp in _order(labels):
                mask = c_names == grp
                fig.add_trace(
                    go.Scatter(
                        x=emb[mask, 0], y=emb[mask, 1], mode="markers",
                        name=grp, legendgroup=grp,
                        showlegend=(col_idx == 1),
                        marker=dict(size=5, opacity=0.75, color=cmap.get(grp, PALETTE[0])),
                    ),
                    row=1, col=col_idx,
                )
    return _style(fig, 420, "So sánh trực quan không gian 2D (PCA vs UMAP vs t-SNE)")


# ---------------------------------------------------------------------------
# Clustering / Benchmark / Profiling
# ---------------------------------------------------------------------------
def plot_cluster_sizes(sizes: dict[int, int]) -> go.Figure:
    ids = sorted(sizes, key=lambda v: (v == NOISE_LABEL, v))
    names = ["Noise" if i == NOISE_LABEL else f"Cluster {i}" for i in ids]
    cmap = color_map(np.array(ids))
    fig = px.bar(x=names, y=[sizes[i] for i in ids], color=names, color_discrete_map=cmap)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Số khách")
    return _style(fig, 300)


def plot_metric_comparison(df: pd.DataFrame, metric: str, title: str, lower_is_better: bool = False) -> go.Figure:
    d = df[["algorithm", metric]].copy()
    d["Thuật toán"] = d["algorithm"].map(ALGORITHMS).fillna(d["algorithm"])
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    fig = px.bar(d, x="Thuật toán", y=metric, color_discrete_sequence=[PALETTE[0]], text_auto=".3f")
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    suffix = "càng thấp càng tốt" if lower_is_better else "càng cao càng tốt"
    return _style(fig, 280, f"{title} ({suffix})")


def plot_runtime_comparison(bench_df: pd.DataFrame) -> go.Figure:
    """Biểu đồ cột chồng tách thời gian giảm chiều và thời gian phân cụm."""
    d = bench_df.copy()
    d["Cấu hình"] = (
        d["space"].map(_SPACE_NAMES).fillna(d["space"])
        + " + "
        + d["algorithm"].map(ALGORITHMS).fillna(d["algorithm"])
    )
    fig = go.Figure()
    fig.add_bar(
        x=d["Cấu hình"], y=d["reduction_runtime"],
        name="Thời gian giảm chiều (s)", marker_color=PALETTE[1],
    )
    fig.add_bar(
        x=d["Cấu hình"], y=d["clustering_runtime"],
        name="Thời gian phân cụm (s)", marker_color=PALETTE[0],
    )
    fig.update_layout(
        barmode="stack",
        xaxis_title=None,
        yaxis_title="Thời gian (giây)",
        xaxis_tickangle=-30,
    )
    return _style(fig, 420, "Phân tách thời gian: Giảm chiều + Phân cụm")


def plot_benchmark_heatmap(
    bench_df: pd.DataFrame,
    metric: str = "silhouette_original",
    title: str = "Heatmap chất lượng (4 không gian × 3 thuật toán)",
    color_scale: str = "Viridis",
) -> go.Figure:
    """Heatmap ma trận 4 không gian x 3 thuật toán."""
    piv = bench_df.pivot(index="space", columns="algorithm", values=metric)
    space_order = [s for s in ("original", "pca", "umap", "tsne") if s in piv.index]
    algo_order = [a for a in ("kmeans", "gmm", "dbscan") if a in piv.columns]
    piv = piv.reindex(index=space_order, columns=algo_order)
    piv.index = [_SPACE_NAMES.get(s, s) for s in piv.index]
    piv.columns = [ALGORITHMS.get(a, a) for a in piv.columns]
    fig = px.imshow(
        piv.astype(float),
        text_auto=".3f",
        color_continuous_scale=color_scale,
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Thuật toán", yaxis_title="Không gian dữ liệu")
    return _style(fig, 360, title)


def plot_metric_by_space(
    bench_df: pd.DataFrame,
    metric: str = "silhouette_original",
    title: str = "So sánh chỉ số theo không gian dữ liệu",
) -> go.Figure:
    """Grouped bar chart so sánh chỉ số giữa các thuật toán trên 4 không gian."""
    d = bench_df.copy()
    d["Không gian"] = d["space"].map(_SPACE_NAMES).fillna(d["space"])
    d["Thuật toán"] = d["algorithm"].map(ALGORITHMS).fillna(d["algorithm"])
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    fig = px.bar(
        d, x="Không gian", y=metric, color="Thuật toán", barmode="group",
        color_discrete_sequence=PALETTE, text_auto=".3f",
    )
    fig.update_layout(xaxis_title=None, yaxis_title=metric)
    return _style(fig, 380, title)


def plot_radar(norm_df: pd.DataFrame, names: dict[int, str] | None = None) -> go.Figure:
    """norm_df: index = cluster_id, cột = đặc trưng, giá trị đã chuẩn về [0, 1]."""
    cats = list(norm_df.columns)
    fig = go.Figure()
    for cid, row in norm_df.iterrows():
        color = NOISE_COLOR if cid == NOISE_LABEL else PALETTE[int(cid) % len(PALETTE)]
        label = (names or {}).get(cid, "Noise" if cid == NOISE_LABEL else f"Cluster {cid}")
        vals = row.tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1], name=label,
            line=dict(color=color), fill="toself", opacity=0.35,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)))
    return _style(fig, 480)


def plot_profile_bars(table: pd.DataFrame, names: dict[int, str] | None = None) -> go.Figure:
    """table: index = cluster_id, cột = đặc trưng."""
    d = table.copy()
    d.index = [(names or {}).get(i, "Noise" if i == NOISE_LABEL else f"Cluster {i}") for i in d.index]
    long = d.reset_index(names="Cụm").melt(id_vars="Cụm", var_name="Đặc trưng", value_name="Trung bình")
    fig = px.bar(
        long, x="Đặc trưng", y="Trung bình", color="Cụm", barmode="group",
        color_discrete_sequence=PALETTE,
    )
    return _style(fig, 400)