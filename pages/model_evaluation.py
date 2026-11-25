"""pages/model_evaluation.py - Trang 5: Thực nghiệm Benchmark 4 không gian × 3 thuật toán."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import interface
from utils import visualization as viz
from utils.constants import (
    ALGORITHMS,
    DEFAULT_PARAMS,
    FEATURE_SPACES,
    PCA_DEFAULT_VARIANCE,
    SK,
    TSNE_DEFAULTS,
    UMAP_DEFAULTS,
)


def _sample_benchmark_data(sample_n: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    scale = interface.get(SK.SCALE)
    prep = interface.get(SK.PREP)
    Xs = np.asarray(scale["X_scaled"])
    df_clean = prep["df_clean"].reset_index(drop=True)

    if sample_n > 0 and len(Xs) > sample_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(Xs), size=int(sample_n), replace=False))
        return Xs[idx], df_clean.iloc[idx].reset_index(drop=True)
    return Xs, df_clean


def _build_representations(X_bench: np.ndarray, seed: int, pca_dim, umap_dim: int, tsne_perp: float, tsne_iter: int,
                           warn: bool = True) -> dict[str, dict]:
    """Tạo 4 không gian (original, PCA, UMAP, t-SNE 2D) từ cùng một mẫu."""
    reps: dict[str, dict] = {
        "original": {"X": X_bench, "reduction_runtime": 0.0, "n_components": X_bench.shape[1]},
    }
    pca_res = interface.run_pca(X_bench, n_components=pca_dim, random_state=seed)
    reps["pca"] = {
        "X": pca_res["X_pca"], "reduction_runtime": pca_res["runtime_seconds"],
        "n_components": pca_res["n_components"], "raw_result": pca_res,
    }
    try:
        umap_res = interface.run_umap(
            X_bench, n_components=max(2, int(umap_dim)),
            n_neighbors=UMAP_DEFAULTS["n_neighbors"], min_dist=UMAP_DEFAULTS["min_dist"], random_state=seed,
        )
        reps["umap"] = {
            "X": umap_res["embedding"], "reduction_runtime": umap_res["runtime_seconds"],
            "n_components": umap_res["n_components"], "raw_result": umap_res,
        }
    except Exception as exc:  # noqa: BLE001
        if warn:
            st.warning(f"Bỏ qua UMAP do lỗi: {exc}")
    tsne_res = interface.run_tsne(
        X_bench, n_components=2, perplexity=float(tsne_perp),
        learning_rate=TSNE_DEFAULTS["learning_rate"], n_iter=int(tsne_iter), random_state=seed,
    )
    reps["tsne"] = {
        "X": tsne_res["embedding"], "reduction_runtime": tsne_res["runtime_seconds"],
        "n_components": 2, "raw_result": tsne_res,
    }
    return reps


def _aggregate_seeds(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn qua các seed cho từng (không gian, thuật toán)."""
    all_rows = pd.concat([t.assign(_seed=i) for i, t in enumerate(tables)], ignore_index=True)
    metrics = ["ari_score", "silhouette_original", "davies_bouldin_original", "n_clusters", "noise_ratio", "total_runtime"]
    for m in metrics:
        all_rows[m] = pd.to_numeric(all_rows[m], errors="coerce")
    grp = all_rows.groupby(["space", "algorithm"], sort=False)[metrics].agg(["mean", "std"])
    grp.columns = [f"{a}_{b}" for a, b in grp.columns]
    return grp.reset_index()


def render() -> None:
    st.title("Model Evaluation & Comparison")
    if not interface.require(
        SK.SCALE,
        message="Chưa có dữ liệu chuẩn hóa X_scaled. Hãy chạy bước 1 ở trang Preprocessing & t-SNE trước.",
    ):
        return

    scale = interface.get(SK.SCALE)
    Xs_full = np.asarray(scale["X_scaled"])
    seed = int(interface.get(SK.RANDOM_STATE))

    st.caption(
        "Luồng thực nghiệm độc lập: đánh giá chất lượng giảm chiều (Original / PCA / UMAP / t-SNE) "
        "và so sánh 12 tổ hợp phân cụm (4 không gian × 3 thuật toán) trên cùng một tập mẫu."
    )

    with st.expander("Cấu hình thí nghiệm Benchmark", expanded=True):
        c1, c2, c3 = st.columns(3)
        max_n = len(Xs_full)
        default_sample = min(1500, max_n)
        bench_n = c1.number_input(
            "Số mẫu thực nghiệm (dùng chung cho cả 4 không gian)",
            min_value=min(50, max_n),
            max_value=max_n,
            value=default_sample,
            step=100,
        )
        dim_mode = c2.radio(
            "Chế độ số chiều giảm chiều",
            ["Tiêu chuẩn (PCA 95%, UMAP 5D, t-SNE 2D)", "Ép đồng đều 2D (PCA 2D, UMAP 2D, t-SNE 2D)"],
        )
        k_clusters = c3.slider("Số cụm K (cho K-Means & GMM)", 2, 10, DEFAULT_PARAMS["kmeans"]["n_clusters"])

        c4, c5, c6 = st.columns(3)
        tsne_perp = c4.slider("t-SNE Perplexity", 5, 50, int(TSNE_DEFAULTS["perplexity"]))
        tsne_iter = c5.number_input("t-SNE Iterations", 250, 2000, 500, step=250)
        run_stability = c6.checkbox(
            "Đo Stability (chạy 3 seeds — chậm hơn)",
            value=False,
            help="Chạy lặp lại với 3 random_state để tính độ ổn định của embedding.",
        )
        n_repeats = st.slider(
            "Số lần lặp toàn bộ thí nghiệm với seed khác nhau (báo cáo trung bình ± độ lệch chuẩn)",
            1, 5, 1,
            help="t-SNE/UMAP phụ thuộc seed nên một lần chạy chưa đủ để kết luận. 3 lần trở lên là hợp lý.",
        )

        # Cột nhãn tham chiếu nếu dataset có cột nhãn
        prep = interface.get(SK.PREP)
        df_clean_full = prep["df_clean"]
        label_candidates = [c for c in df_clean_full.columns if c not in prep["feature_columns"]]
        default_ref_idx = 0
        for idx_c, cname in enumerate(label_candidates, start=1):
            if any(h in cname.lower() for h in ("segment", "label", "true", "nhom")):
                default_ref_idx = idx_c
                break
        ref_col = st.selectbox(
            "Cột nhãn thật để tính ARI (nếu có)",
            ["Không có"] + label_candidates,
            index=default_ref_idx,
        )

    if st.button("Chạy toàn bộ Benchmark (4 không gian × 3 thuật toán)", type="primary"):
        interface.reset_from("benchmark")
        X_bench, df_bench = _sample_benchmark_data(int(bench_n), seed)
        labels_true = None if ref_col == "Không có" else df_bench[ref_col].to_numpy()

        force_2d = dim_mode.startswith("Ép đồng đều 2D")
        pca_dim = 2 if force_2d else PCA_DEFAULT_VARIANCE
        umap_dim = 2 if force_2d else min(UMAP_DEFAULTS["n_components"], X_bench.shape[1] - 1)

        with st.spinner("1-3/4 Đang chạy PCA, UMAP, t-SNE..."):
            reps = _build_representations(X_bench, seed, pca_dim, umap_dim, tsne_perp, tsne_iter)

        with st.spinner("4/4 Đang đánh giá chất lượng giảm chiều & chạy 12 mô hình phân cụm..."):
            precomputed_dr = {
                m: {"embedding": reps[m]["X"], "runtime_seconds": reps[m]["reduction_runtime"]}
                for m in ("pca", "umap", "tsne")
                if m in reps
            }
            dr_table = interface.compare_dimensionality_reduction_method(
                X_bench,
                n_components=2,
                n_neighbors=10,
                random_states=(seed, seed + 1, seed + 2) if run_stability else (seed,),
                tsne_params={"perplexity": float(tsne_perp), "max_iter": int(tsne_iter)},
                sample_size=len(X_bench),
                random_state=seed,
                compute_stability=run_stability,
                precomputed=precomputed_dr,
            )

            clust_params = {
                "kmeans": {"n_clusters": int(k_clusters)},
                "gmm": {"n_components": int(k_clusters)},
                "dbscan": {"eps": None, "min_samples": 5},
            }
            bench_out = interface.benchmark_clustering_spaces(
                reps,
                clustering_params=clust_params,
                X_original=X_bench,
                labels_true=labels_true,
                random_state=seed,
            )

        repeat_tables = [bench_out["table"]]
        if n_repeats > 1:
            for extra in range(1, int(n_repeats)):
                with st.spinner(f"Lặp lại thí nghiệm với seed {seed + extra} ({extra + 1}/{n_repeats})..."):
                    reps_i = _build_representations(
                        X_bench, seed + extra, pca_dim, umap_dim, tsne_perp, tsne_iter, warn=False
                    )
                    out_i = interface.benchmark_clustering_spaces(
                        reps_i,
                        clustering_params=clust_params,
                        X_original=X_bench,
                        labels_true=labels_true,
                        random_state=seed + extra,
                    )
                    repeat_tables.append(out_i["table"])
        interface.put(SK.BENCHMARK_REPEATS, repeat_tables)

        interface.put(SK.BENCHMARK_SAMPLE, {"X": X_bench, "df": df_bench})
        interface.put(SK.BENCHMARK_REPRESENTATIONS, reps)
        interface.put(SK.BENCHMARK_REDUCTION_TABLE, dr_table)
        interface.put(SK.BENCHMARK_CLUSTER_TABLE, bench_out["table"])
        interface.put(SK.BENCHMARK_CLUSTER_RESULTS, bench_out["results"])

    dr_table = interface.get(SK.BENCHMARK_REDUCTION_TABLE)
    bench_table = interface.get(SK.BENCHMARK_CLUSTER_TABLE)
    reps = interface.get(SK.BENCHMARK_REPRESENTATIONS, {})
    bench_results = interface.get(SK.BENCHMARK_CLUSTER_RESULTS, {})
    bench_sample = interface.get(SK.BENCHMARK_SAMPLE)
    repeat_tables = interface.get(SK.BENCHMARK_REPEATS) or []

    if dr_table is None or bench_table is None:
        st.info("Bấm nút **Chạy toàn bộ Benchmark** ở trên để thực hiện thí nghiệm.")
        return

    tab_a, tab_b, tab_c = st.tabs([
        "Tab A — Dimensionality Reduction",
        "Tab B — Clustering Benchmark (4×3)",
        "Tab C — Visualization",
    ])

    # -----------------------------------------------------------------------
    # TAB A: DIMENSIONALITY REDUCTION
    # -----------------------------------------------------------------------
    with tab_a:
        st.subheader("Bảng đánh giá chất lượng bảo toàn cấu trúc không gian")
        show_dr = dr_table.copy()
        show_dr["method"] = show_dr["method"].map(FEATURE_SPACES).fillna(show_dr["method"])
        show_dr = show_dr.rename(columns={
            "method": "Phương pháp",
            "n_components": "Số chiều",
            "runtime_seconds": "Thời gian (s)",
            "trustworthiness": "Trustworthiness",
            "continuity": "Continuity",
            "knn_preservation": "k-NN Preservation",
            "spearman_rank_correlation": "Spearman",
            "stability_score": "Stability",
            "note": "Ghi chú",
        })
        num_dr = ["Thời gian (s)", "Trustworthiness", "Continuity", "k-NN Preservation", "Spearman", "Stability"]
        for c in num_dr:
            show_dr[c] = pd.to_numeric(show_dr[c], errors="coerce")
        st.dataframe(
            show_dr.drop(columns=["n_samples", "n_neighbors"], errors="ignore").style.format(
                {c: "{:.3f}" for c in num_dr}, na_rep="N/A"
            ),
            hide_index=True,
            width="stretch",
        )

        # Vẽ song song 2D của PCA, UMAP, t-SNE
        emb_2d = {
            m: reps[m]["X"]
            for m in ("pca", "umap", "tsne")
            if m in reps and np.asarray(reps[m]["X"]).shape[1] >= 2
        }
        ref_labels = bench_results.get("original", {}).get("kmeans", {}).get("labels")
        st.plotly_chart(viz.plot_reduction_comparison(emb_2d, labels=ref_labels), width="stretch")

        if "pca" in reps and "raw_result" in reps["pca"]:
            with st.expander("Chi tiết phương sai giải thích của PCA (Scree Plot)"):
                var_df = interface.get_explained_variance(reps["pca"]["raw_result"])
                st.plotly_chart(viz.plot_scree(var_df, target=PCA_DEFAULT_VARIANCE), width="stretch")

    # -----------------------------------------------------------------------
    # TAB B: CLUSTERING BENCHMARK (12 TRƯỜNG HỢP)
    # -----------------------------------------------------------------------
    with tab_b:
        st.subheader("Bảng so sánh 12 cấu hình (4 không gian × 3 thuật toán)")
        st.caption(
            "**Cách đọc:** chỉ số chính là **ARI** (so với nhãn thật, nếu có) và **Silhouette / DB (Original)**: "
            "nhãn cụm của từng cấu hình được đo lại trên cùng một không gian gốc `X_scaled`, nên so sánh được giữa "
            "các không gian. **Silhouette (Space)** đo trong chính không gian giảm chiều đó nên *không so sánh được*: "
            "embedding 2D luôn gom điểm thành khối nên điểm này cao ngay cả với dữ liệu ngẫu nhiên."
        )
        if bench_table["ari_score"].isna().all():
            st.info("Chưa có nhãn thật nên không tính được ARI. Chọn cột nhãn (ví dụ `nhom` của bộ mô phỏng) ở phần cấu hình.")
        show_b = bench_table.copy()
        show_b["space"] = show_b["space"].map(
            {"original": "Original", "pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}
        ).fillna(show_b["space"])
        show_b["algorithm"] = show_b["algorithm"].map(ALGORITHMS).fillna(show_b["algorithm"])
        show_b["noise_ratio"] = show_b["noise_ratio"] * 100.0
        show_b = show_b.rename(columns={
            "space": "Không gian",
            "algorithm": "Thuật toán",
            "n_features": "Số chiều",
            "n_clusters": "Số cụm",
            "n_noise": "Noise",
            "noise_ratio": "Tỷ lệ Noise (%)",
            "reduction_runtime": "T_giảm chiều (s)",
            "clustering_runtime": "T_phân cụm (s)",
            "total_runtime": "Tổng T (s)",
            "silhouette_space": "Silhouette (Space)",
            "davies_bouldin_space": "DB (Space)",
            "silhouette_original": "Silhouette (Original)",
            "davies_bouldin_original": "DB (Original)",
            "ari_score": "ARI",
            "note": "Ghi chú",
        })
        fmt_cols = {
            "Tỷ lệ Noise (%)": "{:.1f}%",
            "T_giảm chiều (s)": "{:.3f}",
            "T_phân cụm (s)": "{:.3f}",
            "Tổng T (s)": "{:.3f}",
            "Silhouette (Space)": "{:.3f}",
            "DB (Space)": "{:.3f}",
            "Silhouette (Original)": "{:.3f}",
            "DB (Original)": "{:.3f}",
            "ARI": "{:.3f}",
        }
        for c in fmt_cols:
            show_b[c] = pd.to_numeric(show_b[c], errors="coerce")
        main_cols = ["Không gian", "Thuật toán", "Số chiều", "Số cụm", "ARI", "Silhouette (Original)", "DB (Original)"]
        rest_cols = [c for c in show_b.columns if c not in main_cols and c != "n_samples"]
        st.dataframe(
            show_b[[c for c in main_cols if c in show_b.columns] + rest_cols].style.format(fmt_cols, na_rep="N/A"),
            hide_index=True,
            width="stretch",
        )

        if len(repeat_tables) > 1:
            st.markdown(f"**Trung bình ± độ lệch chuẩn qua {len(repeat_tables)} seed**")
            agg = _aggregate_seeds(repeat_tables)
            agg["space"] = agg["space"].map({"original": "Original", "pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}).fillna(agg["space"])
            agg["algorithm"] = agg["algorithm"].map(ALGORITHMS).fillna(agg["algorithm"])

            def _pm(mean, std, nd=3):
                if pd.isna(mean):
                    return "N/A"
                return f"{mean:.{nd}f} ± {0.0 if pd.isna(std) else std:.{nd}f}"

            agg_show = pd.DataFrame({
                "Không gian": agg["space"],
                "Thuật toán": agg["algorithm"],
                "Số cụm": [_pm(m, sd, 1) for m, sd in zip(agg["n_clusters_mean"], agg["n_clusters_std"])],
                "ARI": [_pm(m, sd) for m, sd in zip(agg["ari_score_mean"], agg["ari_score_std"])],
                "Silhouette (Original)": [_pm(m, sd) for m, sd in zip(agg["silhouette_original_mean"], agg["silhouette_original_std"])],
                "DB (Original)": [_pm(m, sd) for m, sd in zip(agg["davies_bouldin_original_mean"], agg["davies_bouldin_original_std"])],
            })
            st.dataframe(agg_show, hide_index=True, width="stretch")

        st.plotly_chart(viz.plot_runtime_comparison(bench_table), width="stretch")

        has_ari = bench_table["ari_score"].notna().any()
        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.plotly_chart(
                viz.plot_benchmark_heatmap(
                    bench_table,
                    metric="ari_score" if has_ari else "silhouette_original",
                    title=(
                        "Heatmap ARI so với nhãn thật (càng cao càng tốt)"
                        if has_ari else "Heatmap Silhouette trên dữ liệu gốc (càng cao càng tốt)"
                    ),
                ),
                width="stretch",
            )
        with c_right:
            st.plotly_chart(
                viz.plot_metric_by_space(
                    bench_table,
                    metric="silhouette_original",
                    title="Silhouette (Original) theo không gian & thuật toán",
                ),
                width="stretch",
            )

    # -----------------------------------------------------------------------
    # TAB C: VISUALIZATION TỪNG CẤU HÌNH
    # -----------------------------------------------------------------------
    with tab_c:
        st.subheader("Trực quan hóa từng cấu hình trong Benchmark")
        avail_spaces = list(bench_results.keys())
        sel_col1, sel_col2, sel_col3 = st.columns(3)
        sel_space = sel_col1.selectbox(
            "Không gian dữ liệu",
            avail_spaces,
            format_func=lambda s: FEATURE_SPACES.get(s, s),
        )
        sel_algo = sel_col2.selectbox(
            "Thuật toán phân cụm",
            list(ALGORITHMS.keys()),
            format_func=lambda a: ALGORITHMS[a],
        )
        view_options = [s for s in ("tsne", "umap", "pca") if s in reps and np.asarray(reps[s]["X"]).shape[1] >= 2]
        view_space = sel_col3.selectbox(
            "Tọa độ hiển thị 2D",
            view_options,
            format_func=lambda s: s.upper(),
        )

        res_item = bench_results[sel_space][sel_algo]
        row_info = bench_table[(bench_table["space"] == sel_space) & (bench_table["algorithm"] == sel_algo)].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Số cụm", int(res_item["n_clusters"]))
        m2.metric("Tỷ lệ Noise", f"{row_info['noise_ratio'] * 100:.1f}%")
        m3.metric("Silhouette (Original)", "N/A" if pd.isna(row_info["silhouette_original"]) else f"{row_info['silhouette_original']:.3f}")
        m4.metric("Tổng thời gian", f"{row_info['total_runtime']:.3f} s")

        left_v, right_v = st.columns([1, 2], gap="large")
        with left_v:
            st.plotly_chart(viz.plot_cluster_sizes(res_item["cluster_sizes"]), width="stretch")
        with right_v:
            coords = np.asarray(reps[view_space]["X"])
            hover = interface.hover_dataframe(bench_sample["df"]) if bench_sample else None
            st.plotly_chart(
                viz.plot_scatter_2d(
                    coords,
                    labels=res_item["labels"],
                    hover_df=hover,
                    axis_prefix=view_space.upper(),
                    title=f"Nhãn {ALGORITHMS[sel_algo]} trên {sel_space.upper()} (vẽ trên {view_space.upper()})",
                ),
                width="stretch",
            )