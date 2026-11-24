"""
check_backend.py - TỰ KIỂM TRA HÀM BACKEND THEO HỢP ĐỒNG v2.0

Chạy:
    python check_backend.py
    python check_backend.py benchmark
"""
from __future__ import annotations

import io
import sys
import traceback

import numpy as np
import pandas as pd

from models import clustering as m_cl
from models import customer_profiling as m_cp
from models import data_processing as m_dp
from models import dimensionality_reduction as m_dr
from models import evaluation as m_ev
from models import prediction as m_pr
from utils import mock_backend as mock
from utils.constants import NOISE_LABEL, RANDOM_STATE
from utils.contract import check_contract


def _make_11cols_sample(n: int = 120, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Tạo mẫu giả lập đúng schema 11 cột của synthetic_customers_11cols.csv."""
    rng = np.random.default_rng(random_state)
    cats = ["Toys", "Electronics", "Books", "Beauty", "Home & Kitchen", "Sports", "Clothing"]
    return pd.DataFrame({
        "customer_id": [f"CUST-{10000 + i}" for i in range(n)],
        "name": [f"Customer {i}" for i in range(n)],
        "first_purchase_date": pd.date_range("2021-01-01", periods=n, freq="7D").strftime("%Y-%m-%d"),
        "last_purchase_date": pd.date_range("2024-06-01", periods=n, freq="2D").strftime("%Y-%m-%d"),
        "total_spent": rng.uniform(100, 9500, size=n).round(2),
        "avg_order_value": rng.lognormal(4.8, 0.7, size=n).round(2),
        "loyalty_points": rng.integers(0, 5000, size=n),
        "favorite_category": rng.choice(cats, size=n),
        "subscription": rng.choice([False, True], size=n),
        "support_tickets": rng.integers(0, 11, size=n),
        "churn_risk": rng.uniform(0.0, 0.5, size=n).round(2),
    })


def build_inputs() -> dict:
    df = mock.generate_sample_dataset(n=180, random_state=RANDOM_STATE)
    feats = mock.SAMPLE_FEATURES
    prep = m_dp.preprocess_data(df, feats, log_transform_columns=["spending"], random_state=RANDOM_STATE)
    scale = m_dp.scale_data(prep["X"])
    Xs = scale["X_scaled"]

    pca = m_dr.run_pca(Xs, 0.95, random_state=RANDOM_STATE)
    try:
        umap_res = m_dr.run_umap(Xs, n_components=3, random_state=RANDOM_STATE)
    except ImportError:
        umap_res = mock.run_umap(Xs, n_components=3, random_state=RANDOM_STATE)

    tsne = m_dr.run_tsne(Xs, n_components=2, n_iter=250, random_state=RANDOM_STATE)

    km = m_cl.run_clustering(Xs, "kmeans", feature_space="original", random_state=RANDOM_STATE)
    gmm = m_cl.run_clustering(Xs, "gmm", feature_space="original", random_state=RANDOM_STATE)
    db = m_cl.run_clustering(Xs, "dbscan", feature_space="original", random_state=RANDOM_STATE)

    prof = m_cp.calculate_cluster_profiles(prep["df_clean"], km["labels"], prep["feature_columns"])
    new = {f: float(prep["df_clean"][f].median()) for f in feats}
    pn = m_pr.preprocess_new_customer(new, prep["preprocess_config"], scale["scaler"])

    df_11 = _make_11cols_sample(120, RANDOM_STATE)

    return dict(
        df=df,
        df_11=df_11,
        feats=feats,
        prep=prep,
        scale=scale,
        pca=pca,
        umap=umap_res,
        tsne=tsne,
        km=km,
        gmm=gmm,
        db=db,
        prof=prof,
        new=new,
        pn=pn,
    )


def _write_and_load(df: pd.DataFrame) -> pd.DataFrame:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return m_dp.load_dataset(buf)


def build_checks(i: dict) -> list[tuple[str, str, callable]]:
    Xs = i["scale"]["X_scaled"]
    Xp = i["pca"]["X_pca"]
    Xu = i["umap"]["embedding"]
    Xt = i["tsne"]["embedding"]
    labels = i["km"]["labels"]

    spaces_dict = {
        "original": {"X": Xs, "reduction_runtime": 0.0},
        "pca": {"X": Xp, "reduction_runtime": i["pca"]["runtime_seconds"]},
        "umap": {"X": Xu, "reduction_runtime": i["umap"]["runtime_seconds"]},
        "tsne": {"X": Xt, "reduction_runtime": i["tsne"]["runtime_seconds"]},
    }

    cols_11 = [
        "first_purchase_date", "last_purchase_date", "total_spent", "avg_order_value",
        "loyalty_points", "favorite_category", "subscription", "support_tickets", "churn_risk",
    ]

    return [
        # DATA PROCESSING
        ("load_dataset", "data_processing.load_dataset", lambda: _write_and_load(i["df"])),
        ("validate_dataset", "data_processing.validate_dataset (11 cols)", lambda: m_dp.validate_dataset(i["df_11"])),
        (
            "preprocess_data",
            "data_processing.preprocess_data (11 cols full)",
            lambda: m_dp.preprocess_data(i["df_11"], cols_11, log_transform_columns=["avg_order_value"]),
        ),
        ("scale_data", "data_processing.scale_data", lambda: m_dp.scale_data(i["prep"]["X"])),

        # DIMENSIONALITY REDUCTION
        ("run_pca", "dimensionality_reduction.run_pca", lambda: m_dr.run_pca(Xs, 0.95)),
        ("get_explained_variance", "dimensionality_reduction.get_explained_variance", lambda: m_dr.get_explained_variance(i["pca"])),
        ("run_umap", "dimensionality_reduction.run_umap", lambda: m_dr.run_umap(Xs, n_components=3)),
        ("run_tsne", "dimensionality_reduction.run_tsne (on X_scaled)", lambda: m_dr.run_tsne(Xs, 2, 30.0, 200.0, 250, 42)),

        # CLUSTERING (4 SPACES)
        ("run_clustering", "clustering.kmeans (original)", lambda: m_cl.run_clustering(Xs, "kmeans", feature_space="original")),
        ("run_clustering", "clustering.gmm (pca)", lambda: m_cl.run_clustering(Xp, "gmm", feature_space="pca")),
        ("run_clustering", "clustering.dbscan (umap)", lambda: m_cl.run_clustering(Xu, "dbscan", feature_space="umap")),
        ("run_clustering", "clustering.kmeans (tsne)", lambda: m_cl.run_clustering(Xt, "kmeans", feature_space="tsne")),

        # EVALUATION & BENCHMARK
        ("evaluate_clustering", "evaluation.evaluate_clustering", lambda: m_ev.evaluate_clustering(Xs, labels, i["df"]["true_segment"].to_numpy(), 0.1)),
        (
            "compare_dimensionality_reduction_method",
            "evaluation.compare_dimensionality_reduction_method",
            lambda: m_ev.compare_dimensionality_reduction_method(Xs, n_components=2, sample_size=120, compute_stability=False),
        ),
        (
            "benchmark_clustering_spaces",
            "evaluation.benchmark_clustering_spaces (4x3=12)",
            lambda: m_ev.benchmark_clustering_spaces(spaces_dict, X_original=Xs),
        ),

        # CUSTOMER PROFILING
        ("calculate_cluster_profiles", "customer_profiling.calculate_cluster_profiles", lambda: m_cp.calculate_cluster_profiles(i["prep"]["df_clean"], labels, i["prep"]["feature_columns"])),
        ("generate_cluster_description", "customer_profiling.generate_cluster_description", lambda: m_cp.generate_cluster_description(int(i["prof"]["cluster_ids"][0]), i["prof"])),

        # PREDICTION (X_scaled main pipeline, no PCA required)
        ("preprocess_new_customer", "prediction.preprocess_new_customer", lambda: m_pr.preprocess_new_customer(i["new"], i["prep"]["preprocess_config"], i["scale"]["scaler"])),
        ("predict_customer_cluster", "prediction.predict_customer_cluster (kmeans)", lambda: m_pr.predict_customer_cluster(i["pn"]["X_new_scaled"], i["km"], Xs, "auto", 5)),
        ("predict_customer_cluster", "prediction.predict_customer_cluster (dbscan)", lambda: m_pr.predict_customer_cluster(i["pn"]["X_new_scaled"], i["db"], Xs, "auto", 5)),
        ("transform_new_customer", "prediction.transform_new_customer (on X_scaled)", lambda: m_pr.transform_new_customer(i["pn"]["X_new_scaled"], i["tsne"], Xs, "knn", 5)),
    ]


def extra_rules(label: str, name: str, result) -> list[str]:
    p: list[str] = []
    if name == "run_clustering":
        labels = np.asarray(result["labels"])
        if "kmeans" in label or "gmm" in label:
            if (labels == NOISE_LABEL).any():
                p.append("K-Means/GMM không được có nhãn -1")
        if "dbscan" in label and not np.issubdtype(labels.dtype, np.integer):
            p.append("Nhãn DBSCAN phải có kiểu số nguyên")
    elif name == "benchmark_clustering_spaces":
        tbl = result["table"]
        if len(tbl) != 12:
            p.append(f"Benchmark 4 không gian × 3 thuật toán phải có 12 dòng, nhận {len(tbl)}")
    return p


def main() -> int:
    flt = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    inputs = build_inputs()
    todo = ok = bad = 0

    print(f"{'Trạng thái':<10} Hàm")
    print("-" * 80)
    for name, label, fn in build_checks(inputs):
        if flt and flt not in label.lower():
            continue
        try:
            result = fn()
        except NotImplementedError:
            print(f"{'TODO':<10} {label}")
            todo += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{'LOI':<10} {label}\n           {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            bad += 1
            continue

        problems = check_contract(name, result) + extra_rules(label, name, result)
        if problems:
            print(f"{'LOI':<10} {label}")
            for prob in problems:
                print(f"           - {prob}")
            bad += 1
        else:
            print(f"{'OK':<10} {label}")
            ok += 1

    print("-" * 80)
    print(f"OK: {ok}   TODO: {todo}   LOI: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())