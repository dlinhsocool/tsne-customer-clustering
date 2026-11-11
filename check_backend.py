"""
check_backend.py - TỰ KIỂM TRA HÀM CỦA BẠN (không cần mở giao diện)

Chạy:
    python check_backend.py              # kiểm tra tất cả hàm
    python check_backend.py clustering   # chỉ các hàm có tên chứa "clustering"

Mỗi hàm trong models/ được gọi với đầu vào lấy từ bản mock (nên bạn kiểm tra được hàm của
mình ngay cả khi hàm của bạn khác chưa viết xong), rồi kiểm tra output theo hợp đồng.

Kết quả từng hàm:
    TODO  - chưa viết (còn NotImplementedError)
    OK    - chạy được và đúng hợp đồng
    LOI   - chạy lỗi hoặc output sai hợp đồng (kèm lý do)
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

from models import clustering as m_cl
from models import customer_profiling as m_cp
from models import data_processing as m_dp
from models import dimensionality_reduction as m_dr
from models import evaluation as m_ev
from models import prediction as m_pr
from utils import mock_backend as mock
from utils.constants import NOISE_LABEL, RANDOM_STATE
from utils.contract import check_contract


def build_inputs() -> dict:
    """Chuỗi đầu vào dựng từ mock để mỗi hàm được thử độc lập."""
    df = mock.generate_sample_dataset(n=300, random_state=RANDOM_STATE)
    feats = mock.SAMPLE_FEATURES
    prep = mock.preprocess_data(df, feats, log_transform_columns=["spending"])
    scale = mock.scale_data(prep["X"])
    pca = mock.run_pca(scale["X_scaled"], 0.95)
    tsne = mock.run_tsne(pca["X_pca"], 2)
    km = mock.run_clustering(scale["X_scaled"], "kmeans")
    gmm = mock.run_clustering(scale["X_scaled"], "gmm")
    db = mock.run_clustering(scale["X_scaled"], "dbscan")
    prof = mock.calculate_cluster_profiles(prep["df_clean"], km["labels"], feats)
    new = {f: float(prep["df_clean"][f].median()) for f in feats}
    pn = mock.preprocess_new_customer(new, prep["preprocess_config"], scale["scaler"], pca["pca_model"])
    return dict(df=df, feats=feats, prep=prep, scale=scale, pca=pca, tsne=tsne, km=km, gmm=gmm,
                db=db, prof=prof, new=new, pn=pn)


def build_checks(i: dict) -> list[tuple[str, str, callable]]:
    """(tên hợp đồng, nhãn hiển thị, hàm gọi)"""
    Xs, Xp = i["scale"]["X_scaled"], i["pca"]["X_pca"]
    labels = i["km"]["labels"]
    return [
        ("load_dataset", "data_processing.load_dataset", lambda: _write_and_load(i["df"])),
        ("validate_dataset", "data_processing.validate_dataset", lambda: m_dp.validate_dataset(i["df"])),
        ("preprocess_data", "data_processing.preprocess_data",
         lambda: m_dp.preprocess_data(i["df"], i["feats"], log_transform_columns=["spending"], sample_size=200)),
        ("scale_data", "data_processing.scale_data", lambda: m_dp.scale_data(i["prep"]["X"])),
        ("run_pca", "dimensionality_reduction.run_pca", lambda: m_dr.run_pca(Xs, 0.95)),
        ("get_explained_variance", "dimensionality_reduction.get_explained_variance",
         lambda: m_dr.get_explained_variance(i["pca"]["pca_model"])),
        ("run_tsne", "dimensionality_reduction.run_tsne", lambda: m_dr.run_tsne(Xp, 2, 30.0, 200.0, 250, 42)),
        ("run_clustering", "clustering.run_kmeans (qua run_clustering)", lambda: m_cl.run_clustering(Xs, "kmeans")),
        ("run_clustering", "clustering.run_gmm (qua run_clustering)", lambda: m_cl.run_clustering(Xs, "gmm")),
        ("run_clustering", "clustering.run_dbscan (qua run_clustering)", lambda: m_cl.run_clustering(Xs, "dbscan")),
        ("evaluate_clustering", "evaluation.evaluate_clustering",
         lambda: m_ev.evaluate_clustering(Xs, labels, i["df"]["true_segment"].to_numpy(), 0.1)),
        ("compare_models", "evaluation.compare_models",
         lambda: m_ev.compare_models({"kmeans": i["km"], "gmm": i["gmm"], "dbscan": i["db"]}, Xs, Xp)),
        ("calculate_cluster_profiles", "customer_profiling.calculate_cluster_profiles",
         lambda: m_cp.calculate_cluster_profiles(i["prep"]["df_clean"], labels, i["feats"])),
        ("generate_cluster_description", "customer_profiling.generate_cluster_description",
         lambda: m_cp.generate_cluster_description(int(i["prof"]["cluster_ids"][0]), i["prof"])),
        ("preprocess_new_customer", "prediction.preprocess_new_customer",
         lambda: m_pr.preprocess_new_customer(i["new"], i["prep"]["preprocess_config"],
                                              i["scale"]["scaler"], i["pca"]["pca_model"])),
        ("predict_customer_cluster", "prediction.predict_customer_cluster (kmeans)",
         lambda: m_pr.predict_customer_cluster(i["pn"]["X_new_scaled"], i["km"], Xs, "auto", 5)),
        ("predict_customer_cluster", "prediction.predict_customer_cluster (dbscan -> kNN)",
         lambda: m_pr.predict_customer_cluster(i["pn"]["X_new_scaled"], i["db"], Xs, "auto", 5)),
        ("transform_new_customer", "prediction.transform_new_customer",
         lambda: m_pr.transform_new_customer(i["pn"]["X_new_pca"], i["tsne"], Xp, "knn", 5)),
    ]


def _write_and_load(df):
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return m_dp.load_dataset(buf)


def extra_rules(label: str, name: str, result) -> list[str]:
    """Các quy tắc nghiệp vụ ngoài kiểm key."""
    p: list[str] = []
    if name == "run_clustering" and "dbscan" in label:
        if result["supports_predict"]:
            p.append("DBSCAN phải có supports_predict=False")
    if name == "run_clustering" and ("kmeans" in label or "gmm" in label):
        if (np.asarray(result["labels"]) == NOISE_LABEL).any():
            p.append("K-Means/GMM không được có nhãn -1")
    return p


def main() -> int:
    flt = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    inputs = build_inputs()
    todo = ok = bad = 0
    print(f"{'Trạng thái':<8} Hàm")
    print("-" * 78)
    for name, label, fn in build_checks(inputs):
        if flt and flt not in label.lower():
            continue
        try:
            result = fn()
        except NotImplementedError:
            print(f"{'TODO':<8} {label}")
            todo += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{'LOI':<8} {label}\n         {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            bad += 1
            continue
        problems = check_contract(name, result) + extra_rules(label, name, result)
        if problems:
            print(f"{'LOI':<8} {label}")
            for p in problems:
                print(f"         - {p}")
            bad += 1
        else:
            print(f"{'OK':<8} {label}")
            ok += 1
    print("-" * 78)
    print(f"OK: {ok}   TODO: {todo}   LOI: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
