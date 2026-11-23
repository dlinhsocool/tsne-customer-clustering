
"""
check_backend.py - TỰ KIỂM TRA HÀM (không cần mở giao diện)

Chạy:
    python check_backend.py
        # Kiểm tra tất cả hàm

    python check_backend.py clustering
        # Chỉ kiểm tra các hàm có tên chứa "clustering"

Mỗi hàm trong models/ được gọi với đầu vào lấy từ bản mock
(nên bạn kiểm tra được hàm của mình ngay cả khi hàm khác chưa viết xong),
rồi kiểm tra output theo hợp đồng.

Kết quả từng hàm:
    TODO - chưa viết (còn NotImplementedError)
    OK   - chạy được và đúng hợp đồng
    LOI  - chạy lỗi hoặc output sai hợp đồng
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

    df = mock.generate_sample_dataset(
        n=300,
        random_state=RANDOM_STATE
    )

    feats = mock.SAMPLE_FEATURES

    prep = mock.preprocess_data(
        df,
        feats,
        log_transform_columns=["spending"]
    )

    scale = mock.scale_data(prep["X"])

    pca = mock.run_pca(
        scale["X_scaled"],
        0.95
    )

    tsne = mock.run_tsne(
        pca["X_pca"],
        2
    )

    # Ba mô hình clustering
    km = mock.run_clustering(
        scale["X_scaled"],
        "kmeans"
    )

    gmm = mock.run_clustering(
        scale["X_scaled"],
        "gmm"
    )

    db = mock.run_clustering(
        scale["X_scaled"],
        "dbscan"
    )

    prof = mock.calculate_cluster_profiles(
        prep["df_clean"],
        km["labels"],
        feats
    )

    new = {
        f: float(prep["df_clean"][f].median())
        for f in feats
    }

    pn = mock.preprocess_new_customer(
        new,
        prep["preprocess_config"],
        scale["scaler"],
        pca["pca_model"]
    )

    return dict(
        df=df,
        feats=feats,
        prep=prep,
        scale=scale,
        pca=pca,
        tsne=tsne,
        km=km,
        gmm=gmm,
        db=db,
        prof=prof,
        new=new,
        pn=pn
    )


def build_checks(i: dict) -> list[tuple[str, str, callable]]:
    """(tên hợp đồng, nhãn hiển thị, hàm gọi)"""

    Xs = i["scale"]["X_scaled"]
    Xp = i["pca"]["X_pca"]

    labels = i["km"]["labels"]

    return [

        # =========================
        # DATA PROCESSING
        # =========================

        (
            "load_dataset",
            "data_processing.load_dataset",
            lambda: _write_and_load(i["df"])
        ),

        (
            "validate_dataset",
            "data_processing.validate_dataset",
            lambda: m_dp.validate_dataset(i["df"])
        ),

        (
            "preprocess_data",
            "data_processing.preprocess_data",
            lambda: m_dp.preprocess_data(
                i["df"],
                i["feats"],
                log_transform_columns=["spending"],
                sample_size=200
            )
        ),

        (
            "scale_data",
            "data_processing.scale_data",
            lambda: m_dp.scale_data(i["prep"]["X"])
        ),

        # =========================
        # DIMENSIONALITY REDUCTION
        # =========================

        (
            "run_pca",
            "dimensionality_reduction.run_pca",
            lambda: m_dr.run_pca(Xs, 0.95)
        ),

        (
            "get_explained_variance",
            "dimensionality_reduction.get_explained_variance",
            lambda: m_dr.get_explained_variance(
                i["pca"]["pca_model"]
            )
        ),

        (
            "run_tsne",
            "dimensionality_reduction.run_tsne",
            lambda: m_dr.run_tsne(
                Xp,
                2,
                30.0,
                200.0,
                250,
                42
            )
        ),

        # =================