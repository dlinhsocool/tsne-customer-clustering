"""
preprocess_customers.py
=======================
Script dòng lệnh gọi lại pipeline chuẩn từ models/data_processing.py
để xử lý Data/raw/synthetic_customers_11cols.csv và lưu ra Data/processed/.
"""
from __future__ import annotations

import os
import joblib
import pandas as pd

from models.data_processing import load_dataset, preprocess_data, scale_data

FEATURE_COLS_11 = [
    "first_purchase_date",
    "last_purchase_date",
    "total_spent",
    "avg_order_value",
    "loyalty_points",
    "favorite_category",
    "subscription",
    "support_tickets",
    "churn_risk",
]


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(root, "Data", "raw", "synthetic_customers_11cols.csv")
    out_dir = os.path.join(root, "Data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    df_raw = load_dataset(raw_path)
    prep = preprocess_data(
        df_raw,
        feature_columns=FEATURE_COLS_11,
        missing_strategy="median",
        drop_duplicates=True,
        outlier_method=None,
        log_transform_columns=["avg_order_value"],
        sample_size=None,
        random_state=42,
    )
    sc = scale_data(prep["X"])

    df_clean = prep["df_clean"]
    df_scaled = pd.DataFrame(sc["X_scaled"], columns=prep["feature_columns"])

    df_clean.to_csv(os.path.join(out_dir, "customers_clean.csv"), index=False)
    df_scaled.to_csv(os.path.join(out_dir, "customers_scaled.csv"), index=False)
    joblib.dump(
        {"scaler": sc["scaler"], "preprocess_config": prep["preprocess_config"]},
        os.path.join(out_dir, "scaler.pkl"),
    )
    print(f"Hoàn tất tiền xử lý 11 cột: {df_clean.shape[0]} dòng, {df_scaled.shape[1]} đặc trưng.")


if __name__ == "__main__":
    main()