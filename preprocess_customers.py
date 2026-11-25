"""
preprocess_customers.py
=======================
Sinh các file bàn giao trong Data/processed từ file thô 20 cột, bằng ĐÚNG hàm mà ứng dụng dùng
(models/data_processing.py) để notebook, script và app cho cùng một kết quả.

Chạy từ thư mục gốc dự án:
    python preprocess_customers.py
    python preprocess_customers.py --raw Data/raw/synthetic_customers_20cols.csv --out Data/processed

Đầu ra:
    customers_clean.csv   record_id + cột tra cứu (thang đo gốc) + recency_days, n_orders
    customers_scaled.csv  record_id + 13 đặc trưng (cột liên tục đã z-score, cột 0/1 giữ nguyên)
    scaler.pkl            {"scaler", "preprocess_config", "danh_muc_chuan"} để xử lý khách mới
    feature_config.json   bản đọc được của preprocess_config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from models.data_processing import load_dataset, preprocess_data, scale_data
from utils.constants import RECOMMENDED_FEATURES

ROOT = Path(__file__).resolve().parent
LOOKUP_COLUMNS = [
    "customer_id", "name", "first_purchase_date", "last_purchase_date",
    "total_spent", "avg_order_value", "loyalty_points", "favorite_category",
    "subscription", "support_tickets",
]


def main(raw_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = load_dataset(raw_path)

    prep = preprocess_data(
        df_raw,
        feature_columns=RECOMMENDED_FEATURES,
        drop_duplicates=True,
        sample_size=None,
        random_state=42,
    )
    scaled = scale_data(prep["X"])

    if prep["n_rows_after"] != len(df_raw):
        raise SystemExit("Số dòng thay đổi sau tiền xử lý: record_id sẽ không khớp dòng nguồn.")

    clean = prep["df_clean"]
    record_id = pd.RangeIndex(1, len(clean) + 1, name="record_id")

    clean_out = clean[[c for c in LOOKUP_COLUMNS if c in clean.columns]].copy()
    clean_out["subscription"] = clean_out["subscription"].astype(bool)
    clean_out["recency_days"] = clean["recency_days"].astype(int).to_numpy()
    clean_out["n_orders"] = clean["n_orders"].astype(int).to_numpy()
    for col in ("loyalty_points", "support_tickets"):
        clean_out[col] = clean_out[col].astype(int)
    clean_out.index = record_id
    clean_out.to_csv(out_dir / "customers_clean.csv")

    X_scaled = pd.DataFrame(scaled["X_scaled"], columns=prep["feature_columns"], index=record_id).round(6)
    X_scaled.to_csv(out_dir / "customers_scaled.csv")

    cfg = prep["preprocess_config"]
    joblib.dump(
        {
            "scaler": scaled["scaler"],
            "preprocess_config": cfg,
            "danh_muc_chuan": cfg["category_levels"].get("favorite_category", []),
        },
        out_dir / "scaler.pkl",
    )
    (out_dir / "feature_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"Hoàn tất: {len(clean_out):,} dòng, {X_scaled.shape[1]} đặc trưng.")
    print("Đặc trưng:", prep["feature_columns"])
    for line in prep["steps_log"]:
        print(" -", line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "Data" / "raw" / "synthetic_customers_20cols.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "Data" / "processed")
    args = parser.parse_args()
    main(args.raw, args.out)
