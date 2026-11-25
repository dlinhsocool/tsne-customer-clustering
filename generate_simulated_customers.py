"""
generate_simulated_customers.py
===============================
Sinh bộ mô phỏng CÓ CẤU TRÚC để đánh giá phân cụm và so sánh các không gian giảm chiều.

Vì sao cần bộ này: file thô 20 cột (Data/raw/synthetic_customers_20cols.csv) là dữ liệu ngẫu nhiên độc lập
(Hopkins ≈ 0.50 trên bộ RFM) nên không có cụm thật để tìm. Ở đây 5 nhóm khách hàng được thiết kế
sao cho TẤT CẢ đặc trưng hành vi đều phụ thuộc vào nhóm:
    recency, số đơn, giá trị đơn, điểm thân thiết, số ticket, tỷ lệ đăng ký, danh mục ưa thích.
Các cột nhân khẩu/nhiễu của file gốc (age, income, credit_score, gender, device_type, marketing_opt_in,
churn_risk, city, country) được lấy ngẫu nhiên độc lập từ file gốc, đúng như trong dữ liệu thật:
chúng là chiều nhiễu, dùng để quan sát điều gì xảy ra khi thêm chiều không mang thông tin.

Đầu ra (Data/simulated):
    customers_simulated_5groups.csv           schema 20 cột của file gốc + cột `nhom` (nhãn thật, để tính ARI)
    customers_simulated_5groups_metadata.csv  customer_id, nhom, so_don, recency_days

Chạy từ thư mục gốc dự án:
    python generate_simulated_customers.py                                   # 4.000 hồ sơ (mặc định)
    python generate_simulated_customers.py --n-per-group 20000 --difficulty 1.5 --tag 100k   # 100.000 hồ sơ
    python generate_simulated_customers.py --extra-noise 30 --tag noise30                    # thêm 30 cột nhiễu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "Data" / "raw" / "synthetic_customers_20cols.csv"
OUT_DIR = ROOT / "Data" / "simulated"
REFERENCE_DATE = pd.Timestamp("2025-05-05")  # trùng ngày tham chiếu của pipeline
NOISE_COLUMNS = [
    "email", "age", "gender", "city", "country", "income",
    "credit_score", "device_type", "churn_risk", "marketing_opt_in",
]
CATEGORIES = ["Beauty", "Books", "Clothing", "Electronics", "Home & Kitchen", "Sports", "Toys"]

# Thứ tự xác suất theo CATEGORIES
GROUPS = [
    {
        "nhom": "Trung thành, chi tiêu cao",
        "recency": (15, 6), "so_don": (34, 5), "gia_tri_don": 180, "loyalty": (3800, 550),
        "tickets": (2, 1.5), "p_subscription": 0.85, "span": (365, 730),
        "category_p": [0.15, 0.05, 0.10, 0.35, 0.25, 0.05, 0.05],
    },
    {
        "nhom": "Mua ít, đơn giá trị cao",
        "recency": (85, 20), "so_don": (5, 2), "gia_tri_don": 650, "loyalty": (1800, 500),
        "tickets": (3, 1.5), "p_subscription": 0.45, "span": (90, 365),
        "category_p": [0.25, 0.05, 0.10, 0.35, 0.15, 0.05, 0.05],
    },
    {
        "nhom": "Khách mới",
        "recency": (10, 4), "so_don": (2, 0.7), "gia_tri_don": 75, "loyalty": (400, 250),
        "tickets": (1, 1.0), "p_subscription": 0.25, "span": (1, 30),
        "category_p": [0.20, 0.10, 0.30, 0.05, 0.05, 0.05, 0.25],
    },
    {
        "nhom": "Lâu chưa quay lại",
        "recency": (280, 25), "so_don": (14, 4), "gia_tri_don": 130, "loyalty": (1500, 600),
        "tickets": (6, 2.0), "p_subscription": 0.20, "span": (365, 900),
        "category_p": [0.05, 0.30, 0.15, 0.05, 0.30, 0.10, 0.05],
    },
    {
        "nhom": "Mua thường xuyên, đơn nhỏ",
        "recency": (30, 8), "so_don": (32, 5), "gia_tri_don": 55, "loyalty": (2500, 600),
        "tickets": (3, 1.5), "p_subscription": 0.60, "span": (180, 730),
        "category_p": [0.05, 0.25, 0.10, 0.05, 0.10, 0.30, 0.15],
    },
]
SIGMA_LOG_GIA_TRI_DON = 0.25
TOTAL_SPENT_CAP = 9999.0  # giữ cùng miền với file gốc (max 9999.68)


def sinh_mot_nhom(nhom: dict, n: int, rng: np.random.Generator, difficulty: float = 1.0) -> pd.DataFrame:
    """`difficulty` nhân độ lệch chuẩn của mọi đặc trưng hành vi: 1.0 = tách rõ, > 1 = các nhóm chồng lấn nhiều hơn."""
    d = float(difficulty)
    recency = np.clip(np.rint(rng.normal(nhom["recency"][0], nhom["recency"][1] * d, size=n)), 1, 366).astype(int)
    so_don = np.clip(np.rint(rng.normal(nhom["so_don"][0], nhom["so_don"][1] * d, size=n)), 1, 50).astype(int)
    avg = rng.lognormal(np.log(nhom["gia_tri_don"]), SIGMA_LOG_GIA_TRI_DON * d, size=n)
    total = np.minimum(so_don * avg, TOTAL_SPENT_CAP).round(2)
    avg = (total / so_don).round(2)

    last = REFERENCE_DATE - pd.to_timedelta(recency, unit="D")
    span = rng.integers(nhom["span"][0], nhom["span"][1] + 1, size=n)
    first = last - pd.to_timedelta(span, unit="D")

    return pd.DataFrame({
        "first_purchase_date": first.strftime("%Y-%m-%d"),
        "last_purchase_date": last.strftime("%Y-%m-%d"),
        "total_spent": total,
        "avg_order_value": avg,
        "loyalty_points": np.clip(np.rint(rng.normal(nhom["loyalty"][0], nhom["loyalty"][1] * d, size=n)), 0, 5000).astype(int),
        "favorite_category": rng.choice(CATEGORIES, size=n, p=nhom["category_p"]),
        "subscription": rng.random(n) < nhom["p_subscription"],
        "support_tickets": np.clip(np.rint(rng.normal(nhom["tickets"][0], nhom["tickets"][1] * d, size=n)), 0, 10).astype(int),
        "nhom": nhom["nhom"],
        "so_don": so_don,
        "recency_days": recency,
    })


def main(n_per_group: int, seed: int, difficulty: float = 1.0, tag: str = "5groups", extra_noise: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    columns = raw.columns.tolist()

    sim = pd.concat([sinh_mot_nhom(g, n_per_group, rng, difficulty) for g in GROUPS], ignore_index=True)
    sim = sim.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # trộn thứ tự dòng
    n = len(sim)

    # Cột nhiễu: mỗi cột bốc độc lập từ file gốc (giữ đúng phân phối, không liên hệ với nhóm)
    for col in NOISE_COLUMNS:
        sim[col] = raw[col].sample(n=n, replace=True, random_state=int(rng.integers(1_000_000))).to_numpy()
    # Kịch bản "nhiều chiều nhiễu": thêm extra_noise cột số độc lập với nhóm (noise_01, noise_02, ...).
    # Dùng để kiểm tra giảm chiều khi dữ liệu có nhiều chiều dư thừa; PHẢI nêu rõ trong báo cáo.
    noise_cols = [f"noise_{i:02d}" for i in range(1, extra_noise + 1)]
    for col in noise_cols:
        sim[col] = rng.normal(0.0, 1.0, size=n).round(4)
    sim["email"] = [f"sim{i:05d}@example.com" for i in range(1, n + 1)]
    sim["customer_id"] = [f"SIM-{i:05d}" for i in range(1, n + 1)]
    sim["name"] = [f"Khách mô phỏng {i:05d}" for i in range(1, n + 1)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sim[columns + noise_cols + ["nhom"]].to_csv(
        OUT_DIR / f"customers_simulated_{tag}.csv", index=False, encoding="utf-8-sig"
    )
    sim[["customer_id", "nhom", "so_don", "recency_days"]].to_csv(
        OUT_DIR / f"customers_simulated_{tag}_metadata.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Đã sinh {n:,} hồ sơ, {len(GROUPS)} nhóm x {n_per_group} -> {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-group", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="nhân độ lệch chuẩn của đặc trưng hành vi; > 1 để các nhóm chồng lấn nhiều hơn")
    parser.add_argument("--tag", default="5groups", help="hậu tố tên file đầu ra")
    parser.add_argument("--extra-noise", type=int, default=0,
                        help="số cột nhiễu số thêm vào (noise_01...), độc lập với nhóm; mặc định 0")
    args = parser.parse_args()
    main(args.n_per_group, args.seed, args.difficulty, args.tag, args.extra_noise)
