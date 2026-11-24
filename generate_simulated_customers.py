import numpy as np
import pandas as pd
import os


def sinh_chi_so_mua_hang(nhom, so_ho_so, rng):
    recency_tho = rng.normal(
        loc=nhom["recency_dien_hinh"],
        scale=nhom["std_recency"],
        size=so_ho_so
    )
    recency = np.clip(np.rint(recency_tho), 1, 366).astype(int)

    so_don_tho = rng.normal(
        loc=nhom["so_don_dien_hinh"],
        scale=nhom["std_so_don"],
        size=so_ho_so
    )
    so_don = np.maximum(np.rint(so_don_tho), 1).astype(int)

    avg_order_value = rng.lognormal(
        mean=np.log(nhom["gia_tri_don_dien_hinh"]),
        sigma=nhom["sigma_log_gia_tri_don"],
        size=so_ho_so
    ).round(2)

    total_spent = (so_don * avg_order_value).round(2)

    return pd.DataFrame({
        "recency_days": recency,
        "so_don": so_don,
        "avg_order_value": avg_order_value,
        "total_spent": total_spent
    })


def them_ngay_mua(mau, ngay_tham_chieu, so_ngay_min, so_ngay_max, rng):
    ket_qua = mau.copy()

    khoang_recency = pd.to_timedelta(
        ket_qua["recency_days"], unit="D"
    )
    ket_qua["last_purchase_date"] = ngay_tham_chieu - khoang_recency

    khoang_mua = rng.integers(
        low=so_ngay_min,
        high=so_ngay_max + 1,
        size=len(ket_qua)
    )
    mot_don = ket_qua["so_don"] == 1
    khoang_mua = np.where(mot_don, 0, khoang_mua)

    ket_qua["first_purchase_date"] = (
        ket_qua["last_purchase_date"]
        - pd.to_timedelta(khoang_mua, unit="D")
    )

    return ket_qua


def them_thong_tin_bo_sung(mau, danh_muc, rng):
    ket_qua = mau.copy()
    so_ho_so = len(ket_qua)

    ket_qua["loyalty_points"] = rng.integers(
        low=0, high=5001, size=so_ho_so
    )
    ket_qua["support_tickets"] = rng.integers(
        low=0, high=11, size=so_ho_so
    )
    ket_qua["favorite_category"] = rng.choice(
        danh_muc, size=so_ho_so, replace=True
    )
    ket_qua["subscription"] = rng.choice(
        [False, True], size=so_ho_so, replace=True
    )

    return ket_qua


def sinh_ho_so_nhom(
    nhom, so_ho_so, ngay_tham_chieu,
    so_ngay_min, so_ngay_max, danh_muc, rng
):
    mau_chi_so = sinh_chi_so_mua_hang(nhom, so_ho_so, rng)

    mau_da_co_ngay = them_ngay_mua(
        mau_chi_so, ngay_tham_chieu, so_ngay_min, so_ngay_max, rng
    )

    ket_qua = them_thong_tin_bo_sung(
        mau_da_co_ngay, danh_muc, rng
    )

    ket_qua["nhom"] = nhom["nhom"]

    return ket_qua

def main():
    thu_muc_du_an = os.path.dirname(os.path.abspath(__file__))

    file_nguon = os.path.join(
        thu_muc_du_an, "Data", "raw", "synthetic_customers_10cols.csv"
    )
    thu_muc_xuat = os.path.join(thu_muc_du_an, "Data", "simulated")

    df_nguon = pd.read_csv(file_nguon)
    cot_csv_nguon = df_nguon.columns.tolist()
    danh_muc_nguon = df_nguon["favorite_category"].dropna().unique()

    thiet_ke_5_nhom = pd.DataFrame({
        "nhom": [
            "Trung thành, chi tiêu cao",
            "Mua ít, đơn giá trị cao",
            "Khách mới",
            "Lâu chưa quay lại",
            "Mua thường xuyên, đơn nhỏ"
        ],
        "recency_dien_hinh": [15, 85, 10, 280, 30],
        "std_recency": [5, 18, 4, 25, 8],
        "so_don_dien_hinh": [34, 5, 2, 14, 32],
        "std_so_don": [5, 2, 0.7, 4, 5],
        "gia_tri_don_dien_hinh": [180, 650, 75, 130, 55],
        "sigma_log_gia_tri_don": 0.25,
        "so_ngay_min": [365, 90, 1, 365, 180],
        "so_ngay_max": [730, 365, 30, 900, 730]
    })

    rng = np.random.default_rng(42)
    ngay_tham_chieu = pd.Timestamp("2025-05-05")
    cac_bang_nhom = []

    for _, nhom in thiet_ke_5_nhom.iterrows():
        bang_nhom = sinh_ho_so_nhom(
            nhom=nhom,
            so_ho_so=100,
            ngay_tham_chieu=ngay_tham_chieu,
            so_ngay_min=int(nhom["so_ngay_min"]),
            so_ngay_max=int(nhom["so_ngay_max"]),
            danh_muc=danh_muc_nguon,
            rng=rng
        )
        cac_bang_nhom.append(bang_nhom)

    mau_co_id = pd.concat(cac_bang_nhom, ignore_index=True)
    so_thu_tu = range(1, len(mau_co_id) + 1)

    mau_co_id["customer_id"] = [f"SIM-{i:05d}" for i in so_thu_tu]
    mau_co_id["name"] = [f"Khách mô phỏng {i:05d}" for i in so_thu_tu]

    khach_hang_mo_phong = mau_co_id[cot_csv_nguon]
    bang_phu_mo_phong = mau_co_id[[
        "customer_id", "so_don", "recency_days", "nhom"
    ]]

    os.makedirs(thu_muc_xuat, exist_ok=True)
    file_ho_so = os.path.join(
        thu_muc_xuat, "customers_simulated_5groups.csv"
    )
    file_bang_phu = os.path.join(
        thu_muc_xuat, "customers_simulated_5groups_metadata.csv"
    )

    khach_hang_mo_phong.to_csv(
        file_ho_so, index=False, date_format="%Y-%m-%d", encoding="utf-8-sig"
    )
    bang_phu_mo_phong.to_csv(
        file_bang_phu, index=False, encoding="utf-8-sig"
    )

    print("File hồ sơ:", khach_hang_mo_phong.shape, file_ho_so)
    print("File bảng phụ:", bang_phu_mo_phong.shape, file_bang_phu)


if __name__ == "__main__":
    main()