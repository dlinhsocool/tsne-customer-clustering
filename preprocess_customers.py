import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def lam_sach_khach_hang(df_nguon, ngay_tham_chieu="2025-05-05"):
    khach_hang_sach = df_nguon.copy()
    khach_hang_sach.insert(0, "record_id", range(1, len(khach_hang_sach) + 1))

    for cot_ngay in ["first_purchase_date", "last_purchase_date"]:
        khach_hang_sach[cot_ngay] = pd.to_datetime(
            khach_hang_sach[cot_ngay],
            format="%Y-%m-%d",
            errors="raise",
        )

    khach_hang_sach["recency_days"] = (
        pd.Timestamp(ngay_tham_chieu) - khach_hang_sach["last_purchase_date"]
    ).dt.days

    return khach_hang_sach

def tao_dac_trung(khach_hang_sach):
    cot_dac_trung = [
        "total_spent",
        "avg_order_value",
        "loyalty_points",
        "support_tickets",
        "subscription",
        "favorite_category",
        "recency_days",
    ]

    bang_theo_ma = khach_hang_sach.set_index(
        "record_id",
        verify_integrity=True,
    )

    return bang_theo_ma[cot_dac_trung].copy()

def bien_doi_dac_trung(dac_trung):
    bang = dac_trung.copy()

    bang["avg_order_value_log"] = np.log1p(bang["avg_order_value"])
    bang = bang.drop(columns="avg_order_value")

    bang["subscription"] = bang["subscription"].astype(int)

    return bang

def ma_hoa_danh_muc(dac_trung, danh_muc_chuan):
    bang = dac_trung.copy()
    danh_muc = bang["favorite_category"]

    if not danh_muc.isin(danh_muc_chuan).all():
        raise ValueError("favorite_category bị thiếu hoặc có danh mục ngoài danh sách chuẩn.")

    bang["favorite_category"] = pd.Categorical(
        danh_muc,
        categories=danh_muc_chuan,
    )

    return pd.get_dummies(
        bang,
        columns=["favorite_category"],
        dtype=int,
    )


COT_CAN_CHUAN_HOA = [
    "total_spent",
    "avg_order_value_log",
    "loyalty_points",
    "support_tickets",
    "recency_days",
]


def fit_bo_chuan_hoa(dac_trung_ma_hoa):
    scaler = StandardScaler()
    scaler.fit(dac_trung_ma_hoa[COT_CAN_CHUAN_HOA])
    return scaler


def chuan_hoa_dac_trung(dac_trung_ma_hoa, scaler):
    bang = dac_trung_ma_hoa.copy()
    bang[COT_CAN_CHUAN_HOA] = scaler.transform(bang[COT_CAN_CHUAN_HOA])
    return bang


def tien_xu_ly_huan_luyen(df_nguon, ngay_tham_chieu="2025-05-05"):
    sach = lam_sach_khach_hang(df_nguon, ngay_tham_chieu)
    dac_trung = tao_dac_trung(sach)
    bien_doi = bien_doi_dac_trung(dac_trung)

    danh_muc_chuan = sorted(bien_doi["favorite_category"].unique())
    ma_hoa = ma_hoa_danh_muc(bien_doi, danh_muc_chuan)

    scaler = fit_bo_chuan_hoa(ma_hoa)
    chuan_hoa = chuan_hoa_dac_trung(ma_hoa, scaler)

    return sach, chuan_hoa, scaler, danh_muc_chuan