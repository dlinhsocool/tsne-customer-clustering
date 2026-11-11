"""
models/data_processing.py
=========================
Người phụ trách: Ngọc Đức (dữ liệu & tiền xử lý)

NHIỆM VỤ FILE: đọc CSV -> kiểm tra -> làm sạch -> chọn đặc trưng -> chuẩn hóa.

QUY ĐỊNH CHUNG (KHÔNG ĐƯỢC TỰ Ý ĐỔI):
  * Chỉ viết thân hàm. Giữ nguyên tên hàm, tên tham số, giá trị mặc định, tên key output.
  * Thứ tự dòng: sau preprocess_data, df_clean / X / X_scaled / mọi nhãn cụm / mọi tọa độ
    đều có CÙNG số dòng và CÙNG thứ tự. Dòng i ở đâu cũng là khách hàng i.
    => df_clean phải reset_index(drop=True).
  * Không có dữ liệu thì trả None, KHÔNG xóa key.
  * Có yếu tố ngẫu nhiên thì dùng random_state (mặc định 42).
  * Không in/print ra màn hình, không gọi streamlit. Lỗi dữ liệu thì raise ValueError
    kèm thông báo tiếng Việt dễ hiểu.
  * Kiểm tra bản của bạn: `python check_backend.py`
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_dataset(file: Any, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Đọc file CSV thành DataFrame (chưa xử lý gì).

    Args:
        file (str | Path | file-like): đường dẫn hoặc đối tượng file (Streamlit UploadedFile).
        sep (str): ký tự phân tách cột. Mặc định ",".
        encoding (str): bảng mã. Mặc định "utf-8".

    Returns:
        pd.DataFrame: dữ liệu gốc, mỗi dòng một khách hàng.

    Raises:
        ValueError: file rỗng, không đọc được, hoặc chỉ có 1 cột (thường do sai sep).

    Ví dụ gọi:
        df = interface.load_dataset(uploaded_file, sep=",")
    """
    raise NotImplementedError("TODO (Ngọc Đức): pd.read_csv + kiểm tra rỗng")


def validate_dataset(df: pd.DataFrame, feature_columns: list[str] | None = None) -> dict:
    """Kiểm tra chất lượng dữ liệu, KHÔNG sửa dữ liệu.

    Args:
        df (pd.DataFrame): dữ liệu gốc.
        feature_columns (list[str] | None): các cột đặc trưng dự kiến dùng. None = kiểm tra toàn bộ.

    Returns:
        dict với các key (đủ key, không đổi tên):
            "is_valid": bool                   - False nếu có lỗi chặn (ví dụ < 2 cột số, rỗng)
            "n_rows": int
            "n_cols": int
            "columns": list[str]
            "dtypes": dict[str, str]           - {tên cột: "float64" ...}
            "missing_per_column": dict[str, int]
            "n_duplicates": int                - số dòng trùng hoàn toàn
            "numeric_columns": list[str]
            "categorical_columns": list[str]
            "errors": list[str]                - lỗi chặn, tiếng Việt
            "warnings": list[str]              - cảnh báo không chặn (cột lệch mạnh, ID trùng...)

    Ví dụ output:
        {"is_valid": True, "n_rows": 600, "n_cols": 8, ..., "errors": [], "warnings": ["Cột income có 12 giá trị thiếu"]}
    """
    raise NotImplementedError("TODO (Ngọc Đức): thống kê missing, trùng, kiểu cột")


def preprocess_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    missing_strategy: str = "median",
    drop_duplicates: bool = True,
    outlier_method: str | None = None,
    log_transform_columns: list[str] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict:
    """Làm sạch dữ liệu và chọn đặc trưng. CHƯA chuẩn hóa (việc của scale_data).

    Thứ tự thực hiện bắt buộc:
        1) bỏ dòng trùng (nếu drop_duplicates)
        2) xử lý giá trị thiếu theo missing_strategy
        3) loại ngoại lai theo outlier_method (nếu có)
        4) lấy mẫu sample_size dòng (nếu có; dùng random_state)
        5) reset_index(drop=True)
        6) log1p các cột trong log_transform_columns -> chỉ áp dụng cho X, KHÔNG áp dụng cho df_clean

    Args:
        df (pd.DataFrame): dữ liệu gốc.
        feature_columns (list[str]): các cột số dùng làm đặc trưng, thứ tự này là thứ tự cột của X.
        missing_strategy (str): "median" | "mean" | "most_frequent" | "drop". Mặc định "median".
        drop_duplicates (bool): bỏ dòng trùng hoàn toàn. Mặc định True.
        outlier_method (str | None): None | "iqr" | "zscore". Mặc định None (không loại).
        log_transform_columns (list[str] | None): cột cần log1p (cột lệch phải). Mặc định None.
        sample_size (int | None): giữ tối đa bấy nhiêu dòng để t-SNE chạy nhanh. None = giữ hết.
        random_state (int): seed lấy mẫu. Mặc định 42.

    Returns:
        dict:
            "df_clean": pd.DataFrame        - dữ liệu sạch, GIỮ THANG ĐO GỐC, giữ mọi cột gốc
                                              (id, tên...), index 0..n-1. Dùng cho hover và profiling.
            "X": pd.DataFrame               - chỉ các cột feature_columns, đã log1p nếu có, chưa scale.
            "feature_columns": list[str]
            "preprocess_config": dict       - để xử lý khách hàng mới ĐÚNG NHƯ KHÁCH CŨ:
                {"feature_columns": [...], "missing_strategy": str,
                 "fill_values": {cột: giá trị điền thiếu},
                 "log_transform_columns": [...], "outlier_method": str | None,
                 "feature_ranges": {cột: (min, max)}}   # trên thang gốc của df_clean
            "n_rows_before": int
            "n_rows_after": int
            "n_removed": int
            "steps_log": list[str]          - mỗi bước một dòng tiếng Việt, ví dụ "Bỏ 3 dòng trùng"

    Ví dụ gọi:
        prep = interface.preprocess_data(df, ["spending", "income"], sample_size=5000)
        prep["df_clean"], prep["X"]
    """
    raise NotImplementedError("TODO (Ngọc Đức): làm sạch + log1p + lấy mẫu + preprocess_config")


def scale_data(X: pd.DataFrame | np.ndarray, scaler: Any | None = None) -> dict:
    """Chuẩn hóa bằng StandardScaler.

    Hai chế độ:
        * scaler=None  : fit trên X rồi transform (dùng lúc huấn luyện).
        * scaler=<đã fit>: chỉ transform, KHÔNG fit lại (dùng cho khách hàng mới).

    Args:
        X (pd.DataFrame | np.ndarray): đặc trưng chưa scale, shape (n, d).
        scaler (sklearn StandardScaler | None): scaler đã fit hoặc None.

    Returns:
        dict:
            "X_scaled": np.ndarray (n, d)   - trung bình ~0, độ lệch chuẩn ~1 theo cột
            "scaler": StandardScaler đã fit - có thuộc tính mean_ và scale_
            "feature_columns": list[str] | None - tên cột nếu X là DataFrame, ngược lại None

    Ví dụ gọi:
        sc = interface.scale_data(prep["X"])
        sc["X_scaled"].shape  # (n, d)
    """
    raise NotImplementedError("TODO (Ngọc Đức): StandardScaler fit_transform / transform")
