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
    try:
        if hasattr(file, "seekable") and file.seekable():
            file.seek(0)
        df = pd.read_csv(file, sep=sep, encoding=encoding)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("File CSV rỗng hoặc không có tiêu đề cột.") from exc
    except (OSError, ValueError, TypeError, LookupError) as exc:
        raise ValueError("Không đọc được file CSV. Hãy kiểm tra file và bảng mã.") from exc

    if df.empty:
        raise ValueError("File CSV không có dòng dữ liệu nào.")
    if df.shape[1] < 2:
        raise ValueError("File chỉ có một cột. Hãy kiểm tra ký tự phân tách.")
    return df



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
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    missing_per_column = {column: int(count) for column, count in df.isna().sum().items()}
    n_duplicates = int(df.duplicated().sum())

    errors = []
    if df.empty:
        errors.append("Dữ liệu rỗng.")
    selected_columns = numeric_columns if feature_columns is None else feature_columns
    missing_columns = [column for column in selected_columns if column not in df.columns]
    non_numeric = [column for column in selected_columns if column in categorical_columns]
    if missing_columns:
        errors.append(f"Thiếu cột đặc trưng {', '.join(missing_columns)}.")
    if non_numeric:
        errors.append(f"Cột đặc trưng phải là kiểu số: {', '.join(non_numeric)}.")
    if len(set(selected_columns).intersection(numeric_columns)) < 2:
        errors.append("Cần ít nhất hai cột đặc trưng số để phân cụm.")

    warnings = [
        f"Cột {column} có {count} giá trị thiếu."
        for column, count in missing_per_column.items()
        if count > 0
    ]
    if n_duplicates:
        warnings.append(f"Có {n_duplicates} dòng trùng hoàn toàn.")
    for column in df.columns:
        name = str(column).lower()
        if (name == "id" or name.endswith("_id")) and df[column].dropna().duplicated().any():
            warnings.append(f"Cột {column} có mã lặp. Hãy kiểm tra ý nghĩa của mỗi dòng.")

    return {
        "is_valid": not errors,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_per_column": missing_per_column,
        "n_duplicates": n_duplicates,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "errors": errors,
        "warnings": warnings,
    }



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
    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Hãy chọn các cột đặc trưng khác nhau.")
    numeric_columns = df.select_dtypes(include="number").columns
    invalid_columns = [column for column in columns if column not in numeric_columns]
    if invalid_columns:
        raise ValueError(f"Các cột sau bị thiếu hoặc không phải cột số: {', '.join(invalid_columns)}.")
    if missing_strategy not in ("median", "mean", "most_frequent", "drop"):
        raise ValueError("Chiến lược xử lý giá trị thiếu không hợp lệ.")
    if outlier_method not in (None, "iqr", "zscore"):
        raise ValueError("Phương pháp xử lý ngoại lai không hợp lệ.")
    if sample_size is not None and (not isinstance(sample_size, (int, np.integer)) or sample_size < 1):
        raise ValueError("Số dòng lấy mẫu phải là số nguyên dương.")

    n_rows_before = len(df)
    steps_log = []
    work = df.copy()
    if drop_duplicates:
        work = work.drop_duplicates()
        steps_log.append(f"Bỏ {n_rows_before - len(work)} dòng trùng hoàn toàn.")

    work[columns] = work[columns].replace([np.inf, -np.inf], np.nan)
    if missing_strategy == "drop":
        n_before_drop = len(work)
        work = work.dropna(subset=columns)
        steps_log.append(f"Bỏ {n_before_drop - len(work)} dòng thiếu đặc trưng.")
    if work.empty:
        raise ValueError("Không còn dòng dữ liệu để tiền xử lý.")

    fill_values = {}
    for column in columns:
        observed = work[column].dropna()
        if observed.empty:
            raise ValueError(f"Cột {column} thiếu toàn bộ giá trị, không thể tính giá trị điền.")
        if missing_strategy == "mean":
            value = observed.mean()
        elif missing_strategy == "most_frequent":
            value = observed.mode().iloc[0]
        else:
            # Median cũng là giá trị dự phòng cho khách mới khi huấn luyện dùng drop.
            value = observed.median()
        fill_values[column] = float(value)
        n_missing = int(work[column].isna().sum())
        if n_missing:
            work[column] = work[column].astype(float).fillna(fill_values[column])
            steps_log.append(f"Điền {n_missing} giá trị thiếu ở {column} bằng {value:.3f}.")

    if outlier_method is not None:
        features = work[columns]
        if outlier_method == "iqr":
            q1 = features.quantile(0.25)
            q3 = features.quantile(0.75)
            iqr = q3 - q1
            is_outlier = (features.lt(q1 - 1.5 * iqr) | features.gt(q3 + 1.5 * iqr)).any(axis=1)
        else:
            std = features.std(ddof=0).replace(0, 1)
            zscore = (features - features.mean()) / std
            is_outlier = zscore.abs().gt(3).any(axis=1)
        work = work.loc[~is_outlier]
        steps_log.append(f"Loại {int(is_outlier.sum())} dòng ngoại lai bằng {outlier_method}.")
    if work.empty:
        raise ValueError("Không còn dòng dữ liệu sau khi loại ngoại lai.")

    if sample_size is not None and len(work) > sample_size:
        work = work.sample(n=sample_size, random_state=random_state).sort_index()
        steps_log.append(f"Lấy mẫu {sample_size} dòng với random_state={random_state}.")

    df_clean = work.reset_index(drop=True)
    X = df_clean[columns].astype(float)
    log_columns = [column for column in (log_transform_columns or []) if column in columns]
    for column in log_columns:
        X[column] = np.log1p(X[column].clip(lower=0))
    if log_columns:
        steps_log.append(f"Áp dụng log1p cho {', '.join(log_columns)}.")

    return {
        "df_clean": df_clean,
        "X": X,
        "feature_columns": columns,
        "preprocess_config": {
            "feature_columns": columns.copy(),
            "missing_strategy": missing_strategy,
            "fill_values": fill_values,
            "log_transform_columns": log_columns,
            "outlier_method": outlier_method,
            "feature_ranges": {
                column: (float(df_clean[column].min()), float(df_clean[column].max()))
                for column in columns
            },
        },
        "n_rows_before": n_rows_before,
        "n_rows_after": len(df_clean),
        "n_removed": n_rows_before - len(df_clean),
        "steps_log": steps_log,
    }

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
    from sklearn.preprocessing import StandardScaler

    columns = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    try:
        values = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dữ liệu chuẩn hóa phải chứa giá trị số.") from exc
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Cần ma trận số hai chiều, có dữ liệu và chỉ chứa giá trị hữu hạn.")

    try:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(values)
        else:
            scaler_input = X if columns is not None and hasattr(scaler, "feature_names_in_") else values
            X_scaled = scaler.transform(scaler_input)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Không thể chuẩn hóa. Hãy kiểm tra scaler đã fit và thứ tự cột.") from exc

    return {"X_scaled": np.asarray(X_scaled), "scaler": scaler, "feature_columns": columns}
