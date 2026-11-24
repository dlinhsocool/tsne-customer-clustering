"""models/data_processing.py - Nạp, kiểm tra, tiền xử lý và chuẩn hóa dữ liệu khách hàng."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ID_HINTS = ("id", "customer_id", "cust_id", "user_id", "ma_kh", "stt")
LABEL_HINTS = ("segment", "cluster", "label", "true_label", "target", "nhom")


def load_dataset(
    filepath: str | Path | io.IOBase,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Đọc CSV/Excel từ đường dẫn hoặc file-like (Streamlit UploadedFile, StringIO...).

    `sep` và `encoding` được hỗ trợ để khớp với trang tải dữ liệu.
    Với Excel, hai tham số này không được sử dụng.
    """
    if filepath is None:
        raise ValueError("Bạn chưa chọn tệp dữ liệu.")

    is_file_like = hasattr(filepath, "read")
    if is_file_like:
        name = str(getattr(filepath, "name", "")).lower()
        suffix = Path(name).suffix.lower() if name else ".csv"
        if hasattr(filepath, "seek"):
            filepath.seek(0)

        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(filepath)
        else:
            read_kwargs: dict[str, Any] = {"encoding": encoding}
            if sep is None:
                read_kwargs.update({"sep": None, "engine": "python"})
            else:
                read_kwargs["sep"] = sep
            df = pd.read_csv(filepath, **read_kwargs)
    else:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            read_kwargs = {"encoding": encoding}
            if sep is None:
                read_kwargs.update({"sep": None, "engine": "python"})
            else:
                read_kwargs["sep"] = sep
            df = pd.read_csv(path, **read_kwargs)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Định dạng tệp không được hỗ trợ: {suffix}")

    if df.empty:
        raise ValueError("Tập dữ liệu rỗng.")

    # Loại BOM ở đầu tên cột (thường gặp khi CSV lưu dạng UTF-8-SIG).
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Kiểm tra tính hợp lệ và thống kê chi tiết các nhóm cột của tập dữ liệu."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "is_valid": False,
            "valid": False,
            "n_rows": 0,
            "n_cols": 0,
            "columns": [],
            "dtypes": {},
            "missing_per_column": {},
            "missing_values": {},
            "n_duplicates": 0,
            "duplicate_rows": 0,
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "id_columns": [],
            "errors": ["DataFrame rỗng hoặc không hợp lệ."],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []

    if len(df) < 10:
        errors.append("Tập dữ liệu có ít hơn 10 dòng, không đủ để phân cụm.")
    if df.shape[1] < 2:
        errors.append("Tập dữ liệu cần ít nhất 2 cột đặc trưng.")

    missing_by_col = df.isna().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_by_col.items()}
    high_missing = missing_by_col[missing_by_col / len(df) > 0.5].index.tolist()
    if high_missing:
        warnings.append(f"Các cột thiếu trên 50% giá trị: {high_missing}")

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        warnings.append(f"Phát hiện {dup_count} dòng trùng lặp hoàn toàn.")

    id_cols: list[str] = []
    datetime_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        c_low = col.lower().strip()
        if any(c_low == h or c_low.endswith(f"_{h}") for h in ID_HINTS):
            id_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif "date" in c_low or "ngay" in c_low or "time" in c_low:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.7:
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "valid": is_valid,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_per_column": missing_dict,
        "missing_values": {k: v for k, v in missing_dict.items() if v > 0},
        "n_duplicates": dup_count,
        "duplicate_rows": dup_count,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "id_columns": id_cols,
        "errors": errors,
        "warnings": warnings,
    }


def _extract_date_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.Timestamp | None]:
    """Chuyển cột ngày thành đặc trưng số; tránh One-Hot từng ngày riêng lẻ."""
    out = df.copy()
    date_cols: list[str] = []

    for col in out.columns:
        col_low = str(col).lower().strip()
        is_date_named = any(token in col_low for token in ("date", "ngay", "time"))
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            date_cols.append(col)
            continue
        if not is_date_named:
            continue

        source = out[col]
        non_empty = source.notna() & source.astype(str).str.strip().ne("")
        if int(non_empty.sum()) == 0:
            continue

        parsed = pd.to_datetime(source, errors="coerce", format="mixed")
        parse_ratio = float(parsed[non_empty].notna().mean())
        # Chỉ coi là ngày khi phần lớn giá trị không rỗng parse được.
        if parse_ratio >= 0.80:
            out[col] = parsed
            date_cols.append(col)

    if not date_cols:
        return out, [], None

    valid_maxes = [out[col].max() for col in date_cols if out[col].notna().any()]
    if not valid_maxes:
        return out.drop(columns=date_cols), date_cols, None

    ref_date = max(valid_maxes)
    first_col = next((col for col in date_cols if "first" in str(col).lower()), None)
    last_col = next((col for col in date_cols if "last" in str(col).lower()), None)

    if last_col is not None:
        out["recency_days"] = (
            ref_date - out[last_col]
        ).dt.days.fillna(0).clip(lower=0)
    if first_col is not None:
        out["tenure_days"] = (
            ref_date - out[first_col]
        ).dt.days.fillna(0).clip(lower=0)
    if first_col is not None and last_col is not None:
        out["active_span_days"] = (
            out[last_col] - out[first_col]
        ).dt.days.fillna(0).clip(lower=0)

    for col in date_cols:
        if col not in {first_col, last_col}:
            out[f"{col}_days_ago"] = (
                ref_date - out[col]
            ).dt.days.fillna(0).clip(lower=0)

    out = out.drop(columns=date_cols)
    return out, date_cols, ref_date


def preprocess_data(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    log_transform_columns: list[str] | None = None,
    encode_categorical: bool = True,
    max_categories: int = 20,
    sample_size: int | None = None,
    random_state: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Làm sạch, trích xuất đặc trưng ngày tháng, log-transform và mã hóa biến phân loại."""
    n_rows_before = int(len(df))
    steps_log: list[str] = []

    df_work = df.copy().drop_duplicates().reset_index(drop=True)
    if len(df_work) < n_rows_before:
        steps_log.append(f"Loại bỏ {n_rows_before - len(df_work)} dòng trùng lặp.")

    if sample_size is not None and 0 < int(sample_size) < len(df_work):
        df_work = df_work.sample(n=int(sample_size), random_state=int(random_state)).reset_index(drop=True)
        steps_log.append(f"Lấy mẫu ngẫu nhiên {len(df_work)} dòng.")

    if feature_columns:
        selected_cols = [c for c in feature_columns if c in df_work.columns]
    else:
        auto_excluded: list[str] = []
        for col in df_work.columns:
            c_low = col.lower().strip()
            if any(c_low == h or c_low.endswith(f"_{h}") for h in ID_HINTS):
                auto_excluded.append(col)
            elif any(h in c_low for h in LABEL_HINTS):
                auto_excluded.append(col)
        user_drops = [c for c in (drop_columns or []) if c in df_work.columns]
        drop_set = set(auto_excluded + user_drops)
        selected_cols = [c for c in df_work.columns if c not in drop_set]

    excluded_cols = [c for c in df_work.columns if c not in selected_cols]

    df_feat_base, extracted_date_cols, ref_date = _extract_date_features(df_work[selected_cols])
    if extracted_date_cols:
        steps_log.append(f"Quy đổi {extracted_date_cols} thành đặc trưng số (recency_days, tenure_days...).")

    num_cols = df_feat_base.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_feat_base.select_dtypes(exclude=[np.number]).columns.tolist()

    medians: dict[str, float] = {}
    for col in num_cols:
        median_val = df_feat_base[col].median()
        fill_val = 0.0 if pd.isna(median_val) else float(median_val)
        medians[col] = fill_val
        if df_feat_base[col].isna().any():
            df_feat_base[col] = df_feat_base[col].fillna(fill_val)

    applied_log_cols: list[str] = []
    for col in (log_transform_columns or []):
        if col in df_feat_base.columns and pd.api.types.is_numeric_dtype(df_feat_base[col]):
            df_feat_base[col] = np.log1p(np.maximum(df_feat_base[col].to_numpy(dtype=float), 0.0))
            applied_log_cols.append(col)
    if applied_log_cols:
        steps_log.append(f"Áp dụng log1p cho các cột: {applied_log_cols}.")

    dropped_high_card: list[str] = []
    for col in cat_cols:
        df_feat_base[col] = df_feat_base[col].fillna("Unknown").astype(str)
        if df_feat_base[col].nunique() > max_categories:
            dropped_high_card.append(col)

    if dropped_high_card:
        df_feat_base = df_feat_base.drop(columns=dropped_high_card)
        cat_cols = [c for c in cat_cols if c not in dropped_high_card]
        steps_log.append(f"Bỏ cột phân loại quá nhiều giá trị (> {max_categories}): {dropped_high_card}.")

    if encode_categorical and cat_cols:
        X_df = pd.get_dummies(df_feat_base, columns=cat_cols, drop_first=False, dtype=float)
        steps_log.append(f"One-Hot Encode cho {cat_cols} -> {X_df.shape[1]} đặc trưng.")
    else:
        X_df = df_feat_base[num_cols].astype(float)

    final_feature_columns = X_df.columns.tolist()

    # Ghi đè cả những cột vốn đã tồn tại trong dữ liệu gốc.
    # Nếu chỉ thêm cột chưa tồn tại, df_clean có thể giữ lại giá trị chưa fillna,
    # chưa log-transform hoặc chưa được xử lý như ma trận X_df.
    df_clean = df_work.copy()
    for col in final_feature_columns:
        df_clean[col] = X_df[col].to_numpy(copy=True)

    n_rows_after = int(len(X_df))
    preprocess_config = {
        "input_columns": selected_cols,
        "feature_columns": final_feature_columns,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "date_columns": extracted_date_cols,
        "ref_date": str(ref_date) if ref_date is not None else None,
        "log_transform_columns": applied_log_cols,
        "medians": medians,
        "encode_categorical": encode_categorical,
    }

    return {
        "df_clean": df_clean,
        "X": X_df,
        "X_processed": X_df.to_numpy(dtype=float),
        "feature_columns": final_feature_columns,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "date_columns": extracted_date_cols,
        "log_transform_columns": applied_log_cols,
        "excluded_columns": excluded_cols + dropped_high_card,
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "n_removed": n_rows_before - n_rows_after,
        "steps_log": steps_log,
        "preprocess_config": preprocess_config,
    }


def scale_data(
    X: np.ndarray | pd.DataFrame,
    method: str = "standard",
    log_transform_skewed: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Chuẩn hóa dữ liệu (tùy chọn log1p cho biến lệch phải mạnh giúp cụm tách đều hơn)."""
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    X_arr = np.asarray(X, dtype=float).copy()

    log_cols_idx: list[int] = []
    if log_transform_skewed and X_arr.shape[0] > 5:
        for j in range(X_arr.shape[1]):
            col_vals = X_arr[:, j]
            if np.nanmin(col_vals) >= 0 and np.nanmax(col_vals) > 10:
                s = pd.Series(col_vals)
                if float(s.skew()) > 1.5:
                    X_arr[:, j] = np.log1p(col_vals)
                    log_cols_idx.append(j)

    m = method.lower()
    if m == "robust":
        scaler = RobustScaler()
    elif m == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_arr)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaler._custom_log_cols_idx = log_cols_idx

    return {
        "X_scaled": X_scaled,
        "scaler": scaler,
        "method": method,
        "log_cols_idx": log_cols_idx,
        "feature_columns": feature_names,
    }