"""models/data_processing.py - Nạp, kiểm tra, tiền xử lý và chuẩn hóa dữ liệu khách hàng.

Quy ước của pipeline (v2):
  * Ngày tham chiếu cố định: mặc định = ngày mua muộn nhất + 1 ngày, được LƯU vào preprocess_config
    để khách mới được tính recency đúng như khách cũ.
  * Cột ngày chỉ sinh `recency_days` (tùy chọn thêm `tenure_days`). Không sinh `active_span_days`
    vì nó là tổ hợp tuyến tính của hai cột kia.
  * Nếu có cả `total_spent` và `avg_order_value`, đặc trưng `n_orders = round(total_spent / avg_order_value)`
    (tần suất mua, tức Frequency của RFM) thay cho `avg_order_value`, tránh cộng tuyến.
  * Cột boolean (subscription) -> một cột 0/1. Biến phân loại -> One-Hot với danh sách mức cố định.
  * `df_clean` luôn ở thang đo gốc (không ghi đè giá trị log).
  * Cột 0/1 (nhị phân, One-Hot) KHÔNG bị StandardScaler; chỉ cột liên tục được chuẩn hóa.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ID_HINTS = ("id", "customer_id", "cust_id", "user_id", "ma_kh", "stt", "name", "email")
LABEL_HINTS = ("segment", "cluster", "label", "true_label", "target", "nhom")
HIGH_CARDINALITY_MIN = 50  # cột chữ có nhiều hơn số mức này được coi là định danh / nhiều chiều


# ---------------------------------------------------------------------------
# 1. Nạp & kiểm tra
# ---------------------------------------------------------------------------
def load_dataset(
    filepath: str | Path | io.IOBase,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Đọc CSV/Excel từ đường dẫn hoặc file-like (Streamlit UploadedFile, StringIO...)."""
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
    df.columns = [str(column).lstrip("﻿").strip() for column in df.columns]
    return df


def _is_id_name(col: str) -> bool:
    c_low = col.lower().strip()
    return any(c_low == h or c_low.endswith(f"_{h}") for h in ID_HINTS)


def _is_label_name(col: str) -> bool:
    c_low = col.lower().strip()
    return any(h in c_low for h in LABEL_HINTS)


def _is_text_dtype(series: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


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
        series = df[col]
        c_low = col.lower().strip()
        if _is_id_name(col):
            id_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        elif "date" in c_low or "ngay" in c_low or "time" in c_low:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.7:
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(series):
            categorical_cols.append(col)  # boolean là biến phân loại nhị phân, không phải số
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        elif series.nunique(dropna=True) > HIGH_CARDINALITY_MIN:
            id_cols.append(col)  # chữ có quá nhiều mức: coi như định danh / hiển thị
        else:
            categorical_cols.append(col)

    dup_ids = [c for c in id_cols if c.lower().endswith("id") and df[c].duplicated().any()]
    for col in dup_ids:
        warnings.append(
            f"Cột '{col}' có {int(df[col].duplicated().sum()):,} giá trị lặp: không dùng làm khóa. "
            "Quy trình giữ mọi dòng và đánh số theo thứ tự dòng."
        )

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


# ---------------------------------------------------------------------------
# 2. Tiền xử lý
# ---------------------------------------------------------------------------
def _parse_date_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Chuyển các cột ngày sang datetime; trả về (df, danh sách cột ngày)."""
    out = df.copy()
    date_cols: list[str] = []
    for col in columns:
        low = str(col).lower().strip()
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            date_cols.append(col)
            continue
        if not any(token in low for token in ("date", "ngay", "time")):
            continue
        source = out[col]
        non_empty = source.notna() & source.astype(str).str.strip().ne("")
        if int(non_empty.sum()) == 0:
            continue
        parsed = pd.to_datetime(source, errors="coerce", format="mixed")
        if float(parsed[non_empty].notna().mean()) >= 0.80:
            out[col] = parsed
            date_cols.append(col)
    return out, date_cols


def _as_binary(series: pd.Series) -> pd.Series | None:
    """Trả về chuỗi 0/1 (float) nếu cột là boolean / True-False / 0-1; ngược lại None."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        values = set(pd.unique(series.dropna()))
        if values and values <= {0, 1, 0.0, 1.0}:
            return series.astype(float)
        return None
    mapping = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0}
    lowered = series.dropna().astype(str).str.strip().str.lower()
    if len(lowered) and set(lowered.unique()) <= set(mapping):
        return series.astype(str).str.strip().str.lower().map(mapping).astype(float)
    return None


def _is_integer_valued(values: pd.Series) -> bool:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return bool(arr.size) and bool(np.allclose(arr, np.round(arr)))


def preprocess_data(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    log_transform_columns: list[str] | None = None,
    encode_categorical: bool = True,
    max_categories: int = 20,
    sample_size: int | None = None,
    random_state: int = 42,
    drop_duplicates: bool = True,
    ref_date: str | pd.Timestamp | None = None,
    derive_n_orders: bool = True,
    include_tenure: bool = False,
) -> dict[str, Any]:
    """Làm sạch, tạo đặc trưng (recency, n_orders), log-transform và mã hóa biến phân loại.

    Tham số lạ bị từ chối (TypeError) thay vì bị bỏ qua âm thầm.
    """
    n_rows_before = int(len(df))
    steps_log: list[str] = []

    df_work = df.copy()
    if drop_duplicates:
        df_work = df_work.drop_duplicates()
        if len(df_work) < n_rows_before:
            steps_log.append(f"Loại bỏ {n_rows_before - len(df_work)} dòng trùng lặp hoàn toàn.")
    df_work = df_work.reset_index(drop=True)

    if sample_size is not None and 0 < int(sample_size) < len(df_work):
        df_work = df_work.sample(n=int(sample_size), random_state=int(random_state)).reset_index(drop=True)
        steps_log.append(f"Lấy mẫu ngẫu nhiên {len(df_work)} dòng (random_state={int(random_state)}).")

    if feature_columns:
        missing_cols = [c for c in feature_columns if c not in df_work.columns]
        if missing_cols:
            raise ValueError(f"Cột không có trong dữ liệu: {missing_cols}")
        selected_cols = list(feature_columns)
    else:
        user_drops = set(drop_columns or [])
        selected_cols = [
            c for c in df_work.columns
            if not _is_id_name(c) and not _is_label_name(c) and c not in user_drops
            and not (_is_text_dtype(df_work[c]) and df_work[c].nunique(dropna=True) > HIGH_CARDINALITY_MIN)
        ]
    if len(selected_cols) < 1:
        raise ValueError("Chưa chọn cột đặc trưng nào.")
    excluded_cols = [c for c in df_work.columns if c not in selected_cols]

    base, date_cols = _parse_date_columns(df_work[selected_cols], selected_cols)
    base = base.copy()
    derived_columns: dict[str, str] = {}  # tên đặc trưng sinh ra -> mô tả

    # --- 2.1 Ngày -> recency_days (+ tenure_days tùy chọn) ---------------------------------
    ref_ts: pd.Timestamp | None = None
    first_col = next((c for c in date_cols if "first" in c.lower()), None)
    last_col = next((c for c in date_cols if "last" in c.lower()), None)
    if date_cols:
        if ref_date is not None:
            ref_ts = pd.Timestamp(ref_date)
        else:
            maxes = [base[c].max() for c in date_cols if base[c].notna().any()]
            ref_ts = (max(maxes) + pd.Timedelta(days=1)) if maxes else None
        if ref_ts is None:
            raise ValueError("Không xác định được ngày tham chiếu từ các cột ngày.")

        if first_col and last_col:
            n_bad = int((base[last_col] < base[first_col]).sum())
            if n_bad:
                steps_log.append(f"Cảnh báo: {n_bad} dòng có ngày mua cuối trước ngày mua đầu.")
        if last_col:
            base["recency_days"] = (ref_ts - base[last_col]).dt.days.clip(lower=0)
            derived_columns["recency_days"] = f"ngày tham chiếu {ref_ts.date()} trừ {last_col}"
        if first_col and (include_tenure or not last_col):
            base["tenure_days"] = (ref_ts - base[first_col]).dt.days.clip(lower=0)
            derived_columns["tenure_days"] = f"ngày tham chiếu {ref_ts.date()} trừ {first_col}"
        for col in date_cols:
            if col not in {first_col, last_col}:
                base[f"{col}_days_ago"] = (ref_ts - base[col]).dt.days.clip(lower=0)
                derived_columns[f"{col}_days_ago"] = f"ngày tham chiếu trừ {col}"
        base = base.drop(columns=date_cols)
        steps_log.append(
            f"Quy đổi {date_cols} thành {[c for c in derived_columns if c.endswith('days') or c.endswith('days_ago')]} "
            f"(ngày tham chiếu cố định {ref_ts.date()})."
        )

    # --- 2.2 n_orders = total_spent / avg_order_value ---------------------------------------
    n_orders_source: list[str] | None = None
    dropped_after_derive: list[str] = []
    if (
        derive_n_orders
        and "total_spent" in base.columns
        and "avg_order_value" in base.columns
        and pd.api.types.is_numeric_dtype(base["total_spent"])
        and pd.api.types.is_numeric_dtype(base["avg_order_value"])
    ):
        avg = pd.to_numeric(base["avg_order_value"], errors="coerce").where(lambda s: s > 0)
        base["n_orders"] = (pd.to_numeric(base["total_spent"], errors="coerce") / avg).round().clip(lower=1)
        base = base.drop(columns=["avg_order_value"])
        n_orders_source = ["total_spent", "avg_order_value"]
        dropped_after_derive.append("avg_order_value")
        derived_columns["n_orders"] = "round(total_spent / avg_order_value), tối thiểu 1"
        steps_log.append(
            "Tạo n_orders = round(total_spent / avg_order_value) (Frequency); "
            "avg_order_value chỉ còn trong bảng sạch để diễn giải, không đưa vào mô hình."
        )

    # --- 2.3 Phân loại cột: nhị phân / số / chữ ----------------------------------------------
    binary_cols: list[str] = []
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in base.columns:
        as_bin = _as_binary(base[col])
        if as_bin is not None:
            base[col] = as_bin
            binary_cols.append(col)
        elif pd.api.types.is_numeric_dtype(base[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    medians: dict[str, float] = {}
    for col in num_cols:
        values = pd.to_numeric(base[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median_val = values.median()
        fill_val = 0.0 if pd.isna(median_val) else float(median_val)
        medians[col] = fill_val
        base[col] = values.fillna(fill_val).astype(float)
    for col in binary_cols:
        medians[col] = 0.0
        base[col] = base[col].fillna(0.0)
    n_filled = int(df_work[selected_cols].isna().sum().sum())
    if n_filled:
        steps_log.append(f"Điền {n_filled} ô thiếu: số bằng trung vị, nhị phân bằng 0, chữ bằng 'Unknown'.")

    dropped_high_card: list[str] = []
    category_levels: dict[str, list[str]] = {}
    for col in list(cat_cols):
        base[col] = base[col].fillna("Unknown").astype(str)
        if base[col].nunique() > max_categories:
            dropped_high_card.append(col)
    if dropped_high_card:
        base = base.drop(columns=dropped_high_card)
        cat_cols = [c for c in cat_cols if c not in dropped_high_card]
        steps_log.append(f"Bỏ cột phân loại quá nhiều giá trị (> {max_categories}): {dropped_high_card}.")

    # --- 2.4 Log-transform (chỉ áp dụng lên X, df_clean giữ thang đo gốc) -------------------
    X_num = base[num_cols].copy()
    applied_log_cols: list[str] = []
    for col in (log_transform_columns or []):
        if col in X_num.columns:
            X_num[col] = np.log1p(np.maximum(X_num[col].to_numpy(dtype=float), 0.0))
            applied_log_cols.append(col)
        elif col in dropped_after_derive:
            steps_log.append(f"Bỏ qua log1p cho {col}: đã được thay bằng n_orders.")
    if applied_log_cols:
        steps_log.append(f"Áp dụng log1p cho các cột: {applied_log_cols}.")

    # --- 2.5 One-Hot với danh sách mức cố định ----------------------------------------------
    onehot_df = pd.DataFrame(index=base.index)
    if encode_categorical and cat_cols:
        for col in cat_cols:
            levels = sorted(base[col].unique().tolist())
            category_levels[col] = levels
            for level in levels:
                onehot_df[f"{col}_{level}"] = (base[col] == level).astype(float)
        steps_log.append(f"One-Hot Encode {cat_cols} -> {onehot_df.shape[1]} cột.")
    else:
        cat_cols = []

    X_df = pd.concat([X_num, base[binary_cols].astype(float), onehot_df], axis=1)
    final_feature_columns = X_df.columns.tolist()

    # --- 2.6 df_clean: thang đo gốc, đủ mọi cột đặc trưng để tra cứu & profiling -------------
    df_clean = df_work.copy()
    for col in num_cols + binary_cols:
        df_clean[col] = base[col].to_numpy(copy=True)  # thang đo gốc (chưa log), đã điền thiếu
    for col in onehot_df.columns:
        df_clean[col] = onehot_df[col].to_numpy(copy=True)

    # --- 2.7 Config cho khách hàng mới -------------------------------------------------------
    raw_ranges = {c: [float(base[c].min()), float(base[c].max())] for c in num_cols}
    feature_ranges = {c: [float(X_df[c].min()), float(X_df[c].max())] for c in final_feature_columns}
    form_fields: list[dict[str, Any]] = []
    for col in num_cols:
        is_int = _is_integer_valued(base[col])
        form_fields.append({
            "name": col,
            "kind": "int" if is_int else "float",
            "min": raw_ranges[col][0],
            "max": raw_ranges[col][1],
            "default": float(round(medians[col])) if is_int else float(medians[col]),
        })
    for col in binary_cols:
        form_fields.append({"name": col, "kind": "bool", "default": bool(base[col].mean() >= 0.5)})
    for col in cat_cols:
        levels = category_levels[col]
        mode = base[col].mode().iloc[0]
        form_fields.append({"name": col, "kind": "category", "options": levels, "default": str(mode)})

    n_rows_after = int(len(X_df))
    preprocess_config = {
        "version": 2,
        "input_columns": selected_cols,
        "feature_columns": final_feature_columns,
        "numeric_columns": num_cols,
        "binary_columns": binary_cols,
        "categorical_columns": cat_cols,
        "category_levels": category_levels,
        "date_columns": date_cols,
        "first_date_column": first_col,
        "last_date_column": last_col,
        "ref_date": str(ref_ts.date()) if ref_ts is not None else None,
        "include_tenure": bool(include_tenure),
        "derived_columns": derived_columns,
        "n_orders_source": n_orders_source,
        "dropped_after_derive": dropped_after_derive,
        "log_transform_columns": applied_log_cols,
        "medians": medians,
        "raw_feature_ranges": raw_ranges,
        "feature_ranges": feature_ranges,
        "form_fields": form_fields,
        "encode_categorical": bool(encode_categorical),
    }

    return {
        "df_clean": df_clean,
        "X": X_df,
        "X_processed": X_df.to_numpy(dtype=float),
        "feature_columns": final_feature_columns,
        "numeric_columns": num_cols,
        "binary_columns": binary_cols,
        "categorical_columns": cat_cols,
        "date_columns": date_cols,
        "log_transform_columns": applied_log_cols,
        "excluded_columns": excluded_cols + dropped_high_card + dropped_after_derive,
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "n_removed": n_rows_before - n_rows_after,
        "steps_log": steps_log,
        "preprocess_config": preprocess_config,
    }


# ---------------------------------------------------------------------------
# 3. Chuẩn hóa
# ---------------------------------------------------------------------------
class PartialScaler:
    """Chuẩn hóa các cột liên tục; cột `passthrough_idx` (0/1) được giữ nguyên.

    Có `mean_`, `scale_` đủ độ dài d (cột giữ nguyên: mean 0, scale 1) và `feature_names_in_`.
    Được định nghĩa ở mức module để pickle/joblib dùng được.
    """

    def __init__(self, base: Any, passthrough_idx: list[int] | None = None) -> None:
        self.base = base
        self.passthrough_idx = sorted(int(i) for i in (passthrough_idx or []))
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None

    def _scaled_idx(self, n_features: int) -> list[int]:
        skip = set(self.passthrough_idx)
        return [j for j in range(n_features) if j not in skip]

    def fit(self, X: Any) -> "PartialScaler":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        idx = self._scaled_idx(arr.shape[1])
        if idx:
            self.base.fit(arr[:, idx])
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float).copy()
        if self.n_features_in_ is not None and arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Số cột ({arr.shape[1]}) không khớp lúc fit ({self.n_features_in_})."
            )
        idx = self._scaled_idx(arr.shape[1])
        if idx:
            arr[:, idx] = self.base.transform(arr[:, idx])
        return arr

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float).copy()
        idx = self._scaled_idx(arr.shape[1])
        if idx:
            arr[:, idx] = self.base.inverse_transform(arr[:, idx])
        return arr

    def _full(self, attr: str, neutral: float) -> np.ndarray | None:
        part = getattr(self.base, attr, None)
        if part is None or self.n_features_in_ is None:
            return None
        out = np.full(self.n_features_in_, neutral, dtype=float)
        out[self._scaled_idx(self.n_features_in_)] = part
        return out

    @property
    def mean_(self) -> np.ndarray | None:
        return self._full("mean_", 0.0)

    @property
    def scale_(self) -> np.ndarray | None:
        return self._full("scale_", 1.0)

    @property
    def data_min_(self) -> np.ndarray | None:
        return self._full("data_min_", 0.0)

    @property
    def data_max_(self) -> np.ndarray | None:
        return self._full("data_max_", 1.0)


def scale_data(
    X: np.ndarray | pd.DataFrame,
    method: str = "standard",
    log_transform_skewed: bool = False,
    passthrough_binary: bool = True,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Chuẩn hóa dữ liệu.

    - `passthrough_binary=True`: cột chỉ gồm 0/1 (nhị phân, One-Hot) giữ nguyên, không z-score.
    - `log_transform_skewed=False` (mặc định): không tự log cột nào; log phải được chọn rõ ở tiền xử lý.
      Bật True để tự log1p các cột không âm, max > 10 và skew > 1.5 (chỉ số cột lưu trong `log_cols_idx`).
    - `scaler`: truyền scaler đã fit để chỉ transform (không fit lại).
    """
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    X_arr = np.asarray(X, dtype=float).copy()

    log_cols_idx: list[int] = []
    if scaler is not None:
        log_cols_idx = [int(i) for i in getattr(scaler, "_custom_log_cols_idx", [])]
        for j in log_cols_idx:
            X_arr[:, j] = np.log1p(np.maximum(X_arr[:, j], 0.0))
        X_scaled = np.nan_to_num(np.asarray(scaler.transform(X_arr), dtype=float))
        return {
            "X_scaled": X_scaled, "scaler": scaler, "method": method,
            "log_cols_idx": log_cols_idx, "passthrough_idx": list(getattr(scaler, "passthrough_idx", [])),
            "feature_columns": feature_names,
        }

    if log_transform_skewed and X_arr.shape[0] > 5:
        for j in range(X_arr.shape[1]):
            col_vals = X_arr[:, j]
            if np.nanmin(col_vals) >= 0 and np.nanmax(col_vals) > 10:
                if float(pd.Series(col_vals).skew()) > 1.5:
                    X_arr[:, j] = np.log1p(col_vals)
                    log_cols_idx.append(j)

    passthrough_idx: list[int] = []
    if passthrough_binary:
        for j in range(X_arr.shape[1]):
            if set(np.unique(X_arr[:, j])) <= {0.0, 1.0}:
                passthrough_idx.append(j)
        if len(passthrough_idx) == X_arr.shape[1]:
            passthrough_idx = []  # toàn bộ là 0/1: chuẩn hóa bình thường

    m = method.lower()
    base = RobustScaler() if m == "robust" else MinMaxScaler() if m == "minmax" else StandardScaler()
    fitted = PartialScaler(base, passthrough_idx)
    fit_input = pd.DataFrame(X_arr, columns=feature_names) if feature_names is not None else X_arr
    X_scaled = fitted.fit_transform(fit_input)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    fitted._custom_log_cols_idx = log_cols_idx

    return {
        "X_scaled": X_scaled,
        "scaler": fitted,
        "method": method,
        "log_cols_idx": log_cols_idx,
        "passthrough_idx": passthrough_idx,
        "feature_columns": feature_names,
    }
