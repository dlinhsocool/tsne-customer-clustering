"""
utils/constants.py
==================
Hằng số dùng CHUNG cho toàn dự án (frontend + backend).

QUY ĐỊNH: không hardcode lại các giá trị này ở file khác. Cần đổi thì sửa ở đây
và báo cả nhóm. Tên key trong CONTRACT_KEYS là "hợp đồng" giữa các module.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Thông tin chung
# ---------------------------------------------------------------------------
APP_TITLE = "Customer Segmentation"
CONTRACT_VERSION = "1.0"

RANDOM_STATE = 42          # seed chung. MỌI hàm có yếu tố ngẫu nhiên phải nhận random_state.
NOISE_LABEL = -1           # nhãn noise của DBSCAN. Dùng đúng giá trị này ở mọi nơi.

# True : hàm backend chưa viết (NotImplementedError) -> giao diện tự dùng mock + hiện cảnh báo.
# False: hàm chưa viết -> giao diện báo lỗi và dừng (dùng khi nộp bài, để không lẫn mock).
USE_MOCK_WHEN_NOT_IMPLEMENTED = True

# ---------------------------------------------------------------------------
# 2. Thuật toán và tham số mặc định
# ---------------------------------------------------------------------------
# key (dùng trong code, snake_case) -> tên hiển thị
ALGORITHMS = {"kmeans": "K-Means", "gmm": "GMM", "dbscan": "DBSCAN"}

# Thuật toán nào có predict() trực tiếp. DBSCAN KHÔNG có.
SUPPORTS_PREDICT = {"kmeans": True, "gmm": True, "dbscan": False}

DEFAULT_PARAMS = {
    "kmeans": {"n_clusters": 5, "n_init": 10, "max_iter": 300},
    "gmm": {"n_components": 5, "covariance_type": "full", "max_iter": 100},
    "dbscan": {"eps": 0.5, "min_samples": 5},
}

# Không gian dữ liệu mà thuật toán phân cụm chạy trên đó (KHÔNG phải tọa độ t-SNE).
FEATURE_SPACES = {"scaled": "Dữ liệu đã chuẩn hóa", "pca": "Dữ liệu sau PCA"}

PCA_DEFAULT_VARIANCE = 0.95
TSNE_DEFAULTS = {"perplexity": 30.0, "learning_rate": 200.0, "n_iter": 1000}

# ---------------------------------------------------------------------------
# 3. Tiền xử lý
# ---------------------------------------------------------------------------
MISSING_STRATEGIES = ["median", "mean", "most_frequent", "drop"]
OUTLIER_METHODS = [None, "iqr", "zscore"]

# Cột có tên chứa các từ này sẽ KHÔNG được chọn sẵn làm đặc trưng (vẫn chọn tay được).
NON_FEATURE_HINTS = ("id", "name", "email", "segment", "label", "date", "true", "city", "country")

# Tên hiển thị (tùy chọn) cho cột. Cột không có trong dict sẽ hiện nguyên tên.
FEATURE_DISPLAY = {
    "spending": "Spending",
    "purchase_frequency": "Purchase Frequency",
    "online_time": "Online Time",
    "age": "Age",
    "income": "Income",
    "cancellation_rate": "Cancellation Rate",
}

# ---------------------------------------------------------------------------
# 4. Màu sắc (trung tính, dễ đọc)
# ---------------------------------------------------------------------------
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#EECA3B", "#9D755D", "#FF9DA6", "#79706E"]
NOISE_COLOR = "#B0B0B0"
NEW_POINT_COLOR = "#111111"


# ---------------------------------------------------------------------------
# 5. Khóa st.session_state (dùng SK.<TÊN>, không gõ chuỗi tay)
# ---------------------------------------------------------------------------
class SK:
    RAW_DF = "raw_df"                    # DataFrame gốc vừa tải
    IS_SAMPLE = "is_sample_data"         # True nếu là dữ liệu mẫu giả lập
    UPLOAD_SIG = "upload_signature"
    VALIDATION = "validation_result"
    FEATURE_COLUMNS = "feature_columns"
    PREP = "prep_result"                 # kết quả preprocess_data
    SCALE = "scale_result"               # kết quả scale_data
    PCA = "pca_result"                   # kết quả run_pca
    TSNE_2D = "tsne_2d_result"
    TSNE_3D = "tsne_3d_result"
    TSNE_INPUT_SPACE = "tsne_input_space"  # "scaled" | "pca"
    CLUSTER_RESULTS = "cluster_results"    # dict {algorithm_key: ClusteringResult}
    ACTIVE_ALGO = "active_algorithm"
    EVAL_TABLE = "evaluation_table"      # DataFrame từ compare_models
    PROFILES = "profiles"                # dict {algorithm_key: {"profiles":..., "descriptions":...}}
    CLUSTER_NAMES = "cluster_names"      # dict {algorithm_key: {cluster_id: tên do người dùng đặt}}
    NEW_CUSTOMER = "new_customer_result"
    RANDOM_STATE = "random_state"
    BACKEND_STATUS = "backend_status"    # dict {function_name: "real" | "mock"}


# ---------------------------------------------------------------------------
# 6. HỢP ĐỒNG OUTPUT: key bắt buộc của dict mà mỗi hàm trả về
#    (utils/contract.py kiểm tra tự động; thiếu key => giao diện báo lỗi)
# ---------------------------------------------------------------------------
CONTRACT_KEYS: dict[str, list[str]] = {
    "validate_dataset": [
        "is_valid", "n_rows", "n_cols", "columns", "dtypes", "missing_per_column",
        "n_duplicates", "numeric_columns", "categorical_columns", "errors", "warnings",
    ],
    "preprocess_data": [
        "df_clean", "X", "feature_columns", "preprocess_config",
        "n_rows_before", "n_rows_after", "n_removed", "steps_log",
    ],
    "scale_data": ["X_scaled", "scaler", "feature_columns"],
    "run_pca": [
        "pca_model", "X_pca", "n_components", "explained_variance_ratio",
        "cumulative_variance", "components",
    ],
    "run_tsne": [
        "embedding", "n_components", "perplexity", "learning_rate", "n_iter",
        "kl_divergence", "tsne_model", "backend",
    ],
    "run_clustering": [
        "algorithm", "model", "labels", "n_clusters", "n_noise", "cluster_sizes",
        "cluster_centers", "probabilities", "params", "feature_space",
        "supports_predict", "runtime_seconds", "extra",
    ],
    "evaluate_clustering": [
        "silhouette_score", "davies_bouldin_score", "ari_score", "n_clusters",
        "n_noise", "noise_ratio", "runtime_seconds", "note",
    ],
    "calculate_cluster_profiles": ["profile_table", "z_scores", "global_mean", "cluster_ids"],
    "generate_cluster_description": ["cluster_id", "name", "description", "top_features"],
    "preprocess_new_customer": [
        "X_new_raw", "X_new_scaled", "X_new_pca", "out_of_range_features", "warnings",
    ],
    "predict_customer_cluster": [
        "cluster", "is_noise", "method_used", "confidence", "probabilities",
        "neighbor_indices", "neighbor_distances", "mean_neighbor_distance",
        "is_outlier", "warnings",
    ],
    "transform_new_customer": [
        "coordinates", "n_components", "method_used", "neighbor_indices", "warnings",
    ],
}

# Hàm trả về DataFrame: cột bắt buộc
CONTRACT_COLUMNS: dict[str, list[str]] = {
    "get_explained_variance": ["component", "explained_variance_ratio", "cumulative_variance"],
    "compare_models": [
        "algorithm", "n_clusters", "n_noise", "silhouette_score",
        "davies_bouldin_score", "ari_score", "runtime_seconds",
    ],
}

# Hàm trả về DataFrame (không kiểm key/cột) hoặc kiểu đơn giản
CONTRACT_TYPES: dict[str, str] = {
    "load_dataset": "DataFrame",
}
