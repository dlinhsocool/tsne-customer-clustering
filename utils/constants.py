"""
utils/constants.py
==================
Hằng số dùng CHUNG cho toàn dự án (frontend + backend).
Phiên bản hợp đồng: 2.0 (Tách luồng ứng dụng chính và luồng Benchmark 4 không gian).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Thông tin chung
# ---------------------------------------------------------------------------
APP_TITLE = "Customer Segmentation"
CONTRACT_VERSION = "2.1"

RANDOM_STATE = 42
NOISE_LABEL = -1

# True : hàm backend chưa viết (NotImplementedError) -> giao diện tự dùng mock + hiện cảnh báo.
# False: hàm chưa viết -> giao diện báo lỗi và dừng (dùng khi nộp bài).
USE_MOCK_WHEN_NOT_IMPLEMENTED = False

# ---------------------------------------------------------------------------
# 2. Thuật toán và tham số mặc định
# ---------------------------------------------------------------------------
ALGORITHMS = {"kmeans": "K-Means", "gmm": "GMM", "dbscan": "DBSCAN"}

SUPPORTS_PREDICT = {"kmeans": True, "gmm": True, "dbscan": False}

DEFAULT_PARAMS = {
    "kmeans": {"n_clusters": 5, "n_init": 10, "max_iter": 300},
    "gmm": {"n_components": 5, "covariance_type": "full", "max_iter": 100},
    "dbscan": {"eps": None, "min_samples": 5},  # eps=None -> tự ước lượng theo k-NN từng không gian
}

# Bốn không gian dữ liệu chuẩn hóa trong toàn hệ thống
FEATURE_SPACES = {
    "original": "Dữ liệu gốc đã chuẩn hóa (X_scaled)",
    "pca": "Dữ liệu sau PCA (X_pca)",
    "umap": "Dữ liệu sau UMAP (X_umap)",
    "tsne": "Embedding t-SNE (X_tsne)",
}

PCA_DEFAULT_VARIANCE = 0.95
UMAP_DEFAULTS = {
    "n_components": 5,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "euclidean",
}
TSNE_DEFAULTS = {
    "perplexity": 30.0,
    "learning_rate": 200.0,
    "n_iter": 1000,
}

# ---------------------------------------------------------------------------
# 3. Tiền xử lý & hiển thị cột
# ---------------------------------------------------------------------------
# Cột chứa các từ này là định danh/nhãn/nhiều chiều -> không chọn mặc định làm đặc trưng
NON_FEATURE_HINTS = ("id", "name", "email", "segment", "label", "true", "nhom", "city", "country")

# Bộ cột khuyến nghị khi tải file thô 20 cột (synthetic_customers_20cols.csv). Kết quả: 13 đặc trưng
#   recency_days, n_orders, total_spent, loyalty_points, support_tickets, subscription + 7 cột danh mục.
# `avg_order_value` được thay bằng `n_orders` (= total_spent / avg_order_value) để tránh cộng tuyến.
RECOMMENDED_FEATURES = [
    "first_purchase_date",
    "last_purchase_date",
    "total_spent",
    "avg_order_value",
    "loyalty_points",
    "support_tickets",
    "subscription",
    "favorite_category",
]
# Cột nhân khẩu/nhiễu trong file 20 cột: độc lập ngẫu nhiên với hành vi mua, KHÔNG chọn mặc định.
# Có thể thêm thủ công để quan sát ảnh hưởng của chiều nhiễu lên phân cụm và giảm chiều.
OPTIONAL_NOISE_FEATURES = ["age", "income", "credit_score", "gender", "device_type", "marketing_opt_in", "churn_risk"]

# Ngày tham chiếu cố định cho recency (= ngày mua muộn nhất của bộ dữ liệu + 1 ngày)
REFERENCE_DATE = "2025-05-05"

FEATURE_DISPLAY = {
    "n_orders": "Số đơn hàng (Frequency)",
    "total_spent": "Tổng chi tiêu",
    "avg_order_value": "Giá trị đơn TB",
    "loyalty_points": "Điểm thân thiết",
    "support_tickets": "Số ticket hỗ trợ",
    "churn_risk": "Rủi ro rời bỏ",
    "subscription": "Đăng ký thành viên",
    "favorite_category": "Danh mục yêu thích",
    "first_purchase_date": "Ngày mua đầu tiên",
    "last_purchase_date": "Ngày mua gần nhất",
    "recency_days": "Số ngày từ lần mua cuối (Recency)",
    "tenure_days": "Thời gian gắn bó (Tenure)",
    # Tương thích dữ liệu mẫu giả lập
    "spending": "Spending",
    "purchase_frequency": "Purchase Frequency",
    "online_time": "Online Time",
    "age": "Age",
    "income": "Income",
    "cancellation_rate": "Cancellation Rate",
}

# ---------------------------------------------------------------------------
# 4. Màu sắc
# ---------------------------------------------------------------------------
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#EECA3B", "#9D755D", "#FF9DA6", "#79706E",
]
NOISE_COLOR = "#B0B0B0"
NEW_POINT_COLOR = "#111111"


# ---------------------------------------------------------------------------
# 5. Khóa st.session_state (Tách riêng Luồng chính & Luồng Benchmark)
# ---------------------------------------------------------------------------
class SK:
    # Luồng chính
    RAW_DF = "raw_df"
    IS_SAMPLE = "is_sample_data"
    UPLOAD_SIG = "upload_signature"
    VALIDATION = "validation_result"
    RAW_FEATURE_COLUMNS = "raw_feature_columns"
    FEATURE_COLUMNS = "feature_columns"
    PREP = "prep_result"
    SCALE = "scale_result"
    PCA = "pca_result"
    MAIN_TSNE_2D = "main_tsne_2d_result"
    MAIN_TSNE_3D = "main_tsne_3d_result"
    TSNE_2D = MAIN_TSNE_2D  # bí danh tương thích ngược
    TSNE_3D = MAIN_TSNE_3D  # bí danh tương thích ngược
    TSNE_INPUT_SPACE = "tsne_input_space"
    CLUSTER_RESULTS = "cluster_results"
    ACTIVE_ALGO = "active_algorithm"
    EVAL_TABLE = "evaluation_table"
    PROFILES = "profiles"
    CLUSTER_NAMES = "cluster_names"
    NEW_CUSTOMER = "new_customer_result"

    # Luồng Benchmark (tách biệt hoàn toàn khỏi luồng chính)
    BENCHMARK_SAMPLE = "benchmark_sample"
    BENCHMARK_REPRESENTATIONS = "benchmark_representations"
    BENCHMARK_REDUCTION_TABLE = "benchmark_reduction_table"
    BENCHMARK_CLUSTER_TABLE = "benchmark_cluster_table"
    BENCHMARK_CLUSTER_RESULTS = "benchmark_cluster_results"
    BENCHMARK_REPEATS = "benchmark_repeat_tables"

    # Hệ thống
    RANDOM_STATE = "random_state"
    BACKEND_STATUS = "backend_status"


# ---------------------------------------------------------------------------
# 6. HỢP ĐỒNG OUTPUT v2.0
# ---------------------------------------------------------------------------
CONTRACT_KEYS: dict[str, list[str]] = {
    "validate_dataset": [
        "is_valid", "n_rows", "n_cols", "columns", "dtypes", "missing_per_column",
        "n_duplicates", "numeric_columns", "categorical_columns",
        "datetime_columns", "id_columns", "errors", "warnings",
    ],
    "preprocess_data": [
        "df_clean", "X", "feature_columns", "preprocess_config",
        "n_rows_before", "n_rows_after", "n_removed", "steps_log",
    ],
    "scale_data": ["X_scaled", "scaler", "feature_columns"],
    "run_pca": [
        "pca_model", "X_pca", "n_components", "explained_variance_ratio",
        "cumulative_variance", "components", "runtime_seconds",
    ],
    "run_umap": [
        "model", "embedding", "n_components", "params",
        "runtime_seconds", "backend",
    ],
    "run_tsne": [
        "embedding", "n_components", "perplexity", "learning_rate", "n_iter",
        "kl_divergence", "tsne_model", "backend", "device", "runtime_seconds",
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
    "benchmark_clustering_spaces": [
        "table", "results",
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

CONTRACT_COLUMNS: dict[str, list[str]] = {
    "get_explained_variance": ["component", "explained_variance_ratio", "cumulative_variance"],
    "compare_models": [
        "algorithm", "n_clusters", "n_noise", "silhouette_score",
        "davies_bouldin_score", "ari_score", "runtime_seconds",
    ],
    "compare_dimensionality_reduction_method": [
        "method", "n_components", "trustworthiness", "continuity", "knn_preservation",
        "spearman_rank_correlation", "runtime_seconds", "stability_score",
        "n_samples", "n_neighbors", "note",
    ],
}

CONTRACT_TYPES: dict[str, str] = {
    "load_dataset": "DataFrame",
}