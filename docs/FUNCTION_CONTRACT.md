# Function Contract – Customer Segmentation (v1.0)

Tài liệu này là "hợp đồng" giữa giao diện (Streamlit) và các module xử lý. Mỗi người chỉ cần
**viết thân hàm trong file của mình**; giao diện đã gọi sẵn đúng tên hàm, đúng tham số, đúng key.

## 1. Cách làm việc (3 bước)

1. Mở file `models/<module>.py` của bạn, thay dòng `raise NotImplementedError(...)` bằng code thật.
   Giữ nguyên tên hàm, tên tham số, giá trị mặc định, tên key của dict trả về.
2. Chạy `python check_backend.py`. Hàm của bạn đổi từ `TODO` sang `OK` là đúng hợp đồng
   (đầu vào lấy từ bản mock nên không phụ thuộc hàm của người khác).
3. Chạy `streamlit run app.py`. Thanh bên trái ghi hàm nào còn là MOCK; khi hàm của bạn là "Thật"
   thì giao diện đã dùng code của bạn.

Mở Pull Request từ nhánh riêng. Không sửa `utils/`, `pages/`, `app.py`. Nếu cần đổi hợp đồng,
nhắn người phụ trách giao diện và cả nhóm cùng thống nhất.

## 2. Quy ước chung (bắt buộc)

| Quy ước | Nội dung |
|---|---|
| Thứ tự dòng | Sau `preprocess_data`, mọi thứ có `n` dòng và cùng thứ tự: `df_clean`, `X`, `X_scaled`, `X_pca`, `embedding`, `labels`. Dòng `i` luôn là khách hàng `i`. `df_clean` phải `reset_index(drop=True)`. |
| Lấy mẫu | Nếu cần giảm số dòng cho t-SNE, làm trong `preprocess_data(sample_size=...)`. Không tự lấy mẫu ở bước sau (sẽ lệch dòng). |
| Không có dữ liệu | Trả `None`, không xóa key. |
| Seed | Hàm có ngẫu nhiên nhận `random_state=42`. |
| Noise | Nhãn noise của DBSCAN là `-1` (`utils.constants.NOISE_LABEL`). K-Means/GMM không có `-1`. |
| Không gian | Phân cụm chạy trên `X_scaled` hoặc `X_pca`, **không** chạy trên tọa độ t-SNE. t-SNE chỉ để nhìn. |
| predict() | Không giả định mọi model có `predict()`. DBSCAN không có (`supports_predict=False`). |
| Kiểu dữ liệu | Ma trận số = `np.ndarray`; bảng = `pd.DataFrame`; nhãn = `np.ndarray` int 1 chiều. Điểm mới luôn có shape `(1, m)`. |
| Đặt tên | `snake_case`. Thông báo cho người dùng viết tiếng Việt. |
| Cấm | `print`, `import streamlit`, đọc file theo đường dẫn cứng trong `models/`. Lỗi dữ liệu thì `raise ValueError("thông báo tiếng Việt")`. |

## 3. Bảng phân công và hàm

| Module | File | Hàm | Input chính | Output | Người phụ trách |
|---|---|---|---|---|---|
| A. Data | `models/data_processing.py` | `load_dataset` | `file, sep, encoding` | `DataFrame` | Ngọc Đức |
| | | `validate_dataset` | `df, feature_columns` | dict (mục 4) | Ngọc Đức |
| | | `preprocess_data` | `df, feature_columns, missing_strategy, drop_duplicates, outlier_method, log_transform_columns, sample_size, random_state` | dict (mục 4) | Ngọc Đức |
| | | `scale_data` | `X, scaler=None` | dict (mục 4) | Ngọc Đức |
| B. Reduction | `models/dimensionality_reduction.py` | `run_pca` | `X_scaled, n_components, random_state` | dict | Huỳnh Lân |
| | | `get_explained_variance` | `pca_model` | `DataFrame` | Huỳnh Lân |
| | | `run_tsne` | `X_input, n_components, perplexity, learning_rate, n_iter, random_state, use_opentsne` | dict | Huỳnh Lân |
| C. Clustering | `models/clustering.py` | `run_kmeans` | `X, n_clusters, n_init, max_iter, random_state` | dict (qua `make_clustering_result`) | Duy Lĩnh |
| | | `run_gmm` | `X, n_components, covariance_type, max_iter, random_state` | dict | Duy Lĩnh |
| | | `run_dbscan` | `X, eps, min_samples` | dict | Duy Lĩnh |
| | | `run_clustering` | `X, algorithm, params, feature_space, random_state` | dict | **đã viết sẵn** |
| D. Evaluation | `models/evaluation.py` | `calculate_silhouette_score` | `X, labels, sample_size, random_state` | `float \| None` | Duy Lĩnh |
| | | `calculate_davies_bouldin_score` | `X, labels` | `float \| None` | Duy Lĩnh |
| | | `calculate_ari` | `labels_true, labels_pred` | `float \| None` | Duy Lĩnh |
| | | `evaluate_clustering` | `X, labels, labels_true, runtime_seconds` | dict | Duy Lĩnh |
| | | `compare_models` | `results, X_scaled, X_pca, labels_true` | `DataFrame` | Duy Lĩnh |
| E. Profiling | `models/customer_profiling.py` | `calculate_cluster_profiles` | `df_clean, labels, feature_columns, include_noise` | dict | Duy Lĩnh |
| | | `generate_cluster_description` | `cluster_id, profiles, feature_labels, top_n, z_threshold` | dict | Duy Lĩnh |
| F. Out-of-sample | `models/prediction.py` | `preprocess_new_customer` | `new_data, preprocess_config, scaler, pca_model` | dict | Ngọc Đức |
| | | `predict_customer_cluster` | `x_new, clustering_result, X_reference, method, k` | dict | Duy Lĩnh |
| | | `transform_new_customer` | `x_new, tsne_result, X_reference, method, k` | dict | Huỳnh Lân |

Cột "Người phụ trách" là đề xuất theo phân vai trong kịch bản; đổi lại nếu nhóm chia khác.

Hàm đã viết sẵn, **dùng không sửa**: `make_clustering_result`, `run_clustering`
(`models/clustering.py`), `get_valid_mask`, `is_computable` (`models/evaluation.py`).

## 4. Cấu trúc output (key cố định)

Danh sách key bắt buộc nằm ở `utils/constants.py` → `CONTRACT_KEYS`; giao diện tự kiểm tra và báo
lỗi nếu thiếu. Tóm tắt:

**`validate_dataset`**: `is_valid, n_rows, n_cols, columns, dtypes, missing_per_column, n_duplicates, numeric_columns, categorical_columns, errors, warnings`

**`preprocess_data`**: `df_clean` (thang đo gốc, giữ mọi cột), `X` (chỉ đặc trưng, đã log1p nếu có, chưa scale), `feature_columns`, `preprocess_config`, `n_rows_before`, `n_rows_after`, `n_removed`, `steps_log`
- `preprocess_config` = `{feature_columns, missing_strategy, fill_values, log_transform_columns, outlier_method, feature_ranges}` – dùng để xử lý khách mới đúng như khách cũ.

**`scale_data`**: `X_scaled (n,d)`, `scaler` (đã fit, có `mean_`, `scale_`), `feature_columns`

**`run_pca`**: `pca_model` (có `.transform`), `X_pca (n,k)`, `n_components`, `explained_variance_ratio`, `cumulative_variance`, `components (k,d)`

**`run_tsne`**: `embedding (n, 2|3)`, `n_components`, `perplexity`, `learning_rate`, `n_iter`, `kl_divergence` (None nếu không có), `tsne_model` (None nếu không có transform), `backend` (`"sklearn"|"opentsne"`)

**`run_clustering` / `run_kmeans` / `run_gmm` / `run_dbscan`**:
`algorithm, model, labels, n_clusters (không tính noise), n_noise, cluster_sizes ({nhãn: số khách}), cluster_centers (None với DBSCAN), probabilities (chỉ GMM), params, feature_space ("scaled"|"pca"), supports_predict, runtime_seconds, extra ({inertia, bic, aic, core_sample_indices} – N/A thì None)`

**`evaluate_clustering`**: `silhouette_score, davies_bouldin_score, ari_score, n_clusters, n_noise, noise_ratio, runtime_seconds, note`

**`compare_models`** (DataFrame, mỗi thuật toán một dòng): cột `algorithm, n_clusters, n_noise, silhouette_score, davies_bouldin_score, ari_score, runtime_seconds`

**`get_explained_variance`** (DataFrame): cột `component, explained_variance_ratio, cumulative_variance`

**`calculate_cluster_profiles`**: `profile_table` (index = cluster_id; cột `size, ratio` + trung bình từng đặc trưng), `z_scores`, `global_mean`, `cluster_ids`

**`generate_cluster_description`**: `cluster_id, name, description, top_features ([{feature, direction: "high"|"low", z_score}])`

**`preprocess_new_customer`**: `X_new_raw, X_new_scaled (1,d), X_new_pca (1,k)|None, out_of_range_features, warnings`

**`predict_customer_cluster`**: `cluster, is_noise, method_used ("model_predict"|"knn"), confidence, probabilities, neighbor_indices, neighbor_distances, mean_neighbor_distance, is_outlier, warnings`

**`transform_new_customer`**: `coordinates (1, n_components), n_components, method_used ("knn"|"opentsne"), neighbor_indices, warnings`

## 5. Quy tắc chỉ số đánh giá

- Silhouette và Davies-Bouldin **loại điểm noise** trước khi tính.
- Cần ít nhất 2 cụm (không tính noise): không đủ thì trả `None` (dùng `is_computable`).
- ARI trả `None` khi `labels_true is None`. Dữ liệu thật không có nhãn thì ARI luôn là `None`.
- Giao diện hiển thị `None` là "N/A".

## 6. Luồng dữ liệu

```
CSV → load_dataset → validate_dataset → preprocess_data → scale_data ──┐
                                                                        │ X_scaled
                              run_pca(X_scaled) → X_pca ────────────────┤
                                                                        ▼
      run_tsne(X_pca hoặc X_scaled) → embedding   run_clustering(X_scaled hoặc X_pca) → labels
              (chỉ để vẽ bản đồ)                           │
                                                           ├→ evaluate / compare_models
                                                           └→ calculate_cluster_profiles → generate_cluster_description

Khách mới:  preprocess_new_customer → predict_customer_cluster   (gán cụm)
                                    → transform_new_customer     (vị trí trên bản đồ, chỉ để vẽ)
```

Hai việc cuối là **khác nhau**: cụm được quyết định ở không gian phân cụm; vị trí t-SNE chỉ để hiển thị.

## 7. Điểm dễ gây lỗi khi tích hợp

1. **Lệch dòng**: lọc/lấy mẫu dòng sau `preprocess_data` làm nhãn cụm không khớp khách hàng.
2. **Fit lại scaler/PCA cho khách mới**: phải dùng đối tượng đã fit, chỉ `transform`.
3. **Quên áp dụng log1p cho khách mới**: dùng `preprocess_config["log_transform_columns"]`.
4. **Tính chỉ số khi chỉ có 1 cụm hoặc toàn noise**: phải trả `None`, không để sklearn ném lỗi.
5. **`n_clusters` đếm cả noise**: không được; noise không phải cụm.
6. **`x_new` sai không gian**: `predict_customer_cluster` dùng không gian của `feature_space` (scaled/pca); `transform_new_customer` dùng không gian đã đưa vào t-SNE.
7. **Đổi tên key** hoặc trả `list` thay vì `np.ndarray`: giao diện sẽ báo "không đúng hợp đồng".
8. **sklearn mới đổi tên tham số**: t-SNE dùng `max_iter` thay `n_iter`; hợp đồng giữ tên `n_iter`, tự ánh xạ trong thân hàm.
9. **Tên trang trùng tên file**: `url_path` trong `app.py` không được trùng tên file ở `pages/` (đã xử lý, đừng đổi).

## 8. Cấu trúc project

```
Customer_Segment/
├── app.py                    điều hướng + sidebar (không sửa)
├── check_backend.py          tự kiểm tra hàm của bạn
├── requirements.txt
├── .streamlit/config.toml    theme tối giản
├── docs/FUNCTION_CONTRACT.md tài liệu này
├── models/                   BACKEND – các bạn viết thân hàm
│   ├── data_processing.py        A
│   ├── dimensionality_reduction.py  B
│   ├── clustering.py             C
│   ├── evaluation.py             D
│   ├── customer_profiling.py     E
│   └── prediction.py             F
├── pages/                    FRONTEND – mỗi file có hàm render()
│   ├── overview.py  data_exploration.py  dimensionality_reduction.py
│   ├── clustering.py  model_evaluation.py  customer_profiling.py  new_customer.py
├── utils/
│   ├── constants.py          hằng số, khóa session, CONTRACT_KEYS
│   ├── contract.py           kiểm tra output theo hợp đồng
│   ├── interface.py          lớp trung gian: gọi models/, mock khi chưa có, session state
│   ├── mock_backend.py       hàm giả để chạy thử giao diện
│   └── visualization.py      hàm vẽ Plotly
└── assets/
```
