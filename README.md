# Phân cụm khách hàng và trực quan hóa bằng t-SNE

Đồ án học không giám sát: từ bảng `n` khách hàng với vector hành vi `x ∈ ℝᵈ`, (1) gán mỗi khách vào một nhóm,
(2) vẽ bản đồ 2D để kiểm chứng nhóm, (3) mô tả mỗi nhóm bằng ngôn ngữ nghiệp vụ, (4) so sánh phân cụm trên dữ liệu
gốc, PCA, UMAP và t-SNE, (5) gán khách hàng mới vào nhóm.

## Chạy ứng dụng

```
pip install -r requirements.txt
streamlit run app.py        # hoặc run.bat trên Windows
```

Trên giao diện, bấm **"Dùng bộ mô phỏng 5 nhóm (có nhãn)"** ở trang Data Exploration, rồi đi lần lượt qua các trang:
Preprocessing & t-SNE → Clustering → Model Evaluation → Customer Profiling → New Customer.

Kiểm tra nhanh các hàm backend: `python check_backend.py`.

## Cấu trúc

| Thư mục / file | Nội dung |
|---|---|
| `models/` | Backend: tiền xử lý, PCA/UMAP/t-SNE, phân cụm, đánh giá, mô tả cụm, gán khách mới |
| `pages/`, `app.py`, `utils/` | Giao diện Streamlit và lớp trung gian |
| `New_Data/` | Dữ liệu thô, dữ liệu đã xử lý, bộ mô phỏng có nhãn (xem `README_data.md`) |
| `docs/FUNCTION_CONTRACT.md` | Hợp đồng giữa giao diện và backend |

## Điểm đáng chú ý

- **t-SNE viết tay:** `CustomTSNE` (`models/dimensionality_reduction_cpu.py`) cài đặt t-SNE bằng PyTorch, là phần
  trọng tâm của đồ án và là lựa chọn mặc định. Độ phức tạp `O(n²)` nên giao diện giới hạn mặc định 3.000 dòng.
  openTSNE chỉ là bản đối chiếu (`use_opentsne=True`).
- **t-SNE chỉ để nhìn:** phân cụm chạy trên dữ liệu chuẩn hóa hoặc PCA. Benchmark chạy thêm trên UMAP và t-SNE để
  so sánh. Khách mới được gán cụm dựa trên không gian chuẩn hóa/PCA (`predict()` hoặc kNN); vị trí của khách mới
  trên bản đồ t-SNE chỉ để quan sát.
- **Mô tả nhóm:** tên nhóm (ví dụ "Khách VIP, chi tiêu cao", "Khách mua thường xuyên, đơn nhỏ") được sinh tự động từ
  z-score của các đặc trưng, không trùng nhau giữa các cụm; có thể đổi tên thủ công ở trang Customer Profiling.
- **Mốc recency:** 01/11 của năm lớn nhất trong dữ liệu (hiện là 01/11/2025).
- **Mở rộng t-SNE:** Barnes-Hut và FIt-SNE (FFT) viết tay trong `models/tsne_scalable.py`, thí nghiệm bằng
  `python run_tsne_scaling_experiment.py`, xem `docs/TSNE_SCALING.md` và trang *t-SNE Scaling* trên giao diện.

Kiểm thử hồi quy: `python -m pytest tests -q`.
