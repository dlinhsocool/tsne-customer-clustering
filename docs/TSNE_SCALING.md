# Mở rộng t-SNE: Barnes-Hut và FIt-SNE (viết tay)

Mã: `models/tsne_scalable.py`. Thí nghiệm: `python run_tsne_scaling_experiment.py` (kết quả ở `results/tsne_scaling/`).
Giao diện: trang **t-SNE Scaling (BH & FFT)**. Test: `python -m pytest tests/test_tsne_scalable.py -q`.

## 1. Vấn đề
Gradient KL: ∂KL/∂yᵢ = 4 [ Σⱼ pᵢⱼ q̃ᵢⱼ (yᵢ−yⱼ) − (1/Z) Σⱼ q̃ᵢⱼ² (yᵢ−yⱼ) ], với q̃ᵢⱼ = 1/(1+‖yᵢ−yⱼ‖²), Z = Σₖ≠ₗ q̃ₖₗ.
- Lực hút: P thưa (3·perplexity láng giềng) → O(n·k).
- Lực đẩy: n² cặp → đây là nút thắt; cả hai phương pháp chỉ thay phần này.

## 2. Barnes-Hut (O(n log n))
Xây quadtree; với điểm i, nếu ô có độ rộng r và khoảng cách tới khối tâm d thỏa r/d < θ thì thay cả ô bằng khối tâm
(N điểm → 1 tương tác). θ = 0 trùng exact; θ lớn thì nhanh hơn nhưng sai hơn (mặc định 0,5).

## 3. FIt-SNE (O(n))
Các tổng Σⱼ K(yᵢ,yⱼ)·chargeⱼ với K = q̃ và q̃² là tích chập. Nội suy Lagrange bậc p lên lưới đều → tích chập
bằng FFT (nhúng tuần hoàn 2G) → nội suy ngược về các điểm. Dùng 4 kênh: {1, yₓ, y_y} với q̃² (lực đẩy) và {1} với q̃ (Z),
vì Σⱼ q̃²ᵢⱼ(yᵢ−yⱼ) = yᵢ·φ₁(i) − φ_y(i).

## 4. Thiết kế thí nghiệm công bằng
Cùng P, cùng khởi tạo PCA, cùng bộ tối ưu (momentum + gains + early exaggeration 12 × 250 vòng), cùng số vòng.
Chỉ khác cách tính lực đẩy. Đối chiếu thêm openTSNE (bh/fft). Dữ liệu: 13 đặc trưng chuẩn hóa của bộ mô phỏng 100k.

## 5. Kết quả cần trình bày (xem CSV/PNG)
- `repulsion_scaling.png`: thời gian một lần tính lực đẩy theo n (log-log), độ dốc ≈ 2 / ≈ 1 / ≈ 1.
- `gradient_error.png`: sai số gradient tương đối theo θ và theo số điểm nội suy p.
- `benchmark.png/csv`: thời gian, KL, Trustworthiness, Continuity, kNN preservation, ARI của K-Means trên bản đồ.
- `maps.png`, `map_full.png`: bản đồ; FIt-SNE chạy trọn 100.000 khách.

## 6. Lưu ý trung thực
- Ở n nhỏ (≲ 1–2 nghìn) FIt-SNE không nhanh hơn do chi phí cố định của lưới/FFT; lợi thế xuất hiện khi n lớn.
- Cài đặt Python/Numba trên 2 nhân CPU, không tối ưu bằng C++ của FIt-SNE gốc; so sánh thời gian với openTSNE chỉ mang tính tham khảo.
- Chỉ hỗ trợ nhúng 2 chiều. `CustomTSNE` (PyTorch, exact) vẫn là bản mặc định của đồ án.
