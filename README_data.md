# Dữ liệu và tiền xử lý

## Dữ liệu nguồn và kích thước

Quy trình sử dụng `Data/raw/synthetic_customers_10cols.csv`. Mỗi dòng là một hồ sơ. **`n = 100.000` hồ sơ, `d = 13` đặc trưng đầu vào và 0 dòng bị loại.**

| Bảng | Số dòng | Số cột |
|---|---:|---:|
| `synthetic_customers_10cols.csv` | 100.000 | 10 |
| `customers_clean.csv` | 100.000 | 12 |
| `customers_scaled.csv` | 100.000 | 14 |
| `X`, bảng đầu vào cho mô hình | 100.000 | 13 |

Trong bảng trên, `n` là số dòng hồ sơ và `d` là số cột đặc trưng mà mô hình nhận. File `customers_scaled.csv` có thêm cột `record_id` để tra cứu. Khi đọc file với `index_col="record_id"`, pandas đặt mã này ở index và tạo `X` có kích thước `(100000, 13)`.

Bản CSV nguồn có 0 ô thiếu và 0 dòng trùng hoàn toàn. Có 60.513 giá trị `customer_id` khác nhau trong 100.000 dòng, vì một mã khách hàng có thể xuất hiện ở nhiều hồ sơ. Quy trình giữ cả 100.000 dòng và cấp `record_id` duy nhất theo thứ tự dòng của file nguồn.
## Số cột đã loại và lý do

| Phạm vi | Số cột loại | Giải thích |
|---|---:|---|
| Bản dữ liệu 20 cột → bản nguồn 10 cột | 10 | Bản nguồn của dự án đã lược bỏ `email`, `age`, `gender`, `city`, `country`, `income`, `credit_score`, `device_type`, `churn_risk` và `marketing_opt_in`. |
| Bản nguồn 10 cột → `customers_clean.csv` | 0 | Bảng sạch giữ đủ 10 cột nguồn và thêm `record_id` cùng `recency_days`, thành 12 cột. |
| Chọn đặc trưng cho mô hình từ 10 cột nguồn | 4 | Loại `customer_id`, `name`, `first_purchase_date` và `last_purchase_date` khỏi đầu vào mô hình. Các cột này có trong bảng sạch để tra cứu. `last_purchase_date` được dùng để tính `recency_days`. |

Hai cột nguồn được **thay bằng đặc trưng đã biến đổi**. `avg_order_value`
trở thành `avg_order_value_log`, còn `favorite_category` được thay bằng
7 cột One-Hot.

Số đặc trưng đầu vào là `d = 10 − 4 + 1 + 6 = 13`. Trong đó, `+1` là
`recency_days` và `+6` là số cột tăng thêm khi thay 1 cột danh mục bằng
7 cột One-Hot. `record_id` là mã đối chiếu và nằm ngoài 13 đặc trưng.

Quy trình giữ đủ `n = 100.000` hồ sơ. **Số dòng bị loại là 0.**

## Từ điển cột của `customers_clean.csv`

Bảng sạch phục vụ tra cứu hồ sơ và diễn giải các cụm. Bảng có 10 cột nguồn cùng hai cột được tạo trong quá trình tiền xử lý là `record_id` và `recency_days`.

| Cột | Ý nghĩa |
|---|---|
| `record_id` | Mã duy nhất của từng dòng trong bộ dữ liệu đã xử lý. Dùng để nối với bảng đặc trưng và kết quả phân cụm. |
| `customer_id` | Mã khách hàng trong dữ liệu nguồn. Một mã có thể xuất hiện ở nhiều dòng. |
| `name` | Tên khách hàng ghi trong hồ sơ. |
| `first_purchase_date` | Ngày mua đầu tiên được ghi nhận, theo định dạng `YYYY-MM-DD`. |
| `last_purchase_date` | Ngày mua gần nhất được ghi nhận, theo định dạng `YYYY-MM-DD`. |
| `total_spent` | Tổng số tiền đã chi theo dữ liệu nguồn. |
| `avg_order_value` | Giá trị trung bình của một đơn hàng theo dữ liệu nguồn. |
| `loyalty_points` | Số điểm tích lũy của khách hàng. |
| `favorite_category` | Tên danh mục sản phẩm yêu thích. |
| `subscription` | Trạng thái đăng ký. `True` nghĩa là có đăng ký và `False` nghĩa là không có đăng ký. |
| `support_tickets` | Số yêu cầu hỗ trợ được ghi nhận. |
| `recency_days` | Số ngày từ lần mua cuối đến ngày tham chiếu `2025-05-05`. Giá trị trong bộ dữ liệu nằm từ 1 đến 366 ngày. |

## Từ điển cột của `customers_scaled.csv`

Bảng chứa **5 đặc trưng đã chuẩn hóa**, **8 đặc trưng mã hóa `0/1`** và **1 mã đối chiếu**. Với 7 cột danh mục, mỗi hồ sơ có đúng một cột bằng `1`.

| Cột | Ý nghĩa |
|---|---|
| `record_id` | Mã hồ sơ dùng để tra về bảng sạch. Đọc dưới dạng index khi tạo `X`. |
| `total_spent` | Tổng chi tiêu sau chuẩn hóa. |
| `loyalty_points` | Số điểm tích lũy sau chuẩn hóa. |
| `support_tickets` | Số yêu cầu hỗ trợ sau chuẩn hóa. |
| `subscription` | `1` nếu có đăng ký và `0` nếu không có đăng ký. |
| `recency_days` | Số ngày kể từ lần mua cuối sau chuẩn hóa. |
| `avg_order_value_log` | Giá trị trung bình của một đơn hàng sau khi áp dụng `log1p` rồi chuẩn hóa. |
| `favorite_category_Beauty` | `1` nếu danh mục yêu thích là sản phẩm làm đẹp và `0` nếu là danh mục khác. |
| `favorite_category_Books` | `1` nếu danh mục yêu thích là sách và `0` nếu là danh mục khác. |
| `favorite_category_Clothing` | `1` nếu danh mục yêu thích là quần áo và `0` nếu là danh mục khác. |
| `favorite_category_Electronics` | `1` nếu danh mục yêu thích là đồ điện tử và `0` nếu là danh mục khác. |
| `favorite_category_Home & Kitchen` | `1` nếu danh mục yêu thích là đồ gia dụng và nhà bếp, `0` nếu là danh mục khác. |
| `favorite_category_Sports` | `1` nếu danh mục yêu thích là đồ thể thao và `0` nếu là danh mục khác. |
| `favorite_category_Toys` | `1` nếu danh mục yêu thích là đồ chơi và `0` nếu là danh mục khác. |

Ví dụ, hồ sơ `record_id = 1` trong bảng sạch có `total_spent = 3265.97`, `recency_days = 190` và `favorite_category = Toys`. Trong bảng chuẩn hóa, cùng hồ sơ đó có `total_spent ≈ -0.611769`, `recency_days ≈ 0.065625` và `favorite_category_Toys = 1`.

## Quy trình tiền xử lý

1. Cấp `record_id`, chuyển hai cột ngày sang kiểu ngày tháng và tính `recency_days` bằng ngày tham chiếu trừ ngày mua cuối. `2025-05-05` là mốc chọn cho đồ án, sau ngày mua muộn nhất trong dữ liệu một ngày.
2. Áp dụng `log1p(x) = ln(1 + x)` cho `avg_order_value` để giảm độ lệch phải.
3. Mã hóa `subscription` thành `0/1` và dùng One-Hot Encoding cho 7 danh mục, theo danh sách danh mục đã sắp xếp.
4. Dùng `StandardScaler` cho `total_spent`, `avg_order_value_log`, `loyalty_points`, `support_tickets` và `recency_days`. Bộ chuẩn hóa được fit trên 100.000 hồ sơ và biến đổi từng cột theo công thức `z = (x - μ) / σ`.

## Sử dụng file bàn giao

Chạy đoạn sau từ thư mục gốc dự án để đọc hai bảng và kiểm tra mã hồ sơ theo từng dòng.

```python
import pandas as pd

ho_so = pd.read_csv("Data/processed/customers_clean.csv", index_col="record_id")
X = pd.read_csv("Data/processed/customers_scaled.csv", index_col="record_id")

assert X.shape == (100000, 13)
assert X.index.is_unique and X.index.equals(ho_so.index)
```

Đưa `X` vào mô hình. Sau khi gán nhãn cụm, dùng `record_id` để tra lại số tiền, số ngày và danh mục trong `ho_so`.

`Data/processed/scaler.pkl` lưu bộ chuẩn hóa đã fit dưới khóa `scaler` và danh sách 7 danh mục theo thứ tự mã hóa dưới khóa `danh_muc_chuan`. Đọc file bằng `joblib.load()` khi cần áp dụng cùng cách mã hóa và chuẩn hóa cho hồ sơ mới.

Để tạo lại ba file bàn giao, chạy các cell xử lý dữ liệu nguồn và xuất file trong `Notebooks/01_data_understanding.ipynb`. Notebook gọi các hàm trong `preprocess_customers.py`.

## Giới hạn dữ liệu và bộ mô phỏng

CSV nguồn chỉ có ngày mua đầu và ngày mua cuối. Hai mốc này cho phép tính Recency. Để tính Frequency theo tháng, cần ngày của từng đơn hàng. `total_spent` là tổng chi tiêu được ghi nhận trong nguồn.

`Data/simulated/` chứa bộ mô phỏng riêng gồm 500 hồ sơ thuộc 5 nhóm thiết kế. File metadata đi kèm có số đơn mô phỏng và nhãn nhóm để đối chiếu khi đánh giá phân cụm. Có thể tái tạo bộ mô phỏng bằng `py generate_simulated_customers.py` từ thư mục gốc dự án.