<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông Tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>PHÁT TRIỂN VÀ VẬN HÀNH HỆ THỐNG MÁY HỌC</b></h1>
<h2 align="center"><b>Lab 1 - Xây dựng training pipeline có sử dụng experiment tracking</b></h2>

# Pipeline MLflow Dự Đoán Chất Lượng Rượu Vang

## Tổng Quan  
Dự án này triển khai một pipeline máy học đầu-cuối để dự đoán chất lượng rượu vang, sử dụng Metaflow và MLflow. Pipeline bao gồm các bước: tải dữ liệu, phân tích dữ liệu khám phá (EDA), huấn luyện mô hình với cross-validation, và so sánh các mô hình.

## Các Bước Trong Pipeline

### 1. Tải Dữ Liệu (`load_dataset`)
- Tải bộ dữ liệu chất lượng rượu vang từ kho dữ liệu của MLflow  
- Thực hiện quản lý phiên bản dữ liệu bằng dấu thời gian  
- Ghi log metadata và thống kê dữ liệu vào MLflow  
- Hỗ trợ cả nguồn dữ liệu cục bộ (local) và từ xa (tải internet)

### 2. Phân Tích Dữ Liệu Khám Phá (EDA) (`eda`)
- Sinh thống kê chi tiết về bộ dữ liệu  
- Tạo biểu đồ tương tác bằng VegaChart:
  - Phân phối chất lượng rượu
  - Mối quan hệ giữa các đặc trưng và chất lượng
  - Ma trận tương quan
- Tất cả biểu đồ được hiển thị trong Metaflow cards

### 3. Chuẩn Bị Dữ Liệu (`load_data`)
- Chia dữ liệu thành tập huấn luyện và kiểm thử  
- Áp dụng chia tập theo tỷ lệ 80-20  
- Lưu dữ liệu để sử dụng trong huấn luyện song song nhiều mô hình

### 4. Huấn Luyện Mô Hình (`train_models`)
- Huấn luyện song song nhiều mô hình [classifier](https://github.com/hyperopt/hyperopt-sklearn#classifiers) qua [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
- Tùy chọn tối ưu siêu tham số bằng Hyperopt  
- Theo dõi thời gian huấn luyện cho mỗi mô hình
- Ghi log đầy đủ các chỉ số và tham số vào MLflow
- Hiển thị kết quả chi tiết trong Metaflow cards

### 5. So Sánh Các Mô Hình (`join`)
- So sánh hiệu suất của tất cả mô hình
- Phân tích độ chính xác và thời gian huấn luyện
- Tự động chọn mô hình tốt nhất dựa trên độ chính xác
- Biểu đồ so sánh trực quan với màu sắc thể hiện thời gian huấn luyện
- Ghi log mô hình tốt nhất vào MLflow với signature và input example

## Các Tính Năng Chính
- **Tối Ưu Siêu Tham Số Tự Động**: Sử dụng Hyperopt để điều chỉnh mô hình
- **Huấn Luyện Song Song**: Tăng hiệu suất bằng huấn luyện đồng thời nhiều mô hình
- **Theo Dõi Hiệu Suất**: Đo lường và so sánh cả độ chính xác và thời gian huấn luyện
- **Biểu Đồ Tương Tác**: Cung cấp cái nhìn trực quan sinh động qua Metaflow cards
- **Pipeline Có Thể Tái Sử Dụng**: Dữ liệu và mô hình được quản lý phiên bản rõ ràng

## Công Nghệ Sử Dụng

### Metaflow
- Quản lý và điều phối workflow  
- Khả năng thực thi song song  
- Quản lý tài nguyên và mở rộng quy mô  
- Hỗ trợ hiển thị biểu đồ tương tác (cards)

### MLflow
- Theo dõi thí nghiệm  
- Quản lý mô hình  
- Quản lý phiên bản dữ liệu  
- Ghi log chỉ số và trực quan hóa

## Cài Đặt

### Thiết Lập Môi Trường
1. Tạo môi trường ảo mới:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Khởi động máy chủ MLflow: Mở một terminal riêng và chạy
```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Chạy Pipeline
Trong cùng thư mục của dự án.

### Chạy Cơ Bản
```bash
python main.py run
```

### Bật Tối Ưu Siêu Tham Số 
```bash
python main.py run --use_hyperopt true
```
Lưu ý: Thời gian chạy sẽ lâu nếu dùng nhiều model/model train lâu (~ 20 phút cho tất cả).

### Chỉ Định Đường Dẫn Dữ Liệu Tùy Chỉnh
```bash
python main.py run --data-dir /path/to/data
```

## Xem Kết Quả

### Giao Diện MLflow
```bash
mlflow ui
```
Truy cập tại: http://localhost:5000

### Giao Diện Metaflow
Trong cùng thư mục của dự án, chạy trong terminal mới:
```bash
python main.py card server
```
Truy cập tại: http://localhost:8324

## Ghi Chú
- Hãy đảm bảo máy chủ MLflow đã được khởi động trước khi chạy pipeline  
- Lần chạy đầu tiên sẽ tải bộ dữ liệu  
- Tối ưu siêu tham số có thể mất thời gian lâu hơn  
- Yêu cầu tối thiểu: 4GB RAM

