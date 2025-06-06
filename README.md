<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông Tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>PHÁT TRIỂN VÀ VẬN HÀNH HỆ THỐNG MÁY HỌC</b></h1>

# Pipeline MLflow Dự Đoán Chất Lượng Rượu Vang

- Link video demo main flow xem ở [đây](https://drive.google.com/file/d/1fqHUXSFL31XyWmoZeUkEIUw4oRfo9X3y/view)
- Link video demo docker-compose xem ở [đây](https://drive.google.com/file/d/109bz4EcAOxRBOUyqWTE7Xr9S-T_qwIRp/view)

## Tổng Quan  
Dự án này triển khai một pipeline máy học đầu-cuối để dự đoán chất lượng rượu vang ([nguồn dataset tham khảo](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)), sử dụng Metaflow và MLflow. Pipeline bao gồm các bước: tải dữ liệu, phân tích dữ liệu khám phá (EDA), huấn luyện mô hình, tìm tham số tối ưu, và so sánh các mô hình.

## Các Bước Trong Pipeline

### 1. Tải Dữ Liệu (`load_dataset`)
- Tải bộ dữ liệu chất lượng rượu vang từ kho dữ liệu của MLflow (xem tại [đây](https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv))
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
- Huấn luyện song song nhiều mô hình [classifier](https://github.com/hyperopt/hyperopt-sklearn#classifiers)
- Tùy chọn tối ưu siêu tham số bằng [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
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
- **Tối Ưu Siêu Tham Số Tự Động**: Sử dụng hyperopt-sklearn để điều chỉnh mô hình
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

### Hyperopt-sklearn
- Tự động lựa chọn mô hình và tối ưu siêu tham số
- Tìm kiếm không gian tham số bằng thuật toán Bayesian (TPE), Random Search, Annealing, ...
- Hỗ trợ nhiều thuật toán học máy trong scikit-learn
- Tích hợp dễ dàng với pipeline huấn luyện và theo dõi thí nghiệm

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

## Chạy Bằng Docker Compose

### 1. Build và khởi động toàn bộ hệ thống

Trong thư mục dự án, chạy lệnh sau để build và khởi động các service (MLflow server, train pipeline, model serving):

```bash
docker-compose up --build
```

- Service `mlflow-server`: Chạy MLflow Tracking Server tại [http://localhost:5000](http://localhost:5000)
- Service `train-pipeline`: Tự động huấn luyện và đăng ký mô hình tốt nhất vào MLflow Model Registry
- Service `model-serving`: Tự động phục vụ mô hình tốt nhất qua REST API tại [http://localhost:5050/invocations](http://localhost:5050/invocations)

### 2. Gửi yêu cầu dự đoán (inference)

Sau khi các container đã chạy xong, bạn có thể gửi yêu cầu dự đoán tới API phục vụ mô hình:

```bash
curl -d '{"dataframe_split": {
"columns": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
"data": [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]}}' \
-H 'Content-Type: application/json' -X POST localhost:5050/invocations
```

Kết quả server trả về sẽ có dạng tương ứng với một trong các nhãn dự đoán:

```json
{"predictions": [3]}
```

### 3. Dừng toàn bộ hệ thống

Nhấn `Ctrl+C` trong terminal đang chạy hoặc dùng lệnh:

```bash
docker-compose down
```

### Lưu ý

- Đảm bảo Docker và Docker Compose đã được cài đặt trên máy.
- Lần chạy đầu tiên có thể mất thời gian để build image và tải dữ liệu.
- Có thể kiểm tra logs của từng service bằng lệnh:
  ```bash
  docker-compose logs <service-name>
  ```
  Ví dụ: `docker-compose logs train-pipeline`

---

## Ghi Chú
- Hãy đảm bảo máy chủ MLflow đã được khởi động trước khi chạy pipeline  
- Lần chạy đầu tiên sẽ tải bộ dữ liệu  
- Tối ưu siêu tham số có thể mất thời gian lâu hơn  
- Yêu cầu tối thiểu: 4GB RAM

