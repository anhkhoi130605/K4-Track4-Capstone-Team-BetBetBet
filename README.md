# 🦺 Vin Perception Hackathon — Track 4 Capstone
## 🏥 Hệ Thống Phát Hiện Té Ngã Thời Gian Thực Bằng YOLO (Real-time Fall Detection)

> **Dự án Capstone Hackathon Track 4 (Computer Vision & Robotics) — Khóa AI20k**  
> **Đội thi:** Team BetBetBet  
> **Ứng dụng:** Giám sát an toàn thông minh cho bệnh viện **Vinmec** & nhà thông minh **Vinhomes**

---

## 📌 Giới thiệu Đề tài

Té ngã là một trong những nguyên nhân hàng đầu gây chấn thương nghiêm trọng và đe dọa tính mạng ở người cao tuổi, người bệnh đang hồi sức hoặc người cần chăm sóc đặc biệt. Trong khuôn khổ **Vin Perception Hackathon**, **Team BetBetBet** xây dựng giải pháp **Phát hiện té ngã thời gian thực (Real-time Fall Detection)** ứng dụng thị giác máy tính:

- **Bài toán mục tiêu:** Phân tích luồng video từ camera an ninh để tự động phát hiện tức thời hành vi/trạng thái té ngã (`fall`) so với các tư thế sinh hoạt thông thường (`nonfall` - đứng, ngồi, đi bộ).
- **Ứng dụng thực tế trong hệ sinh thái Vin:**
  - **Vinmec:** Tích hợp camera giám sát buồng bệnh nhân nội trú, hành lang bệnh viện để báo động khẩn cấp tới điều dưỡng trực khi người bệnh ngã khỏi giường hoặc trượt chân.
  - **Vinhomes / VinFast Smart Space:** Giám sát an toàn cho cư dân cao tuổi, thông báo qua ứng dụng cư dân khi phát hiện tai nạn gia đình.

---

## 🌟 Tính Năng Nổi Bật

- ⚡ **Tốc độ thời gian thực (Real-time):** Tận dụng kiến trúc YOLO hiện đại, tối ưu hóa suy luận với độ trễ thấp phù hợp cho hệ thống Edge AI và Camera giám sát liên tục.
- 🎯 **Độ chính xác cao:** Phân biệt rõ rệt giữa tư thế nằm/té ngã bất thường và các cử động sinh hoạt hàng ngày.
- 🔄 **Pipeline tự động hóa từ A - Z:**
  - Tự động chuyển đổi dữ liệu từ Pascal VOC / Roboflow CSV sang định dạng chuẩn YOLO.
  - Kiểm tra và lọc bỏ bounding box lỗi/suy biến.
  - Tối ưu hóa bộ nhớ GPU (VRAM caching & allocation) khi huấn luyện trên Windows (RTX 3050/4060 series).
  - Hỗ trợ cơ chế **Auto-Resume** khi quá trình train bị gián đoạn.

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```text
K4-Track4-Capstone-Team-BetBetBet/
├── fall-detection.v1i.tensorflow/   # Dữ liệu thô từ Roboflow (ảnh + _annotations.csv)
│   ├── train/
│   ├── valid/
│   └── test/
├── yolo_dataset/                   # Dữ liệu đã chuyển đổi theo chuẩn YOLO
│   ├── images/ (train, val, test)
│   └── labels/ (train, val, test)
├── runs/                           # Kết quả huấn luyện (weights, logs, tensorboard)
│   └── detect/
│       ├── fall_detection_exp/
│       └── fall_detection_exp-2/
│           ├── weights/            # best.pt, last.pt
│           ├── results.csv         # Bảng thông số loss, precision, recall, mAP
│           └── labels.jpg
├── convert_to_yolo.py              # Script tiền xử lý & convert annotation sang YOLO format
├── TrainingYOLO.py                  # Script huấn luyện mô hình YOLO
├── InferenceYOLO.py                 # Script chạy dự đoán trên ảnh / video / webcam
├── data.yaml                       # File cấu hình dataset cho Ultralytics YOLO
├── index.html                      # Trang đề bài & tiêu chí Vin Perception Hackathon
└── README.md                       # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cài Đặt Môi Trường

### 1. Yêu cầu hệ thống
- **Hệ điều hành:** Windows 10/11 hoặc Linux
- **Python:** >= 3.8 (khuyên dùng Python 3.10 hoặc 3.11)
- **Phần cứng:** Khuyến nghị GPU NVIDIA hỗ trợ CUDA (ví dụ: RTX 3050 6GB hoặc cao hơn)

### 2. Cài đặt các thư viện phụ thuộc

```bash
# Cài đặt PyTorch với CUDA (ví dụ cho CUDA 12.1 hoặc 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt Ultralytics và các thư viện xử lý dữ liệu
pip install ultralytics pandas opencv-python matplotlib
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Chuẩn bị & Tiền xử lý dữ liệu

Script [convert_to_yolo.py](file:///d:/VinUni/DeepTrack/K4-Track4-Capstone-Team-BetBetBet/convert_to_yolo.py) sẽ đọc các tệp `_annotations.csv` trong thư mục `fall-detection.v1i.tensorflow/`, chuẩn hóa tọa độ bounding box sang dạng `(class_id, x_center, y_center, width, height)` và phân bổ vào thư mục `yolo_dataset/`, đồng thời tự động cập nhật [data.yaml](file:///d:/VinUni/DeepTrack/K4-Track4-Capstone-Team-BetBetBet/data.yaml):

```bash
python convert_to_yolo.py
```

Các nhãn đối tượng bao gồm:
- `0: fall` — Người đang trong trạng thái té ngã
- `1: nonfall` — Người ở trạng thái bình thường (đứng, ngồi, đi lại)

---

### Bước 2: Huấn luyện mô hình (Training)

Chạy script [TrainingYOLO.py](file:///d:/VinUni/DeepTrack/K4-Track4-Capstone-Team-BetBetBet/TrainingYOLO.py). Script đã được cấu hình tối ưu chống tràn VRAM trên Windows (`PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`):

```bash
# Huấn luyện mô hình mới với các tham số mặc định
python TrainingYOLO.py --model yolo26m.pt --epochs 50 --batch 8 --imgsz 640 --device 0

# Đối với máy GPU 6GB VRAM (như RTX 3050 Laptop):
python TrainingYOLO.py --model yolo26m.pt --epochs 50 --batch 4 --imgsz 640 --device 0
```

**Tiếp tục huấn luyện khi bị ngắt (Resume):**
```bash
# Tự động tìm checkpoint last.pt trong runs/detect/fall_detection_exp/weights/
python TrainingYOLO.py --resume

# Hoặc chỉ định cụ thể file checkpoint:
python TrainingYOLO.py --resume runs/detect/fall_detection_exp-2/weights/last.pt
```

---

### Bước 3: Đánh giá & Suy luận (Inference)

Sau khi huấn luyện, trọng số tốt nhất được lưu tại `runs/detect/.../weights/best.pt`.

#### Cách 1: Chạy bằng mã nguồn Python ([InferenceYOLO.py](file:///d:/VinUni/DeepTrack/K4-Track4-Capstone-Team-BetBetBet/InferenceYOLO.py))
```bash
# Chạy dự đoán trực tiếp trên video test Doctors Rush (mặc định lưu kết quả vào runs/detect/predict_doctors_rush)
python InferenceYOLO.py

# Hoặc chỉ định video, ngưỡng tin cậy và hiển thị GUI
python InferenceYOLO.py --source "VideoTest/Doctors rush to help patient who falls suddenly on hospital floor.mp4" --conf 0.4 --show --save

# Chạy trên webcam thời gian thực
python InferenceYOLO.py --source 0 --show
```

#### Cách 2: Chạy trực tiếp qua Ultralytics CLI
```bash
# Suy luận trên video test Doctors Rush
yolo detect predict model=runs/detect/fall_detection_exp-2/weights/best.pt source="VideoTest/Doctors rush to help patient who falls suddenly on hospital floor.mp4" save=True conf=0.4

# Suy luận trên webcam thời gian thực
yolo detect predict model=runs/detect/fall_detection_exp-2/weights/best.pt source=0 show=True conf=0.5
```

---

## 📊 Kết Quả Huấn Luyện (Experimental Results)

Dưới đây là kết quả thực nghiệm thu được từ quá trình huấn luyện:

| Chỉ số (Metric) | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| **Precision (B)** | **~81.0%** | Độ chính xác của các cảnh báo té ngã (hạn chế báo động giả) |
| **Recall (B)** | **~72.2%** | Tỷ lệ phát hiện được các tình huống ngã thực tế (tránh bỏ sót) |
| **mAP@50** | **~78.4%** | Độ chính xác trung bình tại ngưỡng IoU 0.5 |
| **mAP@50-95** | **~44.9%** | Độ chính xác tổng thể qua dải IoU từ 0.5 đến 0.95 |

> **Phân tích ca lỗi & Thách thức thực tế (Failure Cases):**
> 1. **Góc quay camera:** Khi người bị che khuất bởi giường bệnh, bàn ghế hoặc góc quay từ trên cao (top-down view) quá nghiêng.
> 2. **Tư thế gây nhiễu:** Tư thế cúi nhặt đồ hoặc nằm nghỉ ngơi chủ động trên giường đôi khi có thể gây nhầm lẫn nếu không kết hợp kiểm tra thời gian nằm bất động (temporal tracking).
> 3. **Hướng cải tiến:** Bổ sung mô hình Pose Estimation (YOLO-Pose) để theo dõi các điểm khớp xương (Keypoints) và góc nghiêng của trục cột sống theo thời gian.

---

## 👥 Thành Viên Đội Thi (Team BetBetBet)

- **Đơn vị:** Track 4 (Computer Vision & Robotics) — Vin Perception Hackathon
- **Repository:** [anhkhoi130605/K4-Track4-Capstone-Team-BetBetBet](https://github.com/anhkhoi130605/K4-Track4-Capstone-Team-BetBetBet)
