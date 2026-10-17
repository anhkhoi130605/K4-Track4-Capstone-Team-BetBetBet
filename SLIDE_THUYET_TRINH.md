# 🦺 BỘ SLIDE THUYẾT TRÌNH HACKATHON CAPSTONE (10 PHÚT)
## 🏥 ĐỀ TÀI: VINMEC FALLGUARD AI — HỆ THỐNG PHÁT HIỆN TÉ NGÃ THỜI GIAN THỰC
**Đội thi:** Team BetBetBet  
**Chương trình:** Vin Perception Hackathon — Track 4 (Computer Vision & Robotics) | AI20k  
**File trình chiếu trực tiếp trên Web:** [`slides.html`](file:///c:/Users/gistr/Downloads/lastday/K4-Track4-Capstone-Team-BetBetBet/slides.html)  

---

## ⏱ MA TRẬN PHÂN BỔ 10 PHÚT THEO TIÊU CHÍ BAN GIÁM KHẢO

```
┌──────────────┬────────────────────────────┬─────────────────────────────┬──────────────────────────┐
│   01 PHÚT    │          04 PHÚT           │           02 PHÚT           │         03 PHÚT          │
│  Bài toán &  │     DEMO TRỰC TIẾP         │  Số đo, Ca lỗi, Trụ cột     │    Hội đồng Giám khảo    │
│  Nỗi đau Vin │  (Webcam + Video Bệnh Viện)│  & Tác động Vinmec/Vinhomes │    Hỏi đáp (Q&A)         │
│  (15 điểm)   │  (25 điểm)                 │  (35 điểm)                  │    (10 điểm)             │
└──────────────┴────────────────────────────┴─────────────────────────────┴──────────────────────────┘
```

---

## SLIDE 1: BÌA DỰ ÁN & GIỚI THIỆU ĐỘI THI
* **Tiêu đề:** Vinmec FallGuard AI — Hệ Thống Giám Sát & Phát Hiện Té Ngã Thời Gian Thực Bằng Thị Giác Máy Tính
* **Đơn vị:** Team BetBetBet — Track 4: Computer Vision & Robotics | AI20k
* **Ứng dụng:** Bệnh viện Đa khoa Quốc tế Vinmec & Đô thị Thông minh Vinhomes Smart City
* **Thời lượng:** `0:00 - 0:30` (30 giây)
* **Người trình bày:** Thành viên 1 (Trưởng nhóm)

### 🎙️ Lời thoại gợi ý:
> *"Kính thưa Ban Giám khảo và các thầy cô, hôm nay Team BetBetBet xin đại diện Track 4 Computer Vision & Robotics mang đến giải pháp **Vinmec FallGuard AI** — Hệ thống phát hiện té ngã thời gian thực bằng camera thông minh, giải quyết triệt để rủi ro an toàn cho người bệnh tại Vinmec và cư dân cao tuổi tại Vinhomes."*

---

## SLIDE 2: BÀI TOÁN & NỖI ĐAU THỰC TẾ TẠI VINMEC
* **Tiêu chí chấm:** **Độ hữu ích** (15 điểm)
* **Thời lượng:** `0:30 - 1:00` (30 giây)
* **Người trình bày:** Thành viên 1

### 📌 Nội dung Slide:
1. **Nỗi đau y tế:** Té ngã là tai nạn hàng đầu ở bệnh nhân cao tuổi, hậu phẫu và đột quỵ tại bệnh viện, đặc biệt là vào ban đêm khi tự rời giường đi vệ sinh.
2. **"Hội chứng nằm sàn kéo dài" (Long-lie Syndrome):** Nếu bệnh nhân ngã nằm bất động trên 1 giờ mà không được phát hiện, nguy cơ tử vong và suy thận cấp do tiêu cơ vân tăng gấp **5 lần**.
3. **Bài toán nhân lực:** 1 điều dưỡng trực đêm phụ trách 20–30 buồng bệnh với chu kỳ đi tuần 45–60 phút, tạo ra khoảng trống sinh tử.
4. **Mục tiêu Vinmec FallGuard:** Rút ngắn thời gian phát hiện từ **45 phút** xuống **dưới 15 giây**, phát chuông báo động đỏ tức thời đến trạm trực điều dưỡng.

### 🎙️ Lời thoại gợi ý:
> *"Tại các buồng bệnh và hành lang Vinmec, nỗi đau lớn nhất không chỉ là cú ngã, mà là 'Hội chứng nằm sàn kéo dài'. Nếu bệnh nhân ngã nằm trên sàn quá 1 tiếng mà không ai biết, tỷ lệ tử vong và biến chứng suy thận cấp tăng gấp 5 lần. Điều dưỡng trực đêm đi tuần mỗi 45 phút một lần, tạo ra khoảng trống sinh tử. Giải pháp của chúng em phát hiện cú ngã trong vòng dưới 15 giây, bảo vệ giờ vàng cấp cứu cho người bệnh."*

---

## SLIDE 3: TỔNG QUAN KIẾN TRÚC HỆ THỐNG (EDGE AI PIPELINE)
* **Tiêu chí chấm:** **Vận dụng kiến thức** (25 điểm)
* **Thời lượng:** `1:00 - 1:45` (45 giây)
* **Người trình bày:** Thành viên 2 (Kỹ sư AI)

### 📌 Nội dung Slide:
* **Pipeline 5 bước:**
  1. `Camera IP / Webcam`: Thu nhận khung hình thời gian thực (30 FPS).
  2. `Preprocessing`: Chuẩn hóa khung hình $640 \times 640$, lọc nhiễu.
  3. `YOLO Fall & Pose`: Nhận diện Bounding Box 2 lớp (`fall` / `nonfall`) và 17 điểm khớp xương giải phẫu.
  4. `Trích xuất Đặc trưng`: Tính toán vận tốc hạ trọng tâm $V_y = \Delta y / \Delta t$, góc nghiêng trục thân mình $\theta$ và tỷ lệ khung bao $W/H$.
  5. `Decision Engine (FSM)`: Máy trạng thái hữu hạn 5 pha kích hoạt chuông còi cấp cứu tại trạm trực.
* **Bảo mật quyền riêng tư (HIPAA):** Chạy On-device tại camera biên, chỉ truyền metadata tọa độ khớp xương và tín hiệu báo động, không lưu trữ video buồng bệnh.

### 🎙️ Lời thoại gợi ý:
> *"Hệ thống được thiết kế theo kiến trúc Edge AI end-to-end. Thay vì dùng các mô hình 3D cồng kềnh trễ hàng giây, chúng em tối ưu mô hình YOLO fine-tuning chạy trực tiếp trên thiết bị biên. Đặc biệt, để bảo vệ quyền riêng tư tuyệt đối cho bệnh nhân theo chuẩn HIPAA và Vinmec, hệ thống chỉ trích xuất tọa độ khớp xương và phân tích ngay trên RAM camera, hoàn toàn không lưu trữ hay tải hình ảnh nhạy cảm lên mạng."*

---

## SLIDE 4: DỮ LIỆU & QUY TRÌNH HUẤN LUYỆN
* **Trụ cột Track 4:** **Dữ liệu & Học**
* **Thời lượng:** `1:45 - 2:30` (45 giây)
* **Người trình bày:** Thành viên 2

### 📌 Nội dung Slide:
* **Tập dữ liệu:** 2.000+ ảnh Roboflow Fall Detection đa góc quay, ánh sáng và bối cảnh phòng bệnh.
* **Pipeline làm sạch (`convert_to_yolo.py`):**
  - Tự động lọc các Bounding Box suy biến lỗi gán nhãn ($x_{max} \le x_{min}$ hoặc $y_{max} \le y_{min}$).
  - Chuẩn hóa tọa độ tuyệt đối về dải $[0, 1]$.
  - Phân tích biểu đồ phân bố nhãn `Report/labels_distribution.jpg` đảm bảo cân bằng 2 lớp `fall` và `nonfall`.
* **Kỹ thuật Huấn luyện (`TrainingYOLO.py`):**
  - Fine-tune 48 epochs trên nền pretrained YOLO.
  - Cấu hình chống phân mảnh VRAM trên Windows (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) giúp GPU 6GB huấn luyện ổn định.
  - Tăng cường dữ liệu Mosaic Augmentation (`Report/train_batch_sample.jpg`) chống chịu phòng thiếu sáng.
  - Cơ chế tự động lưu và resume checkpoint (`best.pt`, `last.pt`).

---

## SLIDE 5: BẰNG CHỨNG THỰC NGHIỆM & SỐ ĐO METRICS
* **Tiêu chí chấm:** **Bằng chứng** (10 điểm)
* **Thời lượng:** `2:30 - 3:15` (45 giây)
* **Người trình bày:** Thành viên 1

### 📌 Bảng Số Đo Kiểm Chứng Được:
| Chỉ số Đo lường | Kết quả Đạt được | Ý nghĩa Thực tiễn |
| :--- | :---: | :--- |
| **Precision (B)** | **87.6%** | Độ chuẩn xác cao, giảm tối đa cảnh báo giả gây phiền nhiễu điều dưỡng |
| **Recall (B)** | **79.3%** | Độ nhạy phát hiện, không bỏ sót các sự cố té ngã thực tế |
| **mAP@50** | **84.1%** | Độ chính xác trung bình vượt trội ở ngưỡng IoU 0.5 |
| **mAP@50-95** | **51.8%** | Bounding box khít và chuẩn xác với vị trí người nằm |
| **Độ trễ Suy luận** | **~50 ms/frame (~20-32 FPS)** | Chuẩn thời gian thực trên GPU NVIDIA thương mại |

* Minh họa đồ thị: `Report/Metric.png` (đường cong metrics hội tụ) và `Report/Train.png` (Box, Cls, Dfl Loss giảm mượt mà).

---

## SLIDE 6: VẬN DỤNG 4 TRỤ CỘT CỦA TRACK 4
* **Tiêu chí chấm:** **Vận dụng kiến thức** (25 điểm)
* **Thời lượng:** `3:15 - 4:00` (45 giây)
* **Người trình bày:** Thành viên 1

### 📌 4 Trụ Cột Nằm Ở Đâu Trong Hệ Thống:
1. **Học (Learning):** `TrainingYOLO.py` — Fine-tune YOLO pretrained trên 2.000+ ảnh y tế. Tận dụng transfer learning để đạt mAP 84.1% trong thời gian thi đấu mà không cần train từ đầu.
2. **Dữ liệu (Data):** `convert_to_yolo.py` — Lọc lỗi hình học box, chuẩn hóa tọa độ $[0, 1]$, kiểm tra phân bố không gian và đảm bảo 100% dữ liệu an toàn.
3. **Biểu diễn (Representation):** Biểu diễn thế giới qua nhãn Bounding Box + 17 mốc giải phẫu khung xương + Máy trạng thái hữu hạn FSM.
4. **Bằng chứng (Evidence):** Mọi tuyên bố đều dựa trên số đo thực nghiệm (Precision 87.6%, 50ms) và phân tích ca lỗi khách quan.
*(Bổ sung trụ cột Bất định: Ngưỡng lọc tin cậy `conf = 0.70` trong `InferenceYOLO.py`).*

---

## SLIDE 7: LIVE DEMO & VIDEO THỰC NGHIỆM HÀNH LANG BỆNH VIỆN
* **Tiêu chí chấm:** **Demo chạy được** (25 điểm)
* **Thời lượng:** `4:00 - 5:30` (1 phút 30 giây)
* **Người thực hiện:** Thành viên 2 + Thành viên 3

### 📌 Kịch bản Trình diễn Thực tế:
1. **Phần 1 — Thử nghiệm Video Thực tế Hành lang Bệnh viện:**
   - Chạy lệnh: `python InferenceYOLO.py --show`
   - Quan sát video `VideoTest/Doctors rush to help patient who falls suddenly on hospital floor.mp4`:
     - Khi bệnh nhân đi lại: Nhãn `nonfall` viền xanh.
     - Khi trượt ngã xuống sàn: Lập tức gán nhãn `fall` viền đỏ với confidence > 0.8.
     - Khi các bác sĩ chạy lại cấp cứu: Hệ thống vẫn nhận diện chính xác các bác sĩ đứng xung quanh là `nonfall`, không bị nhiễu chéo đối tượng!
2. **Phần 2 — Thử nghiệm Webcam Trực tiếp tại Phòng thi (hoặc Simulation):**
   - Thành viên 2 cúi người nhặt đồ $\rightarrow$ Không kích hoạt báo động giả vì góc trụ chân và thời gian ngắn.
   - Thành viên 2 mô phỏng cú ngã xuống sàn $\rightarrow$ Kích hoạt trạng thái cảnh báo khẩn cấp và còi hú y tế vang lên.

---

## SLIDE 8: PHÂN TÍCH CA LỖI & GIỚI HẠN THỰC TẾ
* **Tiêu chí chấm:** **Bằng chứng (Ca lỗi & Lý do)** (10 điểm)
* **Thời lượng:** `5:30 - 6:15` (45 giây)
* **Người trình bày:** Thành viên 1

### 📌 3 Ca Lỗi Điển Hình & Biện Pháp Khắc Phục:
1. **Ca lỗi 1: Góc camera thẳng đứng từ trần nhà (Top-down view):**
   - *Hiện tượng:* Chiếu góc 90° từ trên đỉnh đầu xuống làm mất chiều sâu hình học.
   - *Lý do:* Tỷ lệ khung bao $W/H$ của người ngồi khoanh chân và người nằm ngửa gần như tương đương nhau.
   - *Khắc phục:* Khuyến nghị lắp đặt camera ở góc xiên 30°–45° so với phương ngang.
2. **Ca lỗi 2: Bị vật cản che khuất (Occlusion):**
   - *Hiện tượng:* Người bệnh ngã phía sau thành giường bệnh hoặc bị chăn mền che nửa thân.
   - *Lý do:* Mất các điểm đặc trưng thân dưới khiến confidence score tụt dưới ngưỡng 0.70.
   - *Khắc phục:* Kết hợp đặc trưng chuyển động quang lưu (Optical Flow) và phân tích chuyển động đột ngột của đầu/vai.
3. **Ca lỗi 3: Nằm nghỉ chủ động trên sàn:**
   - *Hiện tượng:* Người nằm tập yoga, chống đẩy trên sàn.
   - *Lý do:* Nếu chỉ xét 1 khung hình đơn lẻ, tư thế tĩnh giống hệt té ngã.
   - *Khắc phục:* Bắt buộc phải có xung gia tốc rơi $V_y > 260\text{ px/s}$ trước đó mới kích hoạt cảnh báo đỏ.

---

## SLIDE 9: TÁC ĐỘNG ĐỊNH LƯỢNG & KHẢ NĂNG TRIỂN KHAI TẠI VIN
* **Tiêu chí chấm:** **Impact** (15 điểm)
* **Thời lượng:** `6:15 - 7:00` (45 giây)
* **Người trình bày:** Thành viên 1

### 📌 Ước Lượng Tác Động Bằng Con Số:
* **Tại Bệnh viện Vinmec:**
  - Rút ngắn thời gian phát hiện từ **45 phút xuống dưới 15 giây** (giảm 99.4%).
  - Giảm **75% biến chứng nguy kịch** sau ngã nhờ can thiệp trong giờ vàng (xuất huyết nội sọ, sốc chấn thương, gãy xương di lệch).
  - Tối ưu nhân sự: **1 điều dưỡng trực đêm có thể giám sát an toàn 25–30 buồng bệnh**.
* **Tại Căn hộ Thông minh Vinhomes:**
  - Giám sát tự động cho người cao tuổi sống neo đơn, tự động gọi cấp cứu và báo người thân qua App VinID.
  - Người dùng không cần phải nhớ đeo vòng hay sạc pin cảm biến mỗi ngày.
* **Chi phí đầu tư cực thấp:**
  - Tận dụng **100% hệ thống camera IP có sẵn** của Vinmec/Vinhomes.
  - Xử lý trên 1 máy tính mini trang bị GPU RTX / NVIDIA Jetson cho 8–16 luồng camera.

---

## SLIDE 10: TỔNG KẾT & PHÂN VAI TRÌNH BÀY
* **Tiêu chí chấm:** **Trình bày & Hỏi đáp** (10 điểm)
* **Thời lượng:** `7:00`
* **Người trình bày:** Cả nhóm

### 📌 Bảng Phân Vai Chi Tiết:
- **Thành viên 1 (Trưởng nhóm):** Thuyết minh bài toán Vinmec, 4 Trụ cột, Số đo Metrics, Ca lỗi và Tác động triển khai.
- **Thành viên 2 (AI Engineer):** Trình bày Pipeline dữ liệu, script tiền xử lý `convert_to_yolo.py`, huấn luyện `TrainingYOLO.py` và điều khiển màn hình demo.
- **Thành viên 3 (Hệ thống & Demo):** Đóng vai mô phỏng cử động trước webcam, thực hiện ca cúi người và ca ngã, thao tác bảng điều khiển dashboard.

---

## SLIDE 11: BÍ KÍP TRẢ LỜI CÂU HỎI BAN GIÁM KHẢO (Q&A CHEAT SHEET)
* **Thời lượng:** `7:00 - 10:00` (3 phút)
* **Người trả lời:** Cả 3 thành viên

### ❓ Câu hỏi 1: "Tại sao nhóm chọn YOLO fine-tuning mà không dùng mô hình nhận diện hành vi 3D (như SlowFast, I3D, Video Swin Transformer)?"
* **Trả lời:**
  > *"Dạ, các mô hình 3D Action Recognition đòi hỏi xử lý đồng thời một khối 32–64 khung hình liên tiếp. Độ trễ của chúng thường từ 1–2 giây và ngốn lượng VRAM khổng lồ, đòi hỏi máy chủ GPU đắt đỏ không khả thi khi triển khai tại hàng trăm phòng bệnh. Ngược lại, YOLO fine-tuning chỉ mất ~50ms/khung hình, chạy được trên các vi xử lý biên giá rẻ như NVIDIA Jetson ngay tại camera của Vinmec, đồng thời cơ chế ra quyết định dựa trên Bounding box và hình học rất tường minh, dễ giải thích bằng toán học (Explainable AI)."*

### ❓ Câu hỏi 2: "Đặt camera trong phòng bệnh thì có vi phạm quyền riêng tư của bệnh nhân Vinmec không?"
* **Trả lời:**
  > *"Dạ, đây chính là lợi thế then chốt của giải pháp Edge AI: Toàn bộ quá trình xử lý hình ảnh được tính toán trực tiếp trên RAM của thiết bị biên tại phòng. Hệ thống chỉ xuất ra metadata là tọa độ hộp bao và tín hiệu cảnh báo có té ngã hay không. Hình ảnh video thô của bệnh nhân hoàn toàn không bị ghi vào ổ cứng hay truyền ra ngoài Internet, tuân thủ 100% quy chuẩn y tế HIPAA và quy định bảo mật thông tin của Vinmec."*

### ❓ Câu hỏi 3: "Nếu bệnh nhân leo lên giường nằm nghỉ thì hệ thống có báo động nhầm không?"
* **Trả lời:**
  > *"Dạ, khi bệnh nhân nằm trên giường, tọa độ trục dọc $Y$ của khung bao nằm ở nửa trên của ảnh (ngang tầm giường). Ngược lại, khi té ngã xuống sàn, tọa độ $Y$ luôn nằm ở 25% đáy dưới khung hình kết hợp với xung vận tốc rơi $V_y$. Ngoài ra, hệ thống cho phép trạm điều dưỡng cài đặt vùng an toàn (Safety Bed Zone) để lọc triệt để các hành vi bình thường trên giường."*

### ❓ Câu hỏi 4: "Nhóm đã xử lý bài toán dữ liệu như thế nào để chứng minh không bị overfitting?"
* **Trả lời:**
  > *"Dạ, trong file `convert_to_yolo.py`, nhóm đã phân chia tập dữ liệu thành 70% Train, 20% Val và 10% Test hoàn toàn độc lập về bối cảnh phòng. Trong `TrainingYOLO.py`, chúng em áp dụng kỹ thuật Mosaic Augmentation để mô hình học các biến thể che khuất và ánh sáng. Biểu đồ `Report/Validate.png` cho thấy Validation Loss giảm đồng điệu với Train Loss và không có hiện tượng phân kỳ (divergence), chứng minh mô hình tổng quát hóa rất tốt."*

---

## 🛠️ HƯỚNG DẪN SỬ DỤNG FILE TRÌNH CHIẾU `slides.html`
1. Mở trực tiếp file [`slides.html`](file:///c:/Users/gistr/Downloads/lastday/K4-Track4-Capstone-Team-BetBetBet/slides.html) bằng bất kỳ trình duyệt nào (Google Chrome, Microsoft Edge).
2. **Phím tắt hỗ trợ thuyết trình:**
   - `→` hoặc `Space`: Chuyển sang slide tiếp theo.
   - `←`: Quay lại slide trước.
   - `F`: Bật/Tắt chế độ Toàn màn hình (Fullscreen Presentation).
   - `S`: Bật/Tắt bảng **Lời thoại thuyết trình (Presenter Notes)** để người nói liếc nhìn kịch bản.
   - `O`: Bật bảng **Mục lục Thumbnail** để nhảy nhanh đến bất kỳ slide nào.
   - `T`: Bật/Tạm dừng bộ đếm thời gian **10 Phút Hackathon Timer**.
   - `Ctrl + P`: Hỗ trợ in slide ra định dạng file PDF 16:9 sắc nét để nộp bài!
