import argparse
import os
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chạy dự đoán phát hiện té ngã (Fall Detection Inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/fall_detection_exp-2/weights/best.pt",
        help="Đường dẫn đến file weights (.pt) đã huấn luyện"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="VideoTest/Doctors rush to help patient who falls suddenly on hospital floor.mp4",
        help="Nguồn dữ liệu đầu vào: đường dẫn video/ảnh, '0' cho webcam"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.7,
        help="Ngưỡng tin cậy (Confidence threshold) [0.0 - 1.0]"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Thiết bị chạy: '0' (GPU NVIDIA) hoặc 'cpu'"
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hiển thị kết quả trên cửa sổ GUI (--show hoặc --no-show)"
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lưu kết quả dự đoán có gắn bounding box vào thư mục output"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Thư mục cha chứa kết quả dự đoán"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict_doctors_rush",
        help="Tên thư mục lưu kết quả dự đoán"
    )
    return parser.parse_args()


def run_inference(args):
    # Kiểm tra weights
    if not os.path.exists(args.model):
        alt_model = "yolo26m.pt"
        if os.path.exists(alt_model):
            print(f"[CẢNH BÁO] Không tìm thấy '{args.model}', chuyển sang dùng model gốc: '{alt_model}'")
            args.model = alt_model
        else:
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {args.model}")

    print(f"\n>>> [INFERENCE] Nạp mô hình từ: {args.model}")
    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not os.path.exists(source) and not source.startswith(("http://", "https://", "rtsp://")):
        print(f"[CẢNH BÁO] File nguồn '{source}' không tồn tại!")

    print(f">>> Bắt đầu dự đoán trên source: {args.source}")
    print(f">>> Cấu hình: Conf={args.conf} | Device={args.device} | Save={args.save} | Show={args.show}")

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=args.save,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    save_dir = os.path.join(args.project, args.name)
    print("\n>>> [HOÀN TẤT] Dự đoán kết thúc thành công!")
    if args.save:
        print(f">>> Video/ảnh kết quả đã được lưu tại: {os.path.abspath(save_dir)}")
    return results


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
