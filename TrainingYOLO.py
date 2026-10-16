import argparse
import os

# Cấu hình chống phân mảnh VRAM trên Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLO phát hiện té ngã (Fall Detection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Các tham số mô hình & dữ liệu
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="Tên model (vd: yolo26m.pt, yolo26s.pt)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Đường dẫn đến file cấu hình dataset data.yaml")
    # 2. Các siêu tham số huấn luyện (Hyperparameters)
    parser.add_argument("--epochs", type=int, default=50, help="Tổng số epoch cần huấn luyện")
    parser.add_argument("--batch", type=int, default=8, help="Kích thước batch size (chọn 4 cho bản Medium trên RTX 3050 6GB)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (pixels)")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị chạy: '0' (GPU NVIDIA) hoặc 'cpu'")
    parser.add_argument("--workers", type=int, default=2, help="Số lượng worker nạp dữ liệu (tối ưu cho Windows)")
     # 3. Đường dẫn lưu checkpoint & TensorBoard
    parser.add_argument("--project", type=str, default="runs/detect",help="Thư mục cha chứa kết quả huấn luyện và log TensorBoard")
    parser.add_argument("--name", type=str, default="fall_detection_exp",help="Tên thư mục của đợt huấn luyện này")
     # 4. Tham số tiếp tục train (Resume) khi bị ngắt quãng
    parser.add_argument("--resume", nargs="?", const="default", default=None,help="Tiếp tục train từ file last.pt khi bị ngắt. Cú pháp: --resume hoặc --resume duong_dan/last.pt")
    
    return parser.parse_args()


def train(args):
    project_dir = os.path.abspath(args.project)

    # ==========================================================
    # TRƯỜNG HỢP 1: TIẾP TỤC TRAIN TỪ CHECKPOINT (RESUME)
    # ==========================================================
    if args.resume:
        # Nếu chỉ gõ --resume thì tự động tìm file last.pt theo project/name
        if args.resume == "default":
            last_ckpt = os.path.join(project_dir, args.name, "weights", "last.pt")
        else:
            last_ckpt = os.path.abspath(args.resume)

        if not os.path.exists(last_ckpt):
            raise FileNotFoundError(
                f"\n[LỖI] Không tìm thấy file checkpoint để resume: '{last_ckpt}'\n"
                f"Vui lòng kiểm tra lại đường dẫn file last.pt!"
            )

        print(f"\n>>> [RESUME] Đang tiếp tục huấn luyện từ checkpoint: {last_ckpt}")
        model = YOLO(last_ckpt)
        results = model.train(resume=True)
        return results

    # ==========================================================
    # TRƯỜNG HỢP 2: BẮT ĐẦU HUẤN LUYỆN MỚI
    # ==========================================================
    print(f"\n>>> [START] Khởi tạo mô hình mới từ: {args.model}")
    print(f">>> Cấu hình: {args.epochs} epochs | Batch: {args.batch} | Image Size: {args.imgsz}")
    print(f">>> Kết quả sẽ lưu tại: {os.path.join(project_dir, args.name)}")
    
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=args.name,
        save=True,               # YOLO tự động lưu weights/best.pt (tốt nhất) và weights/last.pt (epoch gần nhất)
        save_period=1            # Cập nhật checkpoint sau mỗi 1 epoch để đảm bảo không mất dữ liệu
    )
    
    weights_dir = os.path.join(project_dir, args.name, "weights")
    print("\n>>> [HOÀN TẤT]")
    print(f"Trọng số tốt nhất (best.pt) lưu tại: {os.path.join(weights_dir, 'best.pt')}")
    print(f"Trọng số epoch cuối (last.pt) lưu tại: {os.path.join(weights_dir, 'last.pt')}")
    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)