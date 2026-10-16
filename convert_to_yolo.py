import os
import shutil
import pandas as pd

# 1. Cấu hình
SOURCE_DIR = 'fall-detection.v1i.tensorflow'
OUTPUT_DIR = 'yolo_dataset'

CLASS_MAPPING = {
    'fall': 0,
    'nonfall': 1
}

# Ánh xạ tên thư mục nguồn -> tên thư mục YOLO ('valid' của Roboflow -> 'val' của YOLO)
SPLIT_MAP = {
    'train': 'train',
    'valid': 'val',
    'test': 'test'
}

def convert_dataset():
    # 2. Tạo cấu trúc thư mục YOLO
    for yolo_split in set(SPLIT_MAP.values()):
        os.makedirs(f"{OUTPUT_DIR}/images/{yolo_split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{yolo_split}", exist_ok=True)

    # 3. Duyệt qua từng tập train, valid, test
    for src_split, yolo_split in SPLIT_MAP.items():
        src_split_dir = os.path.join(SOURCE_DIR, src_split)
        csv_file = os.path.join(src_split_dir, '_annotations.csv')

        if not os.path.exists(csv_file):
            print(f"Skipping {src_split}: {csv_file} not found")
            continue

        print(f"Processing: {src_split} -> {yolo_split}...")
        df = pd.read_csv(csv_file)
        grouped = df.groupby('filename')

        for filename, group in grouped:
            src_img_path = os.path.join(src_split_dir, filename)
            dest_img_path = os.path.join(OUTPUT_DIR, 'images', yolo_split, filename)

            # Copy ảnh
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dest_img_path)
            else:
                continue

            # Tạo file label .txt
            label_name = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(OUTPUT_DIR, 'labels', yolo_split, label_name)

            lines = []
            for _, row in group.iterrows():
                cls_name = str(row['class']).strip()
                if cls_name not in CLASS_MAPPING:
                    continue

                cls_id = CLASS_MAPPING[cls_name]
                img_w, img_h = row['width'], row['height']
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

                # Lọc bỏ box lỗi suy biến
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Công thức chuẩn hóa YOLO
                x_center = ((xmin + xmax) / 2.0) / img_w
                y_center = ((ymin + ymax) / 2.0) / img_h
                bbox_w = (xmax - xmin) / img_w
                bbox_h = (ymax - ymin) / img_h

                # Đảm bảo giá trị nằm trong [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                bbox_w = max(0.0, min(1.0, bbox_w))
                bbox_h = max(0.0, min(1.0, bbox_h))

                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    # 4. Sinh file data.yaml
    dataset_path = os.path.abspath(OUTPUT_DIR).replace('\\', '/')
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val
test: images/test

names:
  0: fall
  1: nonfall
"""
    with open("data.yaml", "w", encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"Done! YOLO dataset saved to '{OUTPUT_DIR}' and generated 'data.yaml'.")

if __name__ == '__main__':
    convert_dataset()
