import os
import torch
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    print("=== YOLO License Plate Training Script ===")
    
    BASE_DIR = Path(r"C:\Users\pamik\OneDrive\Desktop\ML\Toon")
    DATASET_DIR = BASE_DIR / "License-Plate-Data"
    YAML_PATH = DATASET_DIR / "data.yaml"

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ Dataset ที่: {DATASET_DIR}")

    print(f"[*] กำลังตรวจสอบโครงสร้างข้อมูลที่: {DATASET_DIR}")
    
    train_img_path = DATASET_DIR / "train" / "images"
    test_img_path = DATASET_DIR / "test" / "images"
    val_img_path = DATASET_DIR / "valid" / "images"

    train_rel_path = "train/images" if train_img_path.exists() else "test/images"
    val_rel_path = "valid/images" if val_img_path.exists() else "test/images"

    yaml_content = {
        "path": str(DATASET_DIR.resolve()).replace("\\", "/"),
        "train": train_rel_path,
        "val": val_rel_path,
        "nc": 1,
        "names": ["license_plate"]
    }

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"[✓] สร้าง/อัปเดตไฟล์ {YAML_PATH} เรียบร้อยแล้ว")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] อุปกรณ์ที่ใช้ในการเทรน: {device}")
    
    print("[*] กำลังโหลดโมเดล YOLOv8n Pre-trained...")
    model = YOLO("yolov8n.pt")

    print("\n" + "="*40)
    print("เริ่มการเทรนโมเดล (เน้นประมวลผลเร็ว)...")
    print("="*40 + "\n")

    model.train(
        data=str(YAML_PATH),
        epochs=30,
        imgsz=416,
        batch=16,
        workers=4,
        device="cpu",
        project=str(BASE_DIR / "runs"),
        name="license_plate_detector",
        exist_ok=True,
        verbose=True,
        deterministic=False,
        optimizer='SGD'
    )

    print("\n[✓] การเทรนเสร็จสมบูรณ์เรียบร้อยแล้ว!")
    print(f"[*] โมเดลน้ำหนักที่ได้จะถูกบันทึกไว้ที่: {BASE_DIR / 'runs' / 'license_plate_detector' / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()