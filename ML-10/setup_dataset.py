import os
from ultralytics import YOLO

def train_model(epochs=50, imgsz=640, batch=16, project_name="runs/detect", run_name="exp"):
    data_path = "dataset/data.yaml"
    if not os.path.exists(data_path):
        print(f"[Error] ไม่พบไฟล์คอนฟิกสำหรับ dataset ที่: {os.path.abspath(data_path)}")
        print("กรุณาตรวจสอบว่ามีโฟลเดอร์ dataset และไฟล์ data.yaml อยู่ในโปรเจกต์แล้วหรือยัง")
        return None

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_name,
        name=run_name,
        exist_ok=True
    )
    
    print(f"การเทรนเสร็จสิ้นสำหรับ {run_name}: บันทึกผลลัพธ์ไว้ที่ {project_name}/{run_name}")
    return results

if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")

    print("\n--- Starting Experiment 1: Baseline (30 Epochs) ---")
    train_model(epochs=30, imgsz=640, batch=16, run_name="epoch_30")

    print("\n--- Starting Experiment 2: More Epochs (50 Epochs) ---")
    train_model(epochs=50, imgsz=640, batch=16, run_name="epoch_50")

    print("\n--- Starting Experiment 3: Different Batch/Size (50 Epochs, Batch 8, ImgSize 416) ---")
    train_model(epochs=50, imgsz=416, batch=8, run_name="epoch_50_img416_b8")