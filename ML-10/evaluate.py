import pandas as pd
from ultralytics import YOLO

def evaluate_and_compare(model_paths):
    summary_data = []

    for name, path in model_paths.items():
        print(f"Evaluating {name}...")
        model = YOLO(path)
        
        # ประเมินผลบน Validation/Test Set
        metrics = model.val(data="dataset/data.yaml")
        
        # ดึงค่า Precision, Recall, mAP50, mAP50-95
        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        summary_data.append({
            "Model Configuration": name,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4)
        })

    # แสดงผลเปรียบเทียบในรูปแบบ ตาราง
    df = pd.DataFrame(summary_data)
    print("\n=================== Model Performance Comparison ===================")
    print(df.to_string(index=False))
    
    # บันทึกเป็น CSV
    df.to_csv("runs/evaluation_summary.csv", index=False)
    print("\nบันทึกสรุปผลลัพธ์ไว้ที่ runs/evaluation_summary.csv แล้ว")

if __name__ == "__main__":
    # ระบุโมเดลที่ต้องการนำมาเปรียบเทียบ
    models_to_eval = {
        "Epoch 30": "runs/detect/epoch_30/weights/best.pt",
        "Epoch 50": "runs/detect/epoch_50/weights/best.pt",
        "Epoch 50 (Img416, Batch8)": "runs/detect/epoch_50_img416_b8/weights/best.pt"
    }
    
    evaluate_and_compare(models_to_eval)