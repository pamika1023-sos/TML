# Car License Plate Detection using YOLOv8 (Fast Training Guide)

โครงการนี้เป็นระบบตรวจจับป้ายทะเบียนรถยนต์ (Car License Plate Detection) โดยใช้สถาปัตยกรรม **YOLOv8** ที่ถูกปรับแต่งมาเน้น **ความเร็วสูงสุดในการเทรน ( Fast Training Pipeline)** สามารถเทรนเสร็จสิ้นภายในเวลาไม่ถึง 1 ชั่วโมง (โดยทั่วไปประมาณ 5-15 นาทีบนระบบที่รองรับ GPU)

---

## 📌 Dataset Credit & Reference

Dataset ที่ใช้ในโครงการนี้ถูกดึงมาจาก Kaggle:
* **Dataset Name:** Car Number Plate Dataset YOLO Format
* **Author / Publisher:** Sujay Mann
* **Source Link:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/sujaymann/car-number-plate-dataset-yolo-format)

---

## ⚡ Key Optimizations for <1 Hour Training

เพื่อให้การเทรนเสร็จสมบูรณ์รวดเร็วที่สุด มีการปรับตั้งค่าพารามิเตอร์ดังนี้:

1. **Model Backbone:** ใช้ `YOLOv8n` (Nano Version) ซึ่งมีพารามิเตอร์ขนาดเล็กแต่ให้ความแม่นยำสูง เหมาะสำหรับการตรวจจับวัตถุประเภทเดียวอย่างป้ายทะเบียน
2. **Image Resolution:** ปรับขนาดภาพลงมาเป็น `416x416` พิกเซล ช่วยลดภาระการคำนวณของ GPU/CPU ได้มากกว่า 50% เมื่อเทียบกับขนาดปกติ (`640x640`)
3. **Multi-threading Data Loader:** กำหนด `workers=8` เพื่อให้ CPU ทำการ Pre-process ภาพและโหลดขนานไปกับการประมวลผลของ GPU ป้องกันปัญหา GPU Bottleneck
4. **Batch Size:** ตั้งค่า `batch=32` เพื่อเร่งความเร็วในการส่งข้อมูลผ่านกระบวนการ Forward/Backward Pass

---

## 🛠️ Installation & Setup

### 1. Requirements
* Python 3.8 หรือใหม่กว่า
* CUDA-supported NVIDIA GPU (แนะนำสำหรับการเทรนอย่างรวดเร็ว)

### 2. Install Dependencies
ติดตั้งไลบรารีที่จำเป็นผ่าน `pip`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics kaggle
```

### 3. Download Dataset
คุณสามารถดาวน์โหลด Dataset จาก Kaggle ผ่านคำสั่ง CLI หรือดาวน์โหลดด้วยตนเอง:

```bash
kaggle datasets download -d sujaymann/car-number-plate-dataset-yolo-format
```
*นำไฟล์ `.zip` ที่ได้มาวางไว้ในโฟลเดอร์เดียวกับสคริปต์ `train.py`*

---

## 🚀 How to Run Training

รันสคริปต์การเทรนด้วยคำสั่ง:

```bash
python train.py
```

สคริปต์จะทำการ:
1. ตรวจสอบการทำงานของ GPU และ CPU Automaticaly
2. แตกไฟล์ Dataset และค้นหาไฟล์ `data.yaml`
3. เริ่มต้นกระบวนการเทรนโมเดล YOLOv8n จำนวน 30 Epochs
4. บันทึกผลลัพธ์และน้ำหนักโมเดลที่ดีที่สุดไว้ที่ `runs/detect/license_plate_fast/weights/best.pt`

---

## 📊 Inference / Test the Trained Model

หลังจากเทรนเสร็จแล้ว สามารถนำโมเดลไปทดสอบตรวจจับภาพใหม่ได้ดังนี้:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/license_plate_fast/weights/best.pt')

# Run inference on an image
results = model('path/to/test_image.jpg', save=True, conf=0.25)
```

---

## 📄 License
โปรดระวังเรื่องลิขสิทธิ์และข้อตกลงการใช้งานของ Dataset ตามข้อกำหนดบนหน้า Kaggle ของผู้สร้างเดิม