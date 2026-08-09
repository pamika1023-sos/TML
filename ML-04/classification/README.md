# 🩺 Healthcare Data Loader & KNN Model

---

##  Data Loader (`data_loader.py`)
- โหลดข้อมูลจากไฟล์ CSV
- เลือกคอลัมน์เป้าหมาย (label) เป็น `stroke` หรือ `hypertension`
- จัดการคอลัมน์ `bmi` (แปลงเป็นตัวเลขและเติมค่าที่หายไปด้วย median)
- เข้ารหัสข้อมูล categorical ด้วย `.cat.codes`
- แบ่งข้อมูลเป็น train/test และปรับสเกลด้วย `StandardScaler`

---

##  KNN Model (`knn_model.py`)
- ฟังก์ชัน `train_knn(X_train, y_train, k)`  
  สร้างและฝึกโมเดล **KNeighborsClassifier** โดยกำหนดค่า `k` (จำนวน neighbors)

---

##  Evaluation (`evaluate.py`)
- ฟังก์ชัน `evaluate(model, X_test, y_test, k)`  
  - ทำนายผลจากโมเดล  
  - คำนวณ Accuracy  
  - สร้าง Confusion Matrix และบันทึกเป็นรูปภาพ `outputs/02_confusion_matrix.png`  
  - บันทึกผลการทำนายลงไฟล์ `outputs/predictions.csv`

---

##  Main Script (`main.py`)
- โหลดข้อมูลด้วย `load_data`
- กำหนดค่า `k_values = [3, 5, 7]`
- ฝึกและประเมินโมเดล KNN สำหรับแต่ละค่า k
- วาดกราฟ Accuracy vs k และบันทึกเป็น `outputs/01_k_curve.png`
- แสดงผลค่า k ที่ดีที่สุดและ Accuracy ที่สูงสุด

---

##  การใช้งาน
```bash
python main.py
