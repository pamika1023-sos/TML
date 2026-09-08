# Healthcare Stroke Prediction using Support Vector Machine (SVM)

โปรเจกต์นี้เป็นการสร้างและประเมินโมเดล Machine Learning ประเภท **Support Vector Machine (SVM)** เพื่อทำนายความเสี่ยงในการเกิดโรคหลอดเลือดสมอง (Stroke) โดยใช้ชุดข้อมูลสุขภาพ พร้อมทั้งมีการเปรียบเทียบประสิทธิภาพของ Kernel ต่างๆ และการจูนไฮเปอร์พารามิเตอร์ (Hyperparameter Tuning) สำหรับค่า $C$ และ $\gamma$ (Gamma)

---

## 🛠️ ขั้นตอนการทำงานของโค้ด (Workflow)

### 1. การจัดการข้อมูลดิบและการทำความสะอาด (Data Preprocessing)
- โหลดข้อมูลจากไฟล์ CSV และแปลงคอลัมน์ `bmi` ให้เป็นตัวเลข พร้อมจัดการค่าที่หายไป (Missing Values) ด้วยค่ามัธยฐาน (Median)
- กรองข้อมูลในคอลัมน์ `gender` โดยตัดกลุ่มที่เป็น `"Other"` ออกเพื่อความเสถียรของข้อมูล
- แยกตัวแปรต้น (Features: $X$) และตัวแปรเป้าหมาย (Target: $y$ - คอลัมน์ `stroke`) โดยตัดคอลัมน์ `id` และ `stroke` ออก
- แปลงข้อมูลที่เป็น Categorical ให้เป็น Numerical ด้วยวิธี One-Hot Encoding (`pd.get_dummies`)

### 2. การแบ่งข้อมูลและการปรับสเกล (Train-Test Split & Scaling)
- แบ่งข้อมูลเป็นชุดฝึกสอน (Training Set) 80% และชุดทดสอบ (Test Set) 20% โดยใช้ `stratify=y` เพื่อรักษาสัดส่วนของกลุ่มข้อมูลที่ไม่สมดุล (Class Imbalance)
- ปรับสเกลข้อมูล (Feature Scaling) ให้มีค่ามาตรฐานเดียวกันด้วย `StandardScaler`

### 3. การเปรียบเทียบประสิทธิภาพ SVM ด้วย Kernel ต่างๆ
- ทดสอบโมเดล SVC ด้วย 3 เคอร์เนลหลัก ได้แก่ **Linear**, **Polynomial (degree=3)** และ **RBF** 
- กำหนด `class_weight="balanced"` เนื่องจากข้อมูลโรคหลอดเลือดสมองมักมีความไม่สมดุล (Imbalanced Dataset)
- แสดงผลลัพธ์ความแม่นยำ (Accuracy) ของแต่ละ Kernel และเลือก Kernel ที่ให้ผลลัพธ์ดีที่สุด

### 4. การจูนไฮเปอร์พารามิเตอร์ (Hyperparameter Tuning)
- **C Parameter Tuning:** ทดสอบค่า $C$ ในช่วง $[0.01, 0.1, 1, 10, 100]$ ด้วย RBF Kernel และพล็อตกราฟเปรียบเทียบระหว่าง Training และ Validation Accuracy
- **Gamma Parameter Tuning:** ทดสอบค่า $\gamma$ ในช่วง $[0.0001, 0.001, 0.01, 0.1, 1]$ ด้วย RBF Kernel (กำหนด $C=1$) และพล็อตกราฟเปรียบเทียบประสิทธิภาพเช่นเดียวกัน

---

## 📦 ไลบรารีที่ต้องใช้ (Requirements)

ติดตั้งไลบรารีที่จำเป็นก่อนรันโปรแกรม:
```bash
pip install pandas scikit-learn matplotlib