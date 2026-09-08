# Healthcare Stroke Prediction using Deep Learning (Artificial Neural Network)

โปรเจกต์นี้เป็นการสร้างและประเมินโมเดล Deep Learning ประเภท **Artificial Neural Network (ANN)** โดยใช้ TensorFlow/Keras เพื่อทำนายความเสี่ยงในการเกิดโรคหลอดเลือดสมอง (Stroke) พร้อมทั้งมีการทดลองปรับเปลี่ยนจำนวน Epochs และโครงสร้างสถาปัตยกรรมของโครงข่ายประสาทเทียม (Neural Network Architecture) รวมถึงการแสดงผลลัพธ์ผ่านกราฟและ Confusion Matrix

---

## 🛠️ ขั้นตอนการทำงานของโค้ด (Workflow)

### 1. การจัดการข้อมูลดิบและการทำความสะอาด (Data Preprocessing)
- โหลดข้อมูลจากไฟล์ CSV และแสดงข้อมูลเบื้องต้นของ Dataset รวมถึงตรวจสอบค่าที่สูญหาย (Missing Values)
- แปลงคอลัมน์ `bmi` ให้เป็นตัวเลข และเติมค่าที่หายด้วยค่ามัธยฐาน (Median)
- กรองข้อมูลในคอลัมน์ `gender` โดยตัดกลุ่มที่เป็น `"Other"` ออก
- แยกตัวแปรต้น (Features: $X$) และตัวแปรเป้าหมาย (Target: $y$ - คอลัมน์ `stroke`) โดยตัดคอลัมน์ `id` และ `stroke` ออก
- แปลงข้อมูลที่เป็น Categorical ให้เป็น Numerical ด้วยวิธี One-Hot Encoding (`pd.get_dummies`)

### 2. การแบ่งข้อมูลและการปรับสเกล (Train-Test Split & Scaling)
- แบ่งข้อมูลเป็นชุดฝึกสอน (Training Set) 80% และชุดทดสอบ (Test Set) 20% โดยใช้ `stratify=y` เพื่อรักษาสัดส่วนข้อมูล
- ปรับสเกลข้อมูล (Feature Scaling) ให้มีค่ามาตรฐานเดียวกันด้วย `StandardScaler`

### 3. การสร้างฟังก์ชันจำลองโมเดล (Model Creation Function)
- สร้างฟังก์ชัน `create_model(hidden_layers)` สำหรับกำหนดโครงสร้างของโครงข่ายประสาทเทียม (ANN) แบบ Sequential โดยใช้ Activation Function เป็น **ReLU** สำหรับ Hidden Layers และ **Sigmoid** สำหรับ Output Layer (สำหรับงาน Binary Classification)
- คอมไพล์โมเดลด้วย Optimizer เป็น `adam` และ Loss Function เป็น `binary_crossentropy`

### 4. การทดลองปรับแต่งโมเดล (Experiments)
- **Experiment 1 (Different Epochs):** ทดสอบโมเดลด้วยจำนวน Epochs ที่แตกต่างกัน (10, 20, 30 รอบ) ด้วยสถาปัตยกรรม `[16, 8]`
- **Experiment 2 (Different Architectures):** เปรียบเทียบสถาปัตยกรรมของโครงข่ายประสาทเทียม ระหว่างแบบ 1 Hidden Layer (`[16]`) และ 2 Hidden Layers (`[16, 8]`)

### 5. การเทรนและประเมินโมเดลตัวจริง (Final Model & Evaluation)
- สร้างและเทรนโมเดลสุดท้ายด้วยโครงสร้าง `[16, 8]` เป็นเวลา 20 Epochs พร้อมแบ่ง Validation Split 20%
- ประเมินผลโมเดลกับชุดทดสอบ (Test Set) คำนวณค่า Accuracy และสร้าง Confusion Matrix
- พล็อตกราฟแสดงผลการเรียนรู้ของโมเดล ได้แก่:
  - กราฟเปรียบเทียบ Training vs Validation Accuracy
  - กราฟเปรียบเทียบ Training vs Validation Loss
  - กราฟเปรียบเทียบผลลัพธ์ของ Epoch และสถาปัตยกรรมโมเดล

---

## 📦 ไลบรารีที่ต้องใช้ (Requirements)

ติดตั้งไลบรารีที่จำเป็นก่อนรันโปรแกรม:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow