# Data Loader (data_loader.py)

ไฟล์ `data_loader.py` มีหน้าที่หลักในการเตรียมข้อมูลสำหรับการทำ Machine Learning โดยสรุปได้ดังนี้:

## โหลดข้อมูลจาก CSV
- ใช้ `pandas.read_csv()` เพื่ออ่านไฟล์ข้อมูลที่กำหนด
- ตรวจสอบว่ามีคอลัมน์เป้าหมาย (`stroke` หรือ `hypertension`) หรือไม่
- แยก **Features (X)** และ **Target (y)** ออกจากกัน

## จัดการค่าที่หายไป (Missing Values)
- คอลัมน์ `bmi` ถูกแปลงเป็นตัวเลขด้วย `pd.to_numeric()`
- เติมค่าที่หายไปด้วย **ค่ามัธยฐาน (median)** ของคอลัมน์นั้น

## เข้ารหัสข้อมูลเชิงหมวดหมู่ (Categorical Encoding)
- ฟังก์ชัน `_encode_categorical()` ใช้การเข้ารหัสแบบ **category codes**
- ค่าที่หายไปในคอลัมน์ประเภท object/category จะถูกแทนด้วย `"Unknown"`

## การปรับสเกลข้อมูล (Feature Scaling)
- ใช้ `StandardScaler()` จาก `sklearn.preprocessing`
- ทำการปรับสเกลข้อมูลทั้งหมดให้อยู่ในช่วงมาตรฐาน (ค่าเฉลี่ย = 0, ส่วนเบี่ยงเบนมาตรฐาน = 1)

## 5. ส่งออกผลลัพธ์
- คืนค่าออกมาเป็น `(X_scaled, y, df)`
  - `X_scaled` → ข้อมูลที่ถูกปรับสเกลแล้ว
  - `y` → ค่าผลลัพธ์ (Target) ถ้ามี
  - `df` → DataFrame ดั้งเดิมที่โหลดมา
