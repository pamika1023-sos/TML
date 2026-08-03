import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print("ชื่อคอลัมน์ทั้งหมดในไฟล์:", df.columns.tolist())

if 'id' in df.columns:
    df = df.drop('id', axis=1)

if 'bmi' in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ตรวจสอบว่ามีคอลัมน์ stroke หรือไม่ ถ้าไม่มีให้ใช้คอลัมน์สุดท้ายเป็น Target แทน
target_col = 'stroke' if 'stroke' in df.columns else df.columns[-1]
print(f"ใช้คอลัมน์ '{target_col}' เป็น Target (y)")

X = df.drop(target_col, axis=1)
y = df[target_col]

# หากข้อมูลใน Target เป็นแบบ continuous (ไม่ใช่ตัวเลข 0/1 สำหรับแบ่งกลุ่ม) ให้แปลงเป็นคลาสจำแนกประเภท
if y.dtype == 'float64' or y.nunique() > 10:
    y = pd.qcut(y, q=2, labels=[0, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [3, 5, 7]
accuracy_results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[k] = acc
    print(f"K = {k} -> Accuracy: {acc:.4f}")

best_k = max(accuracy_results, key=accuracy_results.get)
print(f"\nBest k value: {best_k} with accuracy {accuracy_results[best_k]:.4f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
best_y_pred = best_knn.predict(X_test_scaled)

print(f"\nClassification Report for K = {best_k}:")
print(classification_report(y_test, best_y_pred))