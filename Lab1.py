import pandas as pd

df = pd.read_csv('healthcare-dataset-stroke-data-selected-columns.csv')
print("--- 1. Load Dataset (5 แถวแรก) ---")
print(df.head())
print("\n" + "="*50 + "\n")

print("--- 2. Display Shape ---")
print(f"จำนวนแถว (Rows): {df.shape[0]}")
print(f"จำนวนคอลัมน์ (Columns): {df.shape[1]}")
print(f"Shape ทั้งหมด: {df.shape}")
print("\n" + "="*50 + "\n")

print("--- 3. Display Data Types ---")
print(df.dtypes)
print("\n" + "="*50 + "\n")

print("--- 4. Display Summary Statistics ---")
print(df.describe())
print("\n" + "="*50 + "\n")

print("--- 5. Display Missing Values ---")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

print("--- 6. Display Duplicate Records ---")
duplicate_count = df.duplicated().sum()
print(f"จำนวนข้อมูลที่ซ้ำกันทั้งหมด: {duplicate_count} แถว")
print("\n" + "="*50 + "\n")

print("--- 7. Display Class Distribution ---")
if 'class' in df.columns:
    print(df['class'].value_counts())
else:
    last_column = df.columns[-1]
    print(f"ไม่พบคอลัมน์ชื่อ 'class' จึงแสดงคอลัมน์สุดท้าย ({last_column}) ให้แทน:")
    print(df[last_column].value_counts())