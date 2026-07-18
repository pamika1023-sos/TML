import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw() 
print("กรุณาเลือกไฟล์ .csv จากหน้าต่างที่เปิดขึ้นมา...")
file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

if file_path:
    df = pd.read_csv(file_path)
    print(f"✔️ โหลดข้อมูลสำเร็จจากไฟล์: {file_path}")
    print(f"ขนาดของข้อมูลเริ่มต้น: {df.shape}")
    print("\n" + "="*50 + "\n")

    print("--- 1. Missing Value Handling ---")
    df_cleaned = df.copy() 

    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            print(f"-> เติมค่าว่างในคอลัมน์ตัวเลข '{col}' ด้วยค่า Median เรียบร้อย")

    object_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
            print(f"-> เติมค่าว่างในคอลัมน์ข้อความ '{col}' ด้วยคำว่า 'Unknown' เรียบร้อย")
            
    print("\n" + "="*50 + "\n")

    print("--- 2. Duplicate Removal ---")
    initial_rows = df_cleaned.shape[0]
    df_cleaned = df_cleaned.drop_duplicates(keep='first') 
    final_rows = df_cleaned.shape[0]
    print(f"ลบแถวที่ข้อมูลซ้ำกันออกไป: {initial_rows - final_rows} แถว")
    print(f"คงเหลือข้อมูลทั้งหมด: {final_rows} แถว")
    print("\n" + "="*50 + "\n")

    print("--- 3. Incorrect Data Correction ---")
    for col in object_cols:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
    if 'release_year' in df_cleaned.columns:
        df_cleaned.loc[df_cleaned['release_year'] < 0, 'release_year'] = np.nan
        df_cleaned['release_year'] = df_cleaned['release_year'].fillna(df_cleaned['release_year'].median())
        
    print("ทำความสะอาดและจัดฟอร์แมตข้อความ/ข้อมูลผิดพลาดเบื้องต้นเรียบร้อย")
    print("\n" + "="*50 + "\n")

    print("--- 4. Data Type Conversion ---")
    if 'release_year' in df_cleaned.columns:
        df_cleaned['release_year'] = df_cleaned['release_year'].astype(int)
        print("-> แปลงประเภทคอลัมน์ 'release_year' ให้เป็น Integer เรียบร้อย")
        
    print("\nประเภทข้อมูลล่าสุดหลังจากตรวจสอบ:")
    print(df_cleaned.dtypes)
    print("\n" + "="*50 + "\n")

    print("--- Compare: Mean vs Median ---")
    if len(numeric_cols) > 0:
        mean_vals = df_cleaned[numeric_cols].mean()
        median_vals = df_cleaned[numeric_cols].median()

        comparison_table = pd.DataFrame({
            'Mean (ค่าเฉลี่ย)': mean_vals,
            'Median (มัธยฐาน)': median_vals,
            'Difference (ผลต่าง)': abs(mean_vals - median_vals)
        })
        print(comparison_table)
    else:
        print("ไม่พบข้อมูลที่เป็นตัวเลขในชุดข้อมูลนี้")
    print("\n" + "="*50 + "\n")

    output_filename = 'cleaned_data.csv'
    df_cleaned.to_csv(output_filename, index=False)
    print(f"🎉 บันทึกไฟล์ข้อมูลที่สะอาดแล้วลงในชื่อ '{output_filename}' เรียบร้อยแล้วครับ!")

else:
    print("คุณปิดหน้าต่างและไม่ได้เลือกไฟล์ใดๆ โปรดรันโปรแกรมใหม่อีกครั้ง")