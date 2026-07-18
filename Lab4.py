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
    print("\n--- ข้อมูลตัวอย่างก่อนทำ Feature Engineering ---")
    print(df.head(3))
    print("\n" + "="*50 + "\n")

    object_cols = df.select_dtypes(include=['object']).columns
    print(f"คอลัมน์ที่เป็นข้อความในไฟล์ของคุณคือ: {list(object_cols)}")
    
    if len(object_cols) > 0:
        target_col = object_cols[0] 
        print(f"-> ขอเลือกคอลัมน์ '{target_col}' มาสาธิตการทำงานครับ\n")
        print("--- 1. Label Encoding (via Pandas) ---")
        df_label = df.copy()
        df_label[f'{target_col}_encoded'] = df_label[target_col].astype('category').cat.codes
        
        print("จับคู่ข้อความเดิม -> ตัวเลขใหม่:")
        categories = df_label[target_col].astype('category').cat.categories
        for index, class_name in enumerate(categories):
            print(f"   {class_name} ==> {index}")
            
        print("\nผลลัพธ์ข้อมูลหลังทำ Label Encoding (ดูคอลัมน์ใหม่ด้านท้าย):")
        print(df_label[[target_col, f'{target_col}_encoded']].head(5))
        print("\n" + "="*50 + "\n")
        print("--- 2. One-Hot Encoding ---")
        df_onehot = df.copy()

        df_onehot_result = pd.get_dummies(df_onehot, columns=[target_col], prefix=target_col, dtype=int)
        
        new_onehot_cols = [col for col in df_onehot_result.columns if col.startswith(target_col)]
        
        print(f"คอลัมน์ใหม่ที่ถูกสร้างขึ้นมาแทนที่: {new_onehot_cols}\n")
        print("ผลลัพธ์ข้อมูลหลังทำ One-Hot Encoding (เฉพาะคอลัมน์ที่สร้างใหม่):")
        print(df_onehot_result[new_onehot_cols].head(5))
        print("\n" + "="*50 + "\n")
        
    else:
        print("❌ ไม่พบคอลัมน์ที่เป็นข้อความในไฟล์นี้ จึงไม่สามารถทำ Encoding ได้")

else:
    print("คุณไม่ได้เลือกไฟล์ใดๆ")