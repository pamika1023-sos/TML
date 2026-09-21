import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    # จัดการ Missing values ในคอลัมน์ bmi
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    
    # แปลง Categorical variables เป็นตัวเลข (ข้ามคอลัมน์ที่ไม่มีอัตโนมัติ)
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
    # แยก Feature (X) และ Target (y)
    X = df.drop(columns=['id', 'hypertension'])
    y = df['hypertension']
    
    # Standardize input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y