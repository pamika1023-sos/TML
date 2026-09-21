import pandas as pd

def load_data():
    # โหลดไฟล์ข้อมูล CSV ตามชื่อไฟล์ใหม่
    df_stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return df_stroke