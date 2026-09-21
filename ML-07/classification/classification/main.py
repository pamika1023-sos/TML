from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_dataset
from cnn_model import build_cnn_model
from evaluate import evaluate_model, plot_history
from test_cnn import test_model
import pandas as pd
import numpy as np

def main():
    print("1. Loading dataset...")
    df = load_data()
    
    print("2. Preprocessing and Standardizing data...")
    X, y = preprocess_data(df)
    
    print("3. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # CNN แบบ 1D ต้องการ input เป็น 3 มิติ (samples, features, 1)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    input_dim = X_train.shape[1]
    epochs_list = [10, 20]
    configs = [1, 2]
    
    # ตารางเก็บผลสรุปตามที่ Lab ต้องการ
    summary_results = []
    
    for config in configs:
        for epochs in epochs_list:
            print(f"\n{'='*50}\n--- Training Configuration {config} with {epochs} Epochs ---\n{'='*50}")
            model = build_cnn_model(config_type=config, input_dim=input_dim)
            
            history = model.fit(
                X_train_cnn, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            print("\nEvaluating model...")
            accuracy = evaluate_model(model, X_test_cnn, y_test)
            plot_history(history, title=f"Config {config} ({epochs} Epochs)")
            
            print("Testing model predictions...")
            test_model(model, X_test_cnn, y_test)
            
            # เก็บผลลัพธ์ไว้เปรียบเทียบ
            summary_results.append({
                'Configuration': f"Config {config}",
                'Epochs': epochs,
                'Test_Accuracy': f"{accuracy * 100:.2f}%"
            })
            
    # สร้างไฟล์ Output สรุปผลทั้งหมดเพื่อส่ง Lab
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv("Lab_Summary_Comparison.csv", index=False)
    print("\n" + "*"*50)
    print("บันทึกไฟล์ Lab_Summary_Comparison.csv และไฟล์กราฟ .png สำเร็จแล้ว!")
    print(summary_df)

if __name__ == "__main__":
    main()