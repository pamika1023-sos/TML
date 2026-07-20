import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():

    print("--- 1. Loading and Preprocessing Data ---")
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data-selected-columns.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: The file 'healthcare-dataset-stroke-data-selected-columns.csv' was not found.")
        print("Please ensure the CSV file is in the same directory as this script.")
        return

    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # กำหนดค่า Target (Y) คือ 'age' ที่ต้องการทำนาย
    y = df_encoded['age']

    print("\n--- 2. Part 1: Simple Linear Regression ---")
    print("Predicting 'age' using 'bmi' as the single independent variable.")
    
    X_simple = df_encoded[['bmi']]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

    simple_model = LinearRegression()
    simple_model.fit(X_train_s, y_train_s)

    y_pred_s = simple_model.predict(X_test_s)
    mse_s = mean_squared_error(y_test_s, y_pred_s)
    r2_s = r2_score(y_test_s, y_pred_s)

    print(f"Intercept (Beta 0)      : {simple_model.intercept_:.4f}")
    print(f"Coefficient (Beta 1)    : {simple_model.coef_[0]:.4f}")
    print(f"Mean Squared Error (MSE): {mse_s:.4f}")
    print(f"R-squared (R²)          : {r2_s:.4f}")

    print("\n--- 3. Part 2: Multiple Linear Regression ---")
    print("Predicting 'age' using all available features (excluding 'id').")

    X_multiple = df_encoded.drop(columns=['id', 'age'])

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

    multiple_model = LinearRegression()
    multiple_model.fit(X_train_m, y_train_m)

    y_pred_m = multiple_model.predict(X_test_m)
    mse_m = mean_squared_error(y_test_m, y_pred_m)
    r2_m = r2_score(y_test_m, y_pred_m)

    print(f"Mean Squared Error (MSE): {mse_m:.4f}")
    print(f"R-squared (R²)          : {r2_m:.4f}")

    print("\n--- 4. Generating Plots ---")
    plt.figure(figsize=(14, 5))

    # กราฟสำหรับ Simple Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(X_test_s, y_test_s, color='gray', alpha=0.3, label='Actual Data')
    x_line = np.linspace(X_test_s.min(), X_test_s.max(), 100).reshape(-1, 1)
    y_line = simple_model.predict(x_line)
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')
    plt.title(f'Simple Linear Regression\n(R² = {r2_s:.2f})')
    plt.xlabel('BMI')
    plt.ylabel('Age')
    plt.legend()

    # กราฟสำหรับ Multiple Linear Regression
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_m, y_pred_m, color='blue', alpha=0.3)
    plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], color='red', linestyle='--', linewidth=2)
    plt.title(f'Multiple Linear Regression\nActual vs Predicted (R² = {r2_m:.2f})')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')

    plt.tight_layout()
    print("Displaying plots. Close the window to finish.")
    plt.show()

if __name__ == "__main__":
    main()