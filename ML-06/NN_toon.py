# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"D:\Code\toon\ML05\healthcare-dataset-stroke-data.csv"
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# bmi มีค่า N/A
df["bmi"] = pd.to_numeric(
    df["bmi"],
    errors="coerce"
)

# เติม Missing ด้วย Median
df["bmi"] = df["bmi"].fillna(
    df["bmi"].median()
)

# ลบ gender = Other
df = df[df["gender"] != "Other"]


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "stroke",
        "id"
    ]
)

y = df["stroke"]

# One Hot Encoding
X = pd.get_dummies(
    X,
    drop_first=True
)

print("\nFeature Shape:", X.shape)


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# ============================================================
# 6. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling Completed")


# ============================================================
# 7. CREATE MODEL FUNCTION
# ============================================================

def create_model(hidden_layers):

    model = Sequential()

    model.add(
        Dense(
            hidden_layers[0],
            activation="relu",
            input_shape=(X_train.shape[1],)
        )
    )

    for neurons in hidden_layers[1:]:

        model.add(
            Dense(
                neurons,
                activation="relu"
            )
        )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 8. EXPERIMENT 1
# DIFFERENT EPOCHS
# ============================================================

epochs_list = [10, 20, 30]

epoch_results = []

print("\n")
print("=" * 60)
print("EXPERIMENT 1 : DIFFERENT EPOCHS")
print("=" * 60)

for epochs in epochs_list:

    model = create_model([16, 8])

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.20,
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    epoch_results.append(
        [epochs, accuracy]
    )

    print(
        f"Epochs {epochs:3d} | Accuracy = {accuracy:.4f}"
    )

epoch_df = pd.DataFrame(
    epoch_results,
    columns=["Epochs", "Accuracy"]
)


# ============================================================
# 9. EXPERIMENT 2
# DIFFERENT ARCHITECTURE
# ============================================================

configs = {
    "NN-1": [16],
    "NN-2": [16, 8],

}

config_results = []

print("\n")
print("=" * 60)
print("EXPERIMENT 2 : DIFFERENT ARCHITECTURES")
print("=" * 60)

for name, layers in configs.items():

    model = create_model(layers)

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.20,
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    config_results.append(
        [name, str(layers), accuracy]
    )

    print(
        f"{name} | Layers {layers} | Accuracy {accuracy:.4f}"
    )

config_df = pd.DataFrame(
    config_results,
    columns=[
        "Model",
        "Hidden Layers",
        "Accuracy"
    ]
)


# ============================================================
# 10. FINAL MODEL
# ============================================================

final_model = create_model(
    [16, 8]
)

history = final_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.20,
    verbose=1
)


# ============================================================
# 11. FINAL EVALUATION
# ============================================================

test_loss, test_accuracy = final_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n")
print("=" * 60)
print("FINAL MODEL RESULT")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy*100:.2f}%")



# ============================================================
# 12. PREDICTION
# ============================================================

y_pred_prob = final_model.predict(
    X_test,
    verbose=0
)

y_pred = (
    y_pred_prob > 0.5
).astype(int)

y_pred = y_pred.flatten()

print("\nPredictions:")
print(y_pred[:20])

print("\nActual:")
print(y_test.values[:20])


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nPrediction Accuracy")
print(
    f"{accuracy*100:.2f}%"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Stroke",
        "Stroke"
    ]
)

disp.plot(
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.show()


# ============================================================
# 15. TRAINING vs VALIDATION ACCURACY
# ============================================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 16. TRAINING vs VALIDATION LOSS
# ============================================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 17. EPOCH COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    epoch_df["Epochs"],
    epoch_df["Accuracy"],
    marker="o"
)

plt.title(
    "Epoch Comparison"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.grid()

plt.show()


# ============================================================
# 18. CONFIGURATION COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(8,5)
)

plt.bar(
    config_df["Model"],
    config_df["Accuracy"]
)

plt.title(
    "Neural Network Architecture Comparison"
)

plt.xlabel("Architecture")
plt.ylabel("Accuracy")

plt.grid(axis="y")

plt.show()


# ============================================================
# 19. SUMMARY
# ============================================================

print("\n")
print("=" * 60)

print("EPOCH COMPARISON")
print(epoch_df)

print("\nCONFIGURATION COMPARISON")
print(config_df)

print("\nFINAL ACCURACY")
print(
    f"{accuracy*100:.2f}%"
)

print("=" * 60)