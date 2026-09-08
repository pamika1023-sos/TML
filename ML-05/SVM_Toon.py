import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv(r"D:\Code\toon\ML05\healthcare-dataset-stroke-data.csv")

# Handle Missing Values
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Remove rare category
df = df[df["gender"] != "Other"]

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Features and Target
X = df.drop(columns=["stroke", "id"])
y = df["stroke"]

# Convert categorical data
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Linear": SVC(kernel="linear", class_weight="balanced"),
    "Polynomial": SVC(kernel="poly", degree=3, class_weight="balanced"),
    "RBF": SVC(kernel="rbf", class_weight="balanced")
}

print("\n========== Accuracy Results ==========")

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} Kernel Accuracy: {accuracy:.4f}")

# Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

prediction = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": prediction
})

print("\nSample Predictions:")
print(comparison.head())

print("\nBest Kernel:", best_model_name)
print("Accuracy:", round(results[best_model_name], 4))

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Kernel", "Accuracy"]
)

print("\nResults DataFrame:")
print(results_df)

# ======================================================
# Graph C Parameter
# ======================================================

C_values = [0.01, 0.1, 1, 10, 100]

train_acc_c = []
test_acc_c = []

for c in C_values:

    model = SVC(
        kernel="rbf",
        C=c,
        gamma="scale",
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    train_acc_c.append(
        model.score(X_train, y_train)
    )

    test_acc_c.append(
        model.score(X_test, y_test)
    )

plt.figure(figsize=(10, 5))

plt.plot(
    C_values,
    train_acc_c,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    C_values,
    test_acc_c,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xscale("log")

plt.title("Training vs Validation Accuracy (C Parameter)")
plt.xlabel("C Value")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()

# Best C

best_c = C_values[
    test_acc_c.index(max(test_acc_c))
]

print("\nBest C =", best_c)
print("Best Accuracy =", max(test_acc_c))

# ======================================================
# Graph Gamma Parameter
# ======================================================

gamma_values = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1
]

train_acc_gamma = []
test_acc_gamma = []

for g in gamma_values:

    model = SVC(
        kernel="rbf",
        C=1,
        gamma=g,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    train_acc_gamma.append(
        model.score(X_train, y_train)
    )

    test_acc_gamma.append(
        model.score(X_test, y_test)
    )

plt.figure(figsize=(10, 5))

plt.plot(
    gamma_values,
    train_acc_gamma,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    gamma_values,
    test_acc_gamma,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xscale("log")

plt.title("Training vs Validation Accuracy (Gamma Parameter)")
plt.xlabel("Gamma Value")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()

# Best Gamma

best_gamma = gamma_values[
    test_acc_gamma.index(max(test_acc_gamma))
]

print("\nBest Gamma =", best_gamma)
print("Best Accuracy =", max(test_acc_gamma))