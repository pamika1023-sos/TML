import pandas as pd
from sklearn.preprocessing import StandardScaler


def _encode_categorical(df):
    for col in df.select_dtypes(include=["object", "category"]):
        df[col] = df[col].fillna("Unknown").astype("category").cat.codes
    return df


def load_data(file_path):
    df = pd.read_csv(file_path)

    if "stroke" in df.columns:
        X = df.drop(columns=["id", "stroke"], errors="ignore")
        y = df["stroke"]
    elif "hypertension" in df.columns:
        X = df.drop(columns=["id", "hypertension"], errors="ignore")
        y = df["hypertension"]
    else:
        X = df.drop(columns=["id"], errors="ignore")
        y = None

    if "bmi" in X.columns:
        X["bmi"] = pd.to_numeric(X["bmi"], errors="coerce")
        X["bmi"] = X["bmi"].fillna(X["bmi"].median())

    X = _encode_categorical(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df