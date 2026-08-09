import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _encode_categorical(df):
    for col in df.select_dtypes(include=["object", "category"]):
        df[col] = df[col].fillna("Unknown").astype("category").cat.codes
    return df


def load_data(path):
    df = pd.read_csv(path)

    if "stroke" in df.columns:
        y = df["stroke"]
        X = df.drop(columns=["id", "stroke"], errors="ignore")
    elif "hypertension" in df.columns:
        y = df["hypertension"]
        X = df.drop(columns=["id", "hypertension"], errors="ignore")
    else:
        raise ValueError(
            "Expected a healthcare dataset with a 'stroke' or 'hypertension' label column. "
            "Please provide the correct dataset."
        )

    if "bmi" in X.columns:
        X["bmi"] = pd.to_numeric(X["bmi"], errors="coerce")
        X["bmi"] = X["bmi"].fillna(X["bmi"].median())

    X = _encode_categorical(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test