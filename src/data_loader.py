import pandas as pd
import requests
from io import StringIO
import sys

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
REQUIRED_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]
TARGET = "target"

def load_dataset():
    try:
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"FATAL: could not download dataset: {e}")
        sys.exit(1)

    df = pd.read_csv(StringIO(r.text), sep=";")
    # Create binary target: quality >= 7 -> 1 else 0
    df[TARGET] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])

    # Validation checks
    missing = [c for c in REQUIRED_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"FATAL: missing columns: {missing}")
        sys.exit(1)

    if df[TARGET].nunique() < 2:
        print("FATAL: target has only one class")
        sys.exit(1)

    print(f"Loaded dataset: {df.shape}, positive rate={df[TARGET].mean():.3f}")
    return df

if __name__ == "__main__":
    load_dataset()