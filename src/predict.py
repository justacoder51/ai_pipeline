import sys
import joblib
import pandas as pd

from data_loader import REQUIRED_FEATURES

MODEL_PATH = "model/model.joblib"


def load_model(path=MODEL_PATH):
    """Load the trained model pipeline from disk."""
    return joblib.load(path)


def predict(input_dict, model=None):
    """Predict for a single sample given as a dict of feature -> value."""
    if model is None:
        model = load_model()

    missing = [c for c in REQUIRED_FEATURES if c not in input_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([input_dict])[REQUIRED_FEATURES]
    pred = model.predict(X)
    return int(pred[0])


if __name__ == "__main__":
    sample = {c: 1.0 for c in REQUIRED_FEATURES}
    print("Prediction:", predict(sample))