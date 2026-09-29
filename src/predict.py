import sys
import joblib
import pandas as pd
from pathlib import Path

# Make src/ importable when predict.py is imported from tests
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import REQUIRED_FEATURES

# Absolute path to the model, no matter where Python is invoked from
MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.joblib"


def load_model(path=MODEL_PATH):
    """Load the trained model pipeline from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def predict(input_dict, model=None):
    if model is None:
        model = load_model()

    missing = [c for c in REQUIRED_FEATURES if c not in input_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([input_dict])[REQUIRED_FEATURES]
    return int(model.predict(X)[0])


if __name__ == "__main__":
    sample = {c: 1.0 for c in REQUIRED_FEATURES}
    print("Prediction:", predict(sample))