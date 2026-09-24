import pickle
from pathlib import Path

from sklearn.datasets import load_iris


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.pkl"


def predict_sample(sample: list[float]) -> int:
    """Predict the class for a single flower sample."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    return int(model.predict([sample])[0])


if __name__ == "__main__":
    iris = load_iris()
    sample = iris.data[0].tolist()
    prediction = predict_sample(sample)
    print(f"Sample prediction: {prediction}")
