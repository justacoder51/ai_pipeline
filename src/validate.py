import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.pkl"


def validate_model() -> float:
    """Validate the trained model against a held-out test set."""
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
    accuracy = validate_model()
    print(f"Validation accuracy: {accuracy:.4f}")
