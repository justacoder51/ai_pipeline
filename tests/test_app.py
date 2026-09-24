from src.train import train_model
from src.validate import validate_model


def test_training_and_validation_pipeline():
    model = train_model()
    assert model is not None

    accuracy = validate_model()
    assert 0.0 <= accuracy <= 1.0
