import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import load_model, predict
from data_loader import REQUIRED_FEATURES

SAMPLE = {c: 1.0 for c in REQUIRED_FEATURES}

def test_model_can_be_loaded():
    model = load_model()
    assert model is not None

def test_prediction_shape_and_type():
    pred = predict(SAMPLE)
    assert isinstance(pred, int)
    assert pred in (0, 1)

def test_missing_feature_rejected():
    bad = dict(SAMPLE)
    bad.pop(REQUIRED_FEATURES[0])
    with pytest.raises(ValueError, match="Missing required features"):
        predict(bad)