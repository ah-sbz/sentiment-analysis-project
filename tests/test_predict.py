import pytest
from src.predict import load_model, predict_texts

@pytest.fixture(scope='module')
def model():
    return load_model("models/sentiment.joblib")

def test_obvious_positive_sentence(model):
    preds, probs = predict_texts(model, ["I love this movie, it was fantastic and inspiring!"])
    assert preds[0] == 1
    assert probs[0] is None or 0.0 <= probs[0] <= 1.0

def test_obvious_negative_sentence(model):
    preds, probs = predict_texts(model, ["The service was terrible and the food was awful."])
    assert preds[0] == 0
    assert probs[0] is None or 0.0 <= probs[0] <= 1.0
    