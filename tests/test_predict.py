from typing import Any

import pytest
from src.predict import load_model, predict_texts


@pytest.fixture(scope="module")
def model() -> Any:
    return load_model("models/sentiment.joblib")


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("I love this movie, it was fantastic and inspiring!", 1),
        ("The service was terrible and the food was awful.", 0),
    ],
)
def test_predict_obvious_sentiment(model: Any, text: str, expected_label: int) -> None:
    preds, probs = predict_texts(model, [text])
    assert preds[0] == expected_label
    assert probs[0] is None or 0.0 <= probs[0] <= 1.0
