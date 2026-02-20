import os
from typing import Any

import pytest
from dotenv import load_dotenv
from src.predict import load_model, predict_texts, resolve_model_path

load_dotenv()


@pytest.fixture(scope="module")
def model() -> Any:
    return load_model(os.getenv("MODEL_PATH", "models/sentiment.joblib"))


def test_resolve_model_path_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "models/custom-sentiment.joblib")
    assert resolve_model_path() == "models/custom-sentiment.joblib"


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
