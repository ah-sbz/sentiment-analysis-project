import os
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv
from sklearn.metrics import f1_score
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


MIN_F1 = 0.65


def test_f1_score(model: Any) -> None:
    df = pd.read_csv("data/test.csv")
    texts = df["text"].tolist()
    y_true = df["label"].tolist()
    y_pred, _ = predict_texts(model, texts)
    score = f1_score(y_true, y_pred)
    assert score >= MIN_F1, f"F1 {score:.3f} below threshold {MIN_F1:.2f}"
