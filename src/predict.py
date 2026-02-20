"""CLI and helper functions for sentiment model inference."""

import argparse
import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from joblib import load
from numpy.typing import NDArray

load_dotenv()
DEFAULT_MODEL_PATH = "models/sentiment.joblib"


def resolve_model_path() -> str:
    """Return model path from env, falling back to default project path."""
    return os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


def predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Return labels and probability-of-positive for each text."""
    preds: NDArray[Any] = classifier.predict(input_texts)
    probs: list[float | None]
    if hasattr(classifier, "predict_proba"):
        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs = [float(p) for p in probs_arr.tolist()]
    else:
        probs = [None] * len(input_texts)
    return preds.astype(int).tolist(), probs


def format_prediction_lines(
    texts: list[str], preds: list[int], probs: list[float | None]
) -> list[str]:
    """Return tab-separated CLI output lines for each input text."""
    lines: list[str] = []
    for text, pred, prob in zip(texts, preds, probs, strict=False):
        if prob is None:
            lines.append(f"{pred}\t{text}")
        else:
            lines.append(f"{pred}\t{prob:.3f}\t{text}")
    return lines


def main(model_path: str, input_texts: list[str]) -> None:
    """Load model, score input texts, and print CLI output lines."""
    classifier = load_model(model_path)
    preds, probs = predict_texts(classifier, input_texts)
    for line in format_prediction_lines(input_texts, preds, probs):
        print(line)


if __name__ == "__main__":
    env_model_path = resolve_model_path()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=env_model_path,
        help="Path to model file (default: "
        "MODEL_PATH env var or models/sentiment.joblib)",
    )
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)
