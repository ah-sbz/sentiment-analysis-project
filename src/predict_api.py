"""Sentiment inference helpers and a FastAPI prediction endpoint."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from joblib import load
from numpy.typing import NDArray
from pydantic import BaseModel, Field

load_dotenv()
DEFAULT_MODEL_PATH = "models/sentiment.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model at startup and keep it in app state for requests."""
    app.state.classifier = load_model(resolve_model_path())
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    """Request payload for batch sentiment prediction."""

    texts: list[str] = Field(..., min_length=1)


class PredictResponseItem(BaseModel):
    """Single prediction result returned by the API."""

    text: str
    label: int
    probability: float | None


def resolve_model_path() -> str:
    """Return model path from ``MODEL_PATH`` or the default model location."""
    return os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)


def load_model(model_path: str) -> Any:
    """Load and return the persisted sentiment model from disk."""
    return load(model_path)


def predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Predict labels and positive-class probabilities for each input text."""
    preds: NDArray[Any] = classifier.predict(input_texts)
    probs: list[float | None]
    if hasattr(classifier, "predict_proba"):
        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs = [float(p) for p in probs_arr.tolist()]
    else:
        probs = [None] * len(input_texts)
    return preds.astype(int).tolist(), probs


@app.post("/predict", response_model=list[PredictResponseItem])
def predict(payload: PredictRequest) -> list[PredictResponseItem]:
    """Return sentiment predictions for one or more input strings."""
    texts = [text.strip() for text in payload.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=422, detail="Input texts must be non-empty.")

    classifier = app.state.classifier
    preds, probs = predict_texts(classifier, texts)
    return [
        PredictResponseItem(text=text, label=pred, probability=prob)
        for text, pred, prob in zip(texts, preds, probs, strict=False)
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
