from fastapi.testclient import TestClient
from src.predict_api import app


def test_predict_api_returns_predictions() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"texts": ["I absolutely loved it", "That was awful"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["text"] == "I absolutely loved it"
    assert body[1]["text"] == "That was awful"
    assert body[0]["label"] in {0, 1}
    assert body[1]["label"] in {0, 1}


def test_predict_api_rejects_empty_text_item() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"texts": ["   ", "valid"]})

    assert response.status_code == 422
