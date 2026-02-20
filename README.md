# Sentiment Analysis Pipeline

## Setup

### Option 1: Python venv
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Option 2: Conda
```bash
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt
cp .env.example .env
```

## Configuration (Environment Variables)

`src/predict.py` reads the model path from `MODEL_PATH`.

- Local development: keep values in `.env` (ignored by git).
- Template: commit `.env.example` with safe defaults only.
- CI/Prod: set environment variables in the platform (do not commit real secrets).

Example `.env`:
```env
MODEL_PATH=models/sentiment.joblib
```

## Train
```bash
python src/train.py --data data/train.csv --out models/sentiment.joblib
```

## Predict
```bash
python src/predict.py "I absolutely loved it" "That was awful"
```

Output format:
- With probabilities: `label<TAB>probability<TAB>text`
- Without probabilities: `label<TAB>text`

Example:
```text
1	0.982	I absolutely loved it
0	0.015	That was awful
```

## Predict API (FastAPI)

Run the API:
```bash
uvicorn src.predict_api:app --host 127.0.0.1 --port 8000 --reload
```

Send a request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I absolutely loved it", "That was awful"]}'
```

Example response:
```json
[
  {"text": "I absolutely loved it", "label": 1, "probability": 0.98},
  {"text": "That was awful", "label": 0, "probability": 0.02}
]
```

## Model Quality Gate

The test suite includes a performance gate in `tests/test_predict.py`:
- It loads the trained model.
- It evaluates on `data/test.csv`.
- It computes binary F1-score (`sklearn.metrics.f1_score`).
- CI fails if F1 is below `MIN_F1 = 0.65`.

Why `0.65` and not `0.85`:
- The current dataset is intentionally small, so a high absolute threshold is unstable.
- The gate is used to catch regressions in CI/CD flow, not to represent final model quality.
- You can raise `MIN_F1` later as data quality/size improves.

## CI And Production Variables/Secrets

Use this split:
- Variables (non-secret): `MODEL_PATH`
- Secrets: credentials/tokens (for this repo, `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` in GitHub Actions)

GitHub Actions:
- Set secrets in `Settings -> Secrets and variables -> Actions -> Secrets`.
- Set non-secret repo variables in `Settings -> Secrets and variables -> Actions -> Variables` if needed.

Example CI step using a repository variable for `MODEL_PATH`:
```yaml
- name: Run prediction smoke check
  env:
    MODEL_PATH: ${{ vars.MODEL_PATH }}
  run: |
    python src/predict.py "CI smoke test"
```

## Docker Build And Run

Build image:
```bash
docker build -t sentiment-analysis-app:latest .
```

Run API (default container command):
```bash
docker run --rm -p 8000:8000 sentiment-analysis-app:latest
```

Call the API from host:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I absolutely loved it", "That was awful"]}'
```

Run CLI from the same image by overriding the command:
```bash
docker run --rm sentiment-analysis-app:latest \
  python src/predict.py "I absolutely loved it" "That was awful"
```

Override model path with env var at runtime (API or CLI):
```bash
docker run --rm -e MODEL_PATH=models/sentiment.joblib -p 8000:8000 sentiment-analysis-app:latest
```
