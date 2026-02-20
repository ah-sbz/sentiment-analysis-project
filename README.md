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

Run with default model path (from image/filesystem):
```bash
docker run --rm sentiment-analysis-app:latest "I absolutely loved it" "That was awful"
```

Override model path with env var at runtime:
```bash
docker run --rm -e MODEL_PATH=models/sentiment.joblib sentiment-analysis-app:latest "Great movie"
```
