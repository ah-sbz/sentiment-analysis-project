# Base image
FROM python:3.11-slim

# App working directory
WORKDIR /app

# Default model path (can be overridden at build/run time)
ARG MODEL_PATH=models/sentiment.joblib
ENV MODEL_PATH=$MODEL_PATH

# Install runtime dependencies
COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy both entry points (CLI + API) and model assets
COPY src/predict.py src/predict.py
COPY src/predict_api.py src/predict_api.py
COPY models/ models/

# API listens on this container port
EXPOSE 8000

# Default container mode: run FastAPI service
# To use CLI instead, override command at runtime:
# docker run --rm <image> python src/predict.py "I loved it" "Terrible movie"
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
