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

# Copy application assets
COPY src/predict.py src/predict.py
COPY models/ models/

# Fixed command
ENTRYPOINT [ "python", "src/predict.py" ]

# Default input texts
CMD ["I absolutely loved it", "That was awful"]
