FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        "numpy>=1.24" \
        "fastapi>=0.110.0" \
        "python-multipart>=0.0.9" \
        "sentencepiece>=0.1.99" \
        "uvicorn>=0.29.0" \
        "WeTextProcessing>=1.0.4.1" \
        "soundfile" \
        "onnxruntime>=1.20.0" \
        "huggingface_hub" \
        "scipy"

COPY . .
RUN python -m pip install --no-deps -e .

RUN mkdir -p /app/models /app/generated_audio /app/.cache/huggingface

EXPOSE 18083

VOLUME ["/app/models", "/app/generated_audio", "/app/.cache/huggingface"]

CMD ["python", "app_onnx.py", "--host", "0.0.0.0", "--port", "18083"]
