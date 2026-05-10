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

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .

RUN mkdir -p /app/models /app/generated_audio /app/.cache/huggingface

EXPOSE 18083

VOLUME ["/app/models", "/app/generated_audio", "/app/.cache/huggingface"]

CMD ["python", "app_onnx.py", "--host", "0.0.0.0", "--port", "18083"]
