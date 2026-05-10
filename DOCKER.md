# Docker deployment for MOSS-TTS-Nano

This repository now includes a simple Docker build focused on the **ONNX CPU** web demo and HTTP API.

## What it runs

- Web UI: `http://localhost:18183`
- Backend: the FastAPI app from `app_onnx.py`
- Model storage: mounted to `./models`
- Generated audio: mounted to `./generated_audio`

The container starts with:

```bash
python app_onnx.py --host 0.0.0.0 --port 18083
```

If `./models` does not already contain the ONNX assets, the app will download them on first run into the mounted `models/` directory.

## Option 1: Docker Compose

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:18183
```

Run in background:

```bash
docker compose up -d --build
```

Stop:

```bash
docker compose down
```

## Option 2: Plain Docker

Build:

```bash
docker build -t moss-tts-nano:local .
```

Run:

```bash
docker run --rm -it \
  -p 18183:18083 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/generated_audio:/app/generated_audio" \
  -v "$(pwd)/hf-cache:/app/.cache/huggingface" \
  moss-tts-nano:local
```

## Notes

- This Docker setup is intentionally CPU-first and uses the ONNX path for easier local deployment.
- The image installs an ONNX-only dependency set instead of the full repository requirements, then installs the package itself with `--no-deps` so the existing entrypoints still work without pulling the full PyTorch stack.
- First startup may take a while because ONNX assets can be downloaded automatically.
- If you later want a GPU Docker variant, it is better to add a separate `Dockerfile.gpu` based on `onnxruntime-gpu` or a PyTorch CUDA base image.
