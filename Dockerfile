# Food Vision API: CPU inference image.
# Model weights are not in git: put model.pt, class_names.json and model_config.json
# in ./artifacts before building (see artifacts/README.md).
#
#   docker build -t food-vision .
#   docker run -p 7860:7860 food-vision      ->  http://localhost:7860
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACTS_DIR=/app/artifacts \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch keeps the image ~4x smaller than the default CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY artifacts ./artifacts
RUN test -f artifacts/model.pt || (echo "ERROR: artifacts/model.pt missing, see artifacts/README.md" && exit 1)

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn foodvision.api:app --host 0.0.0.0 --port ${PORT}"]
