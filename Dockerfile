# Property valuation API.
#   docker build -t property-valuation .
#   docker run -p 8000:8000 property-valuation
# The image trains on the committed fixture at build time so it serves out of the box.
# To serve a model trained on real data, mount a registry and point the service at it:
#   docker run -p 8000:8000 -v "$PWD/mlruns:/app/mlruns" property-valuation
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 PVS_MLFLOW_URI=/app/mlruns

# libgomp1: OpenMP runtime required by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
RUN chown app:app /app
USER app

ARG TRAIN_DATA=tests/fixtures/sample_sales.csv
RUN python scripts/train.py --data "$TRAIN_DATA" --seed 42

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
