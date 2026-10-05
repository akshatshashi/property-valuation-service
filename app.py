"""FastAPI service (thin): all logic lives in src/serving.py.

    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import settings
from src.registry import Registry
from src.serving import ModelNotLoadedError, ValuationRequest, ValuationResponse, ValuationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

LATENCY_BUCKETS_MS = [10, 25, 50, 100, 250, 500, 1000, 2500, float("inf")]


class Metrics:
    """In-process request metrics (a Prometheus exporter would replace this in production)."""

    def __init__(self) -> None:
        self.requests: Counter[str] = Counter()
        self.status: Counter[str] = Counter()
        self.latency: Counter[str] = Counter()
        self.errors = 0

    def observe(self, path: str, status: int, ms: float) -> None:
        self.requests[path] += 1
        self.status[str(status)] += 1
        if status >= 500:
            self.errors += 1
        bucket = next(b for b in LATENCY_BUCKETS_MS if ms <= b)
        self.latency["+Inf" if bucket == float("inf") else f"le_{int(bucket)}ms"] += 1

    def snapshot(self) -> dict:
        return {"requests_by_path": dict(self.requests), "responses_by_status": dict(self.status),
                "latency_histogram": dict(self.latency), "error_count": self.errors}


metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ValuationService(Registry(settings.mlflow_uri))
    yield


app = FastAPI(title="Property Valuation Service", version="1.0.0", lifespan=lifespan)


def service(request: Request) -> ValuationService:
    return request.app.state.service


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    metrics.observe(request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "problem": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid input", "details": details})


@app.exception_handler(ModelNotLoadedError)
async def not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "no production model loaded", "detail": str(exc)})


@app.post("/value", response_model=ValuationResponse)
def value(req: ValuationRequest, request: Request) -> ValuationResponse:
    return service(request).value(req)


@app.get("/forecast")
def forecast(request: Request, suburb: str = Query(min_length=1),
             horizon: int = Query(default=6, ge=1, le=24)) -> dict:
    try:
        return service(request).forecast(suburb, horizon)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc


@app.get("/quality")
def quality(request: Request) -> dict:
    svc = service(request)
    if svc.quality is None:
        raise ModelNotLoadedError(svc.load_error or "no quality report available")
    return svc.quality


@app.get("/health")
def health(request: Request) -> dict:
    svc = service(request)
    return {"status": "ok" if svc.model_loaded else "degraded", "model_version": svc.model_version,
            "model_loaded": svc.model_loaded, "data_as_of": svc.data_as_of, "load_error": svc.load_error}


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.post("/admin/reload")
def reload(request: Request) -> dict:
    """Re-read the @production alias — used after a promotion or rollback."""
    svc = service(request)
    svc.reload()
    return {"model_version": svc.model_version, "model_loaded": svc.model_loaded}
