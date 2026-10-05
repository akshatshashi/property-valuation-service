"""Valuation service: loads the production model once and answers requests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import settings
from src.features import build_feature_matrix
from src.models.forecast import IndexForecaster, build_price_index
from src.registry import Registry

log = logging.getLogger("pvs.serving")

# confidence_label thresholds on relative interval width = (upper - lower) / estimate.
# Set from the temporal test set: the median 90% interval is ~0.61 of the estimate, so
# "high" (<= 0.45) is roughly the tightest 20% of valuations and "low" (> 0.80) the widest 20%.
HIGH_CONFIDENCE_MAX_WIDTH = 0.45
MEDIUM_CONFIDENCE_MAX_WIDTH = 0.80


class ModelNotLoadedError(RuntimeError):
    """Raised when no production model is available (the API maps it to 503)."""


class ValuationRequest(BaseModel):
    suburb: str = Field(min_length=1, examples=["Richmond"])
    postcode: str = Field(pattern=r"^\d{4}$", examples=["3121"])
    property_type: str = Field(pattern=r"^(house|unit|townhouse)$", examples=["house"])
    bedrooms: int = Field(ge=0, le=20, examples=[3])
    bathrooms: int = Field(ge=0, le=20, examples=[2])
    parking: int | None = Field(default=None, ge=0, le=20, examples=[1])
    land_size_sqm: float | None = Field(default=None, gt=0, le=100_000, examples=[350])
    building_area_sqm: float | None = Field(default=None, gt=0, le=10_000, examples=[150])
    year_built: int | None = Field(default=None, ge=1800, le=2100, examples=[1950])


class ValuationResponse(BaseModel):
    estimate: float
    lower: float
    upper: float
    confidence_label: str  # "high" | "medium" | "low"
    top_factors: list[dict]  # feature, contribution, from SHAP
    model_version: str
    data_as_of: str


def confidence_label(estimate: float, lower: float, upper: float) -> str:
    width = (upper - lower) / estimate
    if width <= HIGH_CONFIDENCE_MAX_WIDTH:
        return "high"
    if width <= MEDIUM_CONFIDENCE_MAX_WIDTH:
        return "medium"
    return "low"


class ValuationService:
    """Loads model, config and reference data from the registry at startup, not per request."""

    def __init__(self, registry: Registry | None, model_name: str = settings.model_name,
                 prediction_log: str = "logs/predictions.jsonl") -> None:
        self.registry = registry
        self.model_name = model_name
        self.model: Any = None
        self.model_version = "none"
        self.config: dict[str, Any] = {}
        self.reference: pd.DataFrame | None = None
        self.quality: dict[str, Any] | None = None
        self.load_error: str | None = None
        self.prediction_log = Path(prediction_log)
        self.reload()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def data_as_of(self) -> str:
        return str(self.config.get("data_as_of", "unknown"))

    def reload(self) -> None:
        """(Re)load whatever is aliased @production. Safe to call after a promotion/rollback."""
        if self.registry is None:
            self.load_error = "no registry configured"
            return
        try:
            mv = self.registry.production_version(self.model_name)
            self.model = self.registry.load_production(self.model_name)
            self.config = self.registry.load_production_json(self.model_name, "config")
            ref_dir = Path(self.registry.download_production_artifact(self.model_name, "reference"))
            self.reference = pd.read_parquet(next(ref_dir.glob("*.parquet")) if ref_dir.is_dir() else ref_dir)
            self.quality = self.registry.load_production_json(self.model_name, "quality")
            self.model_version = f"{self.model_name}/v{mv.version}"
            self.load_error = None
            log.info("loaded %s (data as of %s)", self.model_version, self.data_as_of)
        except Exception as exc:  # any failure leaves the service up but unready (503)
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning("no production model loaded: %s", self.load_error)

    def _require_model(self) -> None:
        if not self.model_loaded:
            raise ModelNotLoadedError(self.load_error or "no production model loaded")

    def value(self, req: ValuationRequest) -> ValuationResponse:
        self._require_model()
        assert self.reference is not None
        valuation_date = pd.Timestamp(self.data_as_of) + pd.Timedelta(days=1)
        row = pd.DataFrame([{**req.model_dump(), "sale_date": valuation_date, "price": np.nan,
                             "sale_id": "request", "state": "VIC", "latitude": np.nan,
                             "longitude": np.nan}])
        row["suburb"] = row["suburb"].str.strip().str.title()
        X, _ = build_feature_matrix(row, self.reference)
        lower, point, upper = self.model.predict_interval(X, alpha=self.config.get("interval_alpha", 0.1))
        shap_df = self.model.explain(X, max_rows=1)
        shap_df = shap_df.reindex(shap_df["shap_value"].abs().sort_values(ascending=False).index).head(5)
        top = [{"feature": r.feature,
                "value": None if pd.isna(r.value) else round(float(r.value), 2),
                "contribution": round(float(r.shap_value), 4),
                "effect_pct": round(float(np.expm1(r.shap_value) * 100), 1)}
               for r in shap_df.itertuples()]
        resp = ValuationResponse(
            estimate=round(float(point[0]), -3), lower=round(float(lower[0]), -3),
            upper=round(float(upper[0]), -3),
            confidence_label=confidence_label(float(point[0]), float(lower[0]), float(upper[0])),
            top_factors=top, model_version=self.model_version, data_as_of=self.data_as_of,
        )
        self._log_prediction(req, resp)
        return resp

    def _log_prediction(self, req: ValuationRequest, resp: ValuationResponse) -> None:
        """Append every prediction with its model_version for later drift analysis."""
        record = {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "model_version": resp.model_version,
                  "request": req.model_dump(), "estimate": resp.estimate, "lower": resp.lower,
                  "upper": resp.upper}
        try:
            self.prediction_log.parent.mkdir(parents=True, exist_ok=True)
            with self.prediction_log.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:  # logging must never fail a valuation
            log.warning("could not write prediction log: %s", exc)

    def forecast(self, suburb: str, horizon: int, method: str = "ets") -> dict[str, Any]:
        self._require_model()
        assert self.reference is not None
        name = suburb if suburb.upper() == "ALL" else suburb.strip().title()
        series = build_price_index(self.reference, "ALL" if name.upper() == "ALL" else name, min_sales=3)
        if len(series) < 6:
            method = "naive"  # too little history for a model; say so in the response
        point, lower, upper = IndexForecaster(method).fit(series).predict(horizon)
        fmt = lambda s: [{"month": i.strftime("%Y-%m"), "value": round(float(v), 0)}  # noqa: E731
                         for i, v in s.items()]
        return {
            "suburb": name, "method": method, "horizon": horizon, "interval": "90%",
            "history": fmt(series.tail(24)),
            "forecast": [{"month": i.strftime("%Y-%m"), "point": round(float(p), 0),
                          "lower": round(float(lo), 0), "upper": round(float(hi), 0)}
                         for i, p, lo, hi in zip(point.index, point, lower, upper, strict=True)],
            "model_version": self.model_version, "data_as_of": self.data_as_of,
        }
