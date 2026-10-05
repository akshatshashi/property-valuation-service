"""Integration: train.py on the committed fixture, the quality gate, and the API."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_sales.csv"
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402


def _args(tmp: Path, data: Path) -> list[str]:
    return ["--data", str(data), "--mlflow-uri", str(tmp / "mlruns"), "--reports-dir", str(tmp / "reports"),
            "--quality-report", str(tmp / "DQ.md"), "--algos", "lightgbm", "--fail-if-not-better-than-baseline"]


def test_quality_gate_blocks_corrupted_input(tmp_path):
    df = pd.read_csv(FIXTURE)
    df.loc[df.sample(frac=0.6, random_state=0).index, "price"] = -1
    df[["bathrooms", "parking", "land_size_sqm", "building_area_sqm", "year_built", "latitude", "longitude"]] = None
    bad = tmp_path / "corrupted.csv"
    df.to_csv(bad, index=False)
    assert train.main(_args(tmp_path, bad)) == train.EXIT_QUALITY_GATE
    assert (tmp_path / "DQ.md").exists()  # report is still written so the failure is explainable
    assert not (tmp_path / "reports" / "training_summary.json").exists()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    assert train.main(_args(tmp, FIXTURE)) == 0
    return tmp


def test_train_registers_model_that_beats_baseline(trained):
    import json
    summary = json.loads((trained / "reports" / "training_summary.json").read_text())
    assert summary["ppe10"] > summary["baseline_ppe10"]
    assert summary["version"] == "1"


def test_api_end_to_end(trained, monkeypatch):
    from fastapi.testclient import TestClient

    from src.config import settings
    monkeypatch.setattr(settings, "mlflow_uri", str(trained / "mlruns"))
    monkeypatch.chdir(trained)
    import app as app_module
    with TestClient(app_module.app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] and health["model_version"] == "property-valuation/v1"
        req = {"suburb": "Richmond", "postcode": "3121", "property_type": "house", "bedrooms": 3,
               "bathrooms": 2, "parking": 1, "land_size_sqm": 300, "building_area_sqm": 150, "year_built": 1950}
        r = client.post("/value", json=req)
        assert r.status_code == 200
        body = r.json()
        assert body["lower"] <= body["estimate"] <= body["upper"]
        assert len(body["top_factors"]) == 5 and body["confidence_label"] in {"high", "medium", "low"}
        assert client.post("/value", json={**req, "bedrooms": -2}).status_code == 422
        fc = client.get("/forecast", params={"suburb": "Richmond", "horizon": 4}).json()
        assert len(fc["forecast"]) == 4
        assert client.get("/forecast", params={"suburb": "Nowhere"}).status_code == 404
        assert client.get("/quality").json()["overall_score"] > 0.7
        assert client.get("/metrics").json()["requests_by_path"]["/value"] == 2
    assert os.path.exists(trained / "logs" / "predictions.jsonl")


def test_api_returns_503_without_model(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.config import settings
    monkeypatch.setattr(settings, "mlflow_uri", str(tmp_path / "empty"))
    import app as app_module
    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        req = {"suburb": "Kew", "postcode": "3101", "property_type": "unit", "bedrooms": 2, "bathrooms": 1}
        assert client.post("/value", json=req).status_code == 503
