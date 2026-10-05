import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    DriftReport,
    detect_feature_drift,
    detect_performance_drift,
    population_stability_index,
    should_retrain,
)

rng = np.random.default_rng(0)


def test_psi_near_zero_for_identical_and_large_for_shifted():
    base = rng.normal(0, 1, 5000)
    assert population_stability_index(base, rng.normal(0, 1, 5000)) < 0.02
    assert population_stability_index(base, rng.normal(2, 1, 5000)) > 1.0


def test_detect_feature_drift_flags_only_shifted_feature():
    train = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    live = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(3, 1, 3000)})
    for method in ["psi", "ks"]:
        reports = {r.feature: r for r in detect_feature_drift(train, live, ["a", "b"], method=method)}
        assert not reports["a"].drifted and reports["b"].drifted


def _perf(drop: float = 0.0, age: int | None = 10) -> dict:
    return {"ppe10_drop_points": drop, "model_age_days": age}


def _drift(n_drifted: int, n: int = 10) -> list[DriftReport]:
    return [DriftReport(feature=f"f{i}", method="psi", statistic=0.5 if i < n_drifted else 0.01,
                        drifted=i < n_drifted) for i in range(n)]


def test_should_retrain_each_rule_independently():
    assert should_retrain(_drift(0), _perf())[0] is False
    fire, reason = should_retrain(_drift(4), _perf())
    assert fire and "drifted" in reason
    fire, reason = should_retrain(_drift(0), _perf(drop=6.0))
    assert fire and "PPE10" in reason
    fire, reason = should_retrain(_drift(0), _perf(age=120))
    assert fire and "days old" in reason
    assert should_retrain(_drift(3), _perf(drop=5.0, age=90))[0] is False  # boundaries do not fire


def test_performance_drift_detects_degradation():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    y = np.full(60, 1_000_000.0)
    good = pd.DataFrame({"sale_date": dates, "y_true": y, "y_pred": y * 1.05})
    bad = pd.DataFrame({"sale_date": dates, "y_true": y, "y_pred": y * 1.25})
    train_metrics = {"ppe10": 0.9, "mae": 50_000.0}
    assert not detect_performance_drift(good, 30, train_metrics)["degraded"]
    res = detect_performance_drift(bad, 30, train_metrics, trained_at="2024-01-01",
                                   now=pd.Timestamp("2024-06-01", tz="UTC"))
    assert res["degraded"] and res["ppe10_drop_points"] == pytest.approx(90.0) and res["model_age_days"] == 152
