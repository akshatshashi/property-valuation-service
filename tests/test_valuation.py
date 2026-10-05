import numpy as np
import pandas as pd
import pytest

from src.models.baseline import SuburbMedianBaseline
from src.models.valuation import ALGOS, ValuationModel


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({
        "area": rng.uniform(50, 300, n),
        "beds": rng.integers(1, 6, n).astype(float),
        "lat": rng.normal(-37.8, 0.05, n),
    })
    X.loc[::17, "area"] = np.nan  # models must cope with missing values
    y = 5000 * X["area"].fillna(150) * (1 + 0.05 * X["beds"]) * rng.lognormal(0, 0.1, n)
    return X, y


@pytest.mark.parametrize("algo", ALGOS)
def test_interface_and_shapes(algo, data):
    X, y = data
    model = ValuationModel(algo, n_estimators=50)
    assert model.fit(X, y) is model
    pred = model.predict(X)
    assert pred.shape == (len(X),) and np.all(pred > 0)
    lower, point, upper = model.predict_interval(X, alpha=0.1)
    assert lower.shape == point.shape == upper.shape == (len(X),)
    assert np.all(lower <= point) and np.all(point <= upper)
    raw_lo, _, raw_hi = model.predict_interval(X, alpha=0.1, calibrated=False)
    assert np.all(raw_lo <= raw_hi)


@pytest.mark.parametrize("algo", ALGOS)
def test_explain_returns_tidy_shap(algo, data):
    X, y = data
    model = ValuationModel(algo, n_estimators=30).fit(X, y)
    tidy = model.explain(X, max_rows=10)
    assert list(tidy.columns) == ["row", "feature", "value", "shap_value"]
    assert len(tidy) == 10 * X.shape[1]
    assert tidy["shap_value"].notna().all()


def test_unknown_algo_and_alpha_mismatch(data):
    X, y = data
    with pytest.raises(ValueError):
        ValuationModel("svm")
    model = ValuationModel("lightgbm", n_estimators=20).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_interval(X, alpha=0.2)


def test_baseline_suburb_median_with_global_fallback():
    X = pd.DataFrame({"suburb": ["A", "A", "A", "B"]})
    y = [100.0, 200.0, 300.0, 1000.0]
    model = SuburbMedianBaseline().fit(X, y)
    pred = model.predict(pd.DataFrame({"suburb": ["A", "B", "Unseen"]}))
    assert list(pred) == [200.0, 1000.0, 250.0]
