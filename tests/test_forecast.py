import numpy as np
import pandas as pd
import pytest

from src.models.forecast import METHODS, IndexForecaster, backtest, build_price_index


@pytest.fixture
def trending() -> pd.Series:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    return pd.Series(800_000 + 5_000 * np.arange(48) + rng.normal(0, 8_000, 48), index=idx)


@pytest.mark.parametrize("method", METHODS)
def test_horizon_length_and_interval_order(method, trending):
    point, lower, upper = IndexForecaster(method).fit(trending).predict(6)
    assert len(point) == len(lower) == len(upper) == 6
    assert point.index[0] == pd.Timestamp("2022-01-01")
    assert (lower <= point).all() and (point <= upper).all()


@pytest.mark.parametrize("method", METHODS)
def test_backtest_returns_all_metrics(method, trending):
    res = backtest(trending, method, horizon=3, n_folds=5)
    for key in ["mae", "rmse", "mape", "mase", "interval_coverage"]:
        assert key in res and np.isfinite(res[key])
    if method == "naive":
        assert res["mase"] == pytest.approx(1.0)


def test_ets_beats_naive_on_clean_trend(trending):
    assert backtest(trending, "ets", horizon=3)["mase"] < 1.0


def test_build_price_index_fills_interior_gaps():
    sales = pd.DataFrame({
        "sale_date": pd.to_datetime(["2020-01-10", "2020-01-20", "2020-03-05", "2020-04-01"]),
        "price": [100.0, 200.0, 300.0, 400.0],
        "suburb": ["Kew"] * 4,
    })
    s = build_price_index(sales, "Kew")
    assert list(s.index.strftime("%Y-%m")) == ["2020-01", "2020-02", "2020-03", "2020-04"]
    assert s.iloc[0] == 150.0 and 150.0 < s.iloc[1] < 300.0  # February interpolated
    with pytest.raises(KeyError):
        build_price_index(sales, "Atlantis")
