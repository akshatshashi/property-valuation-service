import numpy as np
import pandas as pd

from src.models.baseline import SuburbMedianBaseline
from src.models.evaluate import (
    compare_models,
    evaluate_intervals,
    evaluate_valuation,
    resolve_cutoff,
    temporal_split,
)


def test_ppe10_hand_computed():
    y_true = [100, 100, 100, 100, 100]
    y_pred = [105, 110, 89, 120, 100]  # errors 5%, 10%, 11%, 20%, 0% -> 3 of 5 within 10%
    m = evaluate_valuation(y_true, y_pred)
    assert m["ppe10"] == 0.6
    assert m["mae"] == (5 + 10 + 11 + 20 + 0) / 5
    assert np.isclose(m["mdape"], 0.10)


def test_temporal_split_never_leaks_future():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sale_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, 500), "D")})
    train, test = temporal_split(df, "2021-06-01")
    assert len(train) + len(test) == len(df)
    assert train["sale_date"].max() < test["sale_date"].min()


def test_resolve_cutoff_falls_back_when_out_of_range():
    df = pd.DataFrame({"sale_date": pd.date_range("2016-01-01", periods=100, freq="W")})
    assert resolve_cutoff(df, "2023-01-01") != "2023-01-01"
    assert resolve_cutoff(df, "2017-06-01") == "2017-06-01"


def test_interval_calibration_reports_miscalibration():
    y = np.arange(100, 110)
    lower, upper = y - 1, y + 1
    upper[0] = y[0] - 0.5  # 9 of 10 covered -> exactly 90%
    res = evaluate_intervals(y, lower, upper, nominal=0.9)
    assert res["coverage"] == 0.9 and not res["miscalibrated"]
    assert evaluate_intervals(y, y - 1, y + 1, nominal=0.9)["miscalibrated"]  # over-wide is miscalibrated too
    res = evaluate_intervals(y, y + 1, y + 2, nominal=0.9)
    assert res["coverage"] == 0.0 and res["miscalibrated"]


def test_compare_models_leaderboard_sorted():
    X = pd.DataFrame({"suburb": ["A", "B"] * 10})
    y = np.where(X["suburb"] == "A", 100.0, 200.0)
    board = compare_models({"baseline": SuburbMedianBaseline()}, X, y, X, y)
    assert board.loc[0, "model"] == "baseline" and board.loc[0, "ppe10"] == 1.0
