"""Suburb-level median price index forecasting.

This forecasts a suburb's monthly median sale price (an index), not individual
properties. Naive (last value carried forward) is a required method: property
indexes are strongly trending and autocorrelated, so naive is a genuinely hard
baseline. Backtests report MASE relative to naive so it is obvious whether a
fancier model earns its complexity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

METHODS = ("naive", "ets", "sarima")
ALL_SUBURBS = "ALL"


def build_price_index(df: pd.DataFrame, suburb: str, freq: str = "MS", min_sales: int = 1) -> pd.Series:
    """Monthly median price for ``suburb`` (or ``"ALL"`` for the whole market).

    Gap handling (explicit choice): months with fewer than ``min_sales`` sales
    are set missing, then interior gaps are filled by linear interpolation in
    time. Leading/trailing gaps are dropped, never extrapolated — inventing an
    index level the data never showed would bias the forecast origin.
    Interpolation is preferred over forward-fill because forward-fill creates
    artificial flat steps that make naive look better than it is.
    """
    sub = df if suburb == ALL_SUBURBS else df[df["suburb"] == suburb]
    if sub.empty:
        raise KeyError(f"no sales for suburb {suburb!r}")
    s = sub.set_index(pd.to_datetime(sub["sale_date"]))["price"].astype(float)
    grouped = s.resample(freq)
    med, cnt = grouped.median(), grouped.count()
    med[cnt < min_sales] = np.nan
    full = pd.date_range(med.index.min(), med.index.max(), freq=freq)
    med = med.reindex(full)
    med = med.loc[med.first_valid_index(): med.last_valid_index()]
    out = med.interpolate(method="time")
    out.name = f"median_price_{suburb}"
    return out


class IndexForecaster:
    """``naive`` | ``ets`` | ``sarima`` behind one interface with prediction intervals."""

    def __init__(self, method: str = "ets", alpha: float = 0.1) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.alpha = alpha
        self.series_: pd.Series | None = None
        self.result_ = None

    def fit(self, series: pd.Series) -> IndexForecaster:
        self.series_ = series.astype(float)
        self.result_ = None
        y = self.series_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.method == "ets" and len(y) >= 6:
                self.result_ = ETSModel(y, error="add", trend="add", damped_trend=True).fit(disp=False)
            elif self.method == "sarima" and len(y) >= 8:
                seasonal = (1, 0, 0, 12) if len(y) >= 36 else (0, 0, 0, 0)
                self.result_ = SARIMAX(y, order=(1, 1, 1), seasonal_order=seasonal,
                                       enforce_stationarity=False, enforce_invertibility=False
                                       ).fit(disp=False)
        return self

    def _future_index(self, horizon: int) -> pd.DatetimeIndex:
        assert self.series_ is not None
        freq = self.series_.index.freq or pd.infer_freq(self.series_.index) or "MS"
        return pd.date_range(self.series_.index[-1], periods=horizon + 1, freq=freq)[1:]

    def predict(self, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return ``(point, lower, upper)`` for the next ``horizon`` periods.

        Methods fall back to naive when the series is too short to fit them.
        """
        if self.series_ is None:
            raise RuntimeError("IndexForecaster is not fitted")
        idx = self._future_index(horizon)
        y = self.series_
        z = stats.norm.ppf(1 - self.alpha / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.result_ is not None and self.method == "ets":
                frame = self.result_.get_prediction(start=len(y), end=len(y) + horizon - 1
                                                    ).summary_frame(alpha=self.alpha)
                point, lo, hi = frame["mean"].values, frame["pi_lower"].values, frame["pi_upper"].values
            elif self.result_ is not None and self.method == "sarima":
                fc = self.result_.get_forecast(horizon)
                ci = np.asarray(fc.conf_int(alpha=self.alpha))
                point, lo, hi = np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1]
            else:  # naive: random walk, sigma from first differences, widening with sqrt(h)
                diffs = y.diff().dropna()
                sigma = float(diffs.std()) if len(diffs) > 1 else float(y.std() or 0.0)
                h = np.arange(1, horizon + 1)
                point = np.full(horizon, y.iloc[-1])
                lo, hi = point - z * sigma * np.sqrt(h), point + z * sigma * np.sqrt(h)
        mk = lambda v: pd.Series(np.asarray(v, dtype=float), index=idx)  # noqa: E731
        return mk(point), mk(lo), mk(hi)


def backtest(series: pd.Series, method: str, horizon: int, n_folds: int = 5) -> dict[str, float]:
    """Rolling-origin backtest.

    Fold ``k`` trains on everything up to origin ``T - horizon - (n_folds-1-k)``
    and forecasts the next ``horizon`` points. Metrics are pooled over folds.
    MASE here = model MAE / naive MAE on the same folds, so MASE < 1 means the
    method beat naive (naive itself scores exactly 1.0).
    """
    n = len(series)
    first_origin = n - horizon - (n_folds - 1)
    if first_origin < 6:
        raise ValueError(f"series of length {n} too short for horizon={horizon}, n_folds={n_folds}")
    errs, naive_errs, actuals, covered = [], [], [], []
    for k in range(n_folds):
        origin = first_origin + k
        train, test = series.iloc[:origin], series.iloc[origin:origin + horizon]
        point, lo, hi = IndexForecaster(method).fit(train).predict(len(test))
        naive_point, _, _ = IndexForecaster("naive").fit(train).predict(len(test))
        errs.append(point.values - test.values)
        naive_errs.append(naive_point.values - test.values)
        actuals.append(test.values)
        covered.append((test.values >= lo.values) & (test.values <= hi.values))
    e, ne, a = np.concatenate(errs), np.concatenate(naive_errs), np.concatenate(actuals)
    naive_mae = float(np.mean(np.abs(ne)))
    mae = float(np.mean(np.abs(e)))
    return {
        "method": method,
        "horizon": horizon,
        "n_folds": n_folds,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mape": float(np.mean(np.abs(e) / np.abs(a))),
        "mase": mae / naive_mae if naive_mae > 0 else float("nan"),
        "interval_coverage": float(np.mean(np.concatenate(covered))),
    }
