"""Honest, temporal evaluation of valuation models.

Random train/test splits must not be used for property data. A random split
puts sales from next month in the training set and sales from last month in
the test set, so the model is graded on its ability to interpolate a market it
has already seen — including future price movements. That leaks the future
into the past and overstates accuracy. Every split here is by sale date:
train strictly before the cutoff, test on or after it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, cutoff_date: str, date_col: str = "sale_date"
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = sales strictly before ``cutoff_date``; test = on or after it."""
    cutoff = pd.Timestamp(cutoff_date)
    dates = pd.to_datetime(df[date_col])
    return df[dates < cutoff].copy(), df[dates >= cutoff].copy()


def resolve_cutoff(df: pd.DataFrame, cutoff_date: str | None, test_fraction: float = 0.2,
                   date_col: str = "sale_date") -> str:
    """Use ``cutoff_date`` if it leaves data on both sides, otherwise the date
    that puts the most recent ``test_fraction`` of sales in the test set."""
    dates = pd.to_datetime(df[date_col])
    if cutoff_date is not None:
        c = pd.Timestamp(cutoff_date)
        share = float((dates >= c).mean())
        if 0.05 <= share <= 0.5:
            return c.strftime("%Y-%m-%d")
    return dates.quantile(1 - test_fraction).normalize().strftime("%Y-%m-%d")


def evaluate_valuation(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Point-accuracy metrics.

    PPE10 — the share of estimates within 10% of the actual sale price — is the
    industry-standard accuracy measure for automated valuation models (AVMs).
    """
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = yp - yt
    ape = np.abs(err) / np.abs(yt)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(ape)),
        "mdape": float(np.median(ape)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "ppe10": float(np.mean(ape <= 0.10)),
        "ppe20": float(np.mean(ape <= 0.20)),
    }


def evaluate_intervals(y_true: Any, lower: Any, upper: Any, nominal: float = 0.9) -> dict[str, float | bool]:
    """Empirical coverage and width of prediction intervals.

    A 90% interval that only covers 60% of actuals is miscalibrated and is
    reported as such via ``miscalibrated`` (more than 5 points off nominal).
    """
    yt, lo, hi = (np.asarray(a, dtype=float) for a in (y_true, lower, upper))
    coverage = float(np.mean((yt >= lo) & (yt <= hi)))
    width = hi - lo
    return {
        "coverage": coverage,
        "nominal": nominal,
        "coverage_gap": coverage - nominal,
        "miscalibrated": bool(abs(coverage - nominal) > 0.05),
        "mean_width": float(np.mean(width)),
        "median_relative_width": float(np.median(width / yt)),
    }


def compare_models(models: dict[str, Any], X_train: pd.DataFrame, y_train: Any,
                   X_test: pd.DataFrame, y_test: Any) -> pd.DataFrame:
    """Fit each model on train, score on test; leaderboard sorted by PPE10 descending."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append({"model": name, **evaluate_valuation(y_test, model.predict(X_test))})
    board = pd.DataFrame(rows).sort_values("ppe10", ascending=False).reset_index(drop=True)
    return board
