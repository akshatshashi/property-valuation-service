"""Suburb-median baseline: the yardstick every other model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SuburbMedianBaseline:
    """Predicts the training-set median price of the property's suburb.

    Unseen suburbs (or rows with no suburb) fall back to the global median.
    ``X`` must carry a ``suburb`` column; all other columns are ignored.
    """

    def __init__(self, group_col: str = "suburb") -> None:
        self.group_col = group_col
        self.medians_: dict[str, float] = {}
        self.global_median_: float = float("nan")

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> SuburbMedianBaseline:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.medians_ = y.groupby(X[self.group_col]).median().to_dict()
        self.global_median_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = X[self.group_col].map(self.medians_).astype(float)
        return preds.fillna(self.global_median_).to_numpy()
