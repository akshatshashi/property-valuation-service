"""Property-level valuation model with prediction intervals and SHAP explanations.

Every automated valuation must ship a confidence range, not a bare number: a
$900k estimate with a +/-$50k range and one with a +/-$400k range lead to
very different decisions. ``predict_interval`` therefore uses genuine quantile
regression, optionally conformalised so the stated coverage is actually met.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ALGOS = ("lightgbm", "xgboost", "rf")

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": dict(n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1),
    "xgboost": dict(n_estimators=600, learning_rate=0.04, max_depth=7, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist"),
    "rf": dict(n_estimators=200, min_samples_leaf=3, max_features=0.5, max_depth=24, n_jobs=-1),
}
QUANTILE_PARAMS: dict[str, Any] = dict(n_estimators=400, learning_rate=0.04, num_leaves=31,
                                       min_child_samples=30, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, verbose=-1)


class ValuationModel:
    """One interface over LightGBM, XGBoost and Random Forest.

    - Trains on log(price) and returns dollars (property prices are right-skewed;
      errors are multiplicative, so log space matches how AVMs are judged).
    - ``predict_interval`` uses LightGBM quantile regression at ``alpha/2`` and
      ``1 - alpha/2`` for every algo, so intervals are comparable across models.
      The point estimate comes from the main model; bounds are widened if needed
      so ``lower <= point <= upper`` always holds.
    - With ``conformal=True`` (default) the quantile models are fit on the
      earliest 80% of training rows and the latest 20% is used to compute a
      split-conformal correction (CQR, Romano et al. 2019). Raw quantile
      models typically under-cover on future data; the correction fixes the
      stated coverage on the calibration slice. Rows must arrive time-ordered.
    """

    def __init__(self, algo: str = "lightgbm", interval_alpha: float = 0.1, conformal: bool = True,
                 random_state: int = 42, **params: Any) -> None:
        if algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, got {algo!r}")
        self.algo = algo
        self.interval_alpha = interval_alpha
        self.conformal = conformal
        self.random_state = random_state
        self.params = {**DEFAULT_PARAMS[algo], **params}
        self.feature_names_: list[str] = []
        self.model_: Any = None
        self.q_models_: dict[str, lgb.LGBMRegressor] = {}
        self.conformal_offset_: float = 0.0
        self._explainer: Any = None

    # ------------------------------------------------------------------ fitting
    def _make_model(self) -> Any:
        if self.algo == "lightgbm":
            return lgb.LGBMRegressor(random_state=self.random_state, **self.params)
        if self.algo == "xgboost":
            return xgb.XGBRegressor(random_state=self.random_state, **self.params)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("rf", RandomForestRegressor(random_state=self.random_state, **self.params)),
        ])

    def _quantile_model(self, q: float) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.random_state,
                                 **QUANTILE_PARAMS)

    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_names_).astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> ValuationModel:
        self.feature_names_ = list(X.select_dtypes("number").columns)
        Xf = self._X(X)
        y_log = np.log(np.asarray(y, dtype=float))
        self.model_ = self._make_model().fit(Xf, y_log)

        a = self.interval_alpha
        n_fit = int(len(Xf) * 0.8) if self.conformal and len(Xf) >= 50 else len(Xf)
        self.q_models_ = {
            "lower": self._quantile_model(a / 2).fit(Xf.iloc[:n_fit], y_log[:n_fit]),
            "upper": self._quantile_model(1 - a / 2).fit(Xf.iloc[:n_fit], y_log[:n_fit]),
        }
        self.conformal_offset_ = 0.0
        if n_fit < len(Xf):
            Xc, yc = Xf.iloc[n_fit:], y_log[n_fit:]
            lo, hi = self.q_models_["lower"].predict(Xc), self.q_models_["upper"].predict(Xc)
            scores = np.maximum(lo - yc, yc - hi)
            level = min(1.0, (1 - a) * (1 + 1 / len(yc)))
            self.conformal_offset_ = float(np.quantile(scores, level))
        self._explainer = None
        return self

    # --------------------------------------------------------------- inference
    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("ValuationModel is not fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.exp(np.asarray(self.model_.predict(self._X(X)), dtype=float))

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1, calibrated: bool = True
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lower, point, upper)`` in dollars for a ``1 - alpha`` interval.

        ``calibrated=False`` returns the raw quantile-regression bounds, useful
        for reporting how miscalibrated they were before the conformal step.
        """
        self._check_fitted()
        if not np.isclose(alpha, self.interval_alpha):
            raise ValueError(f"model was fitted for alpha={self.interval_alpha}; got alpha={alpha}")
        Xf = self._X(X)
        offset = self.conformal_offset_ if calibrated else 0.0
        point = np.asarray(self.model_.predict(Xf), dtype=float)  # xgboost returns float32
        lower = self.q_models_["lower"].predict(Xf) - offset
        upper = self.q_models_["upper"].predict(Xf) + offset
        lower, upper = np.minimum(lower, point), np.maximum(upper, point)
        return np.exp(lower), np.exp(point), np.exp(upper)

    def explain(self, X: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
        """SHAP values as a tidy frame: ``row, feature, value, shap_value``.

        SHAP values are in log-price units, so ``exp(shap_value) - 1`` is the
        approximate proportional effect of that feature on the estimate.
        """
        self._check_fitted()
        Xf = self._X(X).iloc[:max_rows]
        if self._explainer is None:
            tree_model = self.model_.named_steps["rf"] if self.algo == "rf" else self.model_
            self._explainer = shap.TreeExplainer(tree_model)
        X_in = (pd.DataFrame(self.model_.named_steps["impute"].transform(Xf), columns=Xf.columns,
                             index=Xf.index) if self.algo == "rf" else Xf)
        values = np.asarray(self._explainer.shap_values(X_in, check_additivity=False))
        tidy = pd.DataFrame(values, columns=Xf.columns, index=Xf.index).reset_index(names="row")
        tidy = tidy.melt(id_vars="row", var_name="feature", value_name="shap_value")
        tidy["value"] = Xf.reset_index(drop=True).melt()["value"].to_numpy()
        return tidy[["row", "feature", "value", "shap_value"]]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None  # rebuilt lazily; keeps pickles small and portable
        return state
