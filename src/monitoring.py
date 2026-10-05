"""Drift monitoring and the retraining trigger — the "you build it; you run it" part.

Three independent signals decide when to retrain: input drift (PSI), output
performance decay (PPE10 on recently settled sales), and model age.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

PSI_DRIFT_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.01
DRIFTED_FEATURE_SHARE = 0.30
PPE10_DROP_POINTS = 5.0
MAX_MODEL_AGE_DAYS = 90


class DriftReport(BaseModel):
    feature: str
    method: str  # "psi" | "ks"
    statistic: float
    drifted: bool


def population_stability_index(expected: Any, actual: Any, bins: int = 10) -> float:
    """PSI = sum((a - e) * ln(a / e)) over quantile bins of ``expected``.

    Bins are deciles of the training (expected) distribution; empty bins are
    floored at 1e-4 to keep the log finite. Missing values are dropped.

    Common industry convention (from credit-risk scorecards, not a law):
    < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
    """
    e = pd.Series(expected, dtype=float).dropna().to_numpy()
    a = pd.Series(actual, dtype=float).dropna().to_numpy()
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant feature
        return 0.0 if np.allclose(a, e[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, edges)[0] / len(e)
    a_pct = np.histogram(a, edges)[0] / len(a)
    e_pct, a_pct = np.clip(e_pct, 1e-4, None), np.clip(a_pct, 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def detect_feature_drift(train_df: pd.DataFrame, live_df: pd.DataFrame, features: list[str],
                         method: str = "psi") -> list[DriftReport]:
    """Per-feature drift between training and live inputs.

    ``psi``: drifted when PSI > 0.25. ``ks``: two-sample Kolmogorov-Smirnov,
    drifted when p < 0.01 (statistic reported is the KS D).
    """
    reports = []
    for f in features:
        if f not in train_df or f not in live_df:
            continue
        if method == "ks":
            e, a = train_df[f].dropna().astype(float), live_df[f].dropna().astype(float)
            if len(e) < 2 or len(a) < 2:
                continue
            res = stats.ks_2samp(e, a)
            reports.append(DriftReport(feature=f, method="ks", statistic=float(res.statistic),
                                       drifted=bool(res.pvalue < KS_PVALUE_THRESHOLD)))
        else:
            psi = population_stability_index(train_df[f], live_df[f])
            if np.isnan(psi):
                continue
            reports.append(DriftReport(feature=f, method="psi", statistic=psi,
                                       drifted=bool(psi > PSI_DRIFT_THRESHOLD)))
    return reports


def detect_performance_drift(recent_preds: pd.DataFrame, window_days: int = 30,
                             training_metrics: dict[str, float] | None = None,
                             trained_at: str | None = None,
                             now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Compare PPE10/MAE on recently *settled* sales against training-time values.

    ``recent_preds`` needs ``sale_date``, ``y_true`` and ``y_pred``. Only the last
    ``window_days`` (relative to the newest sale) are scored. ``degraded`` is
    True when PPE10 dropped more than 5 points or MAE rose more than 20%.
    """
    training_metrics = training_metrics or {}
    d = pd.to_datetime(recent_preds["sale_date"])
    recent = recent_preds[d >= d.max() - pd.Timedelta(days=window_days)]
    ape = (recent["y_pred"] - recent["y_true"]).abs() / recent["y_true"].abs()
    ppe10 = float((ape <= 0.10).mean()) if len(recent) else float("nan")
    mae = float((recent["y_pred"] - recent["y_true"]).abs().mean()) if len(recent) else float("nan")
    base_ppe10, base_mae = training_metrics.get("ppe10"), training_metrics.get("mae")
    drop = (base_ppe10 - ppe10) * 100 if base_ppe10 is not None else 0.0
    mae_rise = (mae / base_mae - 1) if base_mae else 0.0
    age = None
    if trained_at:
        utc = lambda t: t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")  # noqa: E731
        now = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        age = int((now - utc(pd.Timestamp(trained_at))).days)
    return {
        "window_days": window_days,
        "n": int(len(recent)),
        "ppe10": ppe10,
        "mae": mae,
        "ppe10_train": base_ppe10,
        "mae_train": base_mae,
        "ppe10_drop_points": float(drop),
        "mae_rise_pct": float(mae_rise * 100),
        "model_age_days": age,
        "degraded": bool(drop > PPE10_DROP_POINTS or mae_rise > 0.20),
    }


def should_retrain(drift: list[DriftReport], perf: dict[str, Any],
                   max_age_days: int = MAX_MODEL_AGE_DAYS) -> tuple[bool, str]:
    """Retrain if >30% of features drifted, OR PPE10 fell >5 points, OR model is >90 days old."""
    reasons = []
    if drift:
        share = sum(r.drifted for r in drift) / len(drift)
        if share > DRIFTED_FEATURE_SHARE:
            names = ", ".join(r.feature for r in drift if r.drifted)
            reasons.append(f"{share:.0%} of features drifted (> {DRIFTED_FEATURE_SHARE:.0%}): {names}")
    drop = perf.get("ppe10_drop_points") or 0.0
    if drop > PPE10_DROP_POINTS:
        reasons.append(f"PPE10 dropped {drop:.1f} points (> {PPE10_DROP_POINTS:.0f})")
    age = perf.get("model_age_days")
    if age is not None and age > max_age_days:
        reasons.append(f"model is {age} days old (> {max_age_days})")
    if reasons:
        return True, "Retrain: " + "; ".join(reasons)
    return False, "No retrain needed: drift, performance and model age are within limits."
