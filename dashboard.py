"""Streamlit dashboard: data quality and model performance side by side.

    streamlit run dashboard.py

Imports from src/ only — no logic is duplicated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import settings
from src.models.evaluate import evaluate_intervals
from src.registry import Registry
from src.serving import ValuationRequest, ValuationService

REPORTS = Path(__file__).parent / "reports"
st.set_page_config(page_title="Property Valuation Service", layout="wide")


@st.cache_resource
def get_service() -> ValuationService:
    return ValuationService(Registry(settings.mlflow_uri))


def read_json(name: str) -> dict | None:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


svc = get_service()
summary = read_json("training_summary.json") or {}
st.title("Property Valuation Service")
cols = st.columns(4)
cols[0].metric("Production model", svc.model_version)
cols[1].metric("PPE10 (test)", f"{summary.get('ppe10', float('nan')):.1%}",
               f"{(summary.get('ppe10', 0) - summary.get('baseline_ppe10', 0)) * 100:+.1f} pts vs baseline")
cols[2].metric("90% interval coverage", f"{summary.get('coverage', float('nan')):.1%}")
cols[3].metric("Data quality score", f"{summary.get('quality_score', float('nan')):.3f}")

tab_q, tab_m, tab_f, tab_try = st.tabs(["Data Quality", "Model Performance", "Forecast", "Try It"])

with tab_q:
    q = svc.quality or read_json("quality_report.json")
    if not q:
        st.warning("No quality report yet — run scripts/train.py")
    else:
        st.subheader(f"Overall score {q['overall_score']:.3f} on {q['n_rows']:,} rows")
        st.dataframe(pd.DataFrame(q["issues"]), width="stretch")
        left, right = st.columns(2)
        left.caption("Sales per month")
        left.bar_chart(pd.Series(q["coverage_by_month"], name="sales"))
        right.caption("Sales per suburb (top 30)")
        right.bar_chart(pd.Series(q["coverage_by_suburb"], name="sales").head(30))
        st.caption("Column completeness")
        prof = pd.DataFrame(q["profiles"]).set_index("column")
        st.bar_chart(prof["completeness"])

with tab_m:
    board_path = REPORTS / "leaderboard.csv"
    if board_path.exists():
        st.subheader("Leaderboard (temporal hold-out, sorted by PPE10)")
        st.dataframe(pd.read_csv(board_path), width="stretch")
    preds_path = REPORTS / "test_predictions.parquet"
    if preds_path.exists():
        p = pd.read_parquet(preds_path)
        st.subheader("Interval calibration by month")
        p["month"] = pd.to_datetime(p["sale_date"]).dt.strftime("%Y-%m")
        calib = p.groupby("month").apply(lambda g: pd.Series({
            "calibrated": evaluate_intervals(g.y_true, g.lower, g.upper)["coverage"],
            "raw quantile": evaluate_intervals(g.y_true, g.lower_raw, g.upper_raw)["coverage"],
            "nominal": 0.9}), include_groups=False)
        st.line_chart(calib)
        st.subheader("Accuracy by property type")
        p["within_10pct"] = (np.abs(p.y_pred - p.y_true) / p.y_true) <= 0.10
        st.bar_chart(p.groupby("property_type")["within_10pct"].mean())
    fc_path = REPORTS / "forecast_backtest.csv"
    if fc_path.exists():
        st.subheader("Forecast backtest (MASE < 1 beats naive)")
        st.dataframe(pd.read_csv(fc_path), width="stretch")

with tab_f:
    if svc.reference is None:
        st.warning("No production model loaded")
    else:
        suburbs = ["ALL", *svc.reference["suburb"].value_counts().head(100).index]
        c1, c2 = st.columns(2)
        suburb = c1.selectbox("Suburb", suburbs)
        horizon = c2.slider("Horizon (months)", 1, 12, 6)
        fc = svc.forecast(suburb, horizon)
        hist = pd.DataFrame(fc["history"]).set_index("month").rename(columns={"value": "history"})
        fut = pd.DataFrame(fc["forecast"]).set_index("month")
        st.line_chart(pd.concat([hist, fut], axis=1))
        st.caption(f"Method: {fc['method']} · 90% interval · data as of {fc['data_as_of']}")

with tab_try:
    if not svc.model_loaded:
        st.warning("No production model loaded")
    else:
        with st.form("value"):
            c1, c2, c3 = st.columns(3)
            suburb = c1.selectbox("Suburb", sorted(svc.reference["suburb"].dropna().unique()))
            postcode = c1.text_input("Postcode", "3121")
            ptype = c1.selectbox("Property type", ["house", "unit", "townhouse"])
            beds = c2.number_input("Bedrooms", 0, 20, 3)
            baths = c2.number_input("Bathrooms", 0, 20, 2)
            parking = c2.number_input("Parking", 0, 20, 1)
            land = c3.number_input("Land size (sqm)", 0.0, 100000.0, 300.0)
            building = c3.number_input("Building area (sqm)", 0.0, 10000.0, 150.0)
            year = c3.number_input("Year built", 1800, 2026, 1960)
            submitted = st.form_submit_button("Value it")
        if submitted:
            req = ValuationRequest(suburb=suburb, postcode=postcode, property_type=ptype, bedrooms=beds,
                                   bathrooms=baths, parking=parking, land_size_sqm=land or None,
                                   building_area_sqm=building or None, year_built=year)
            r = svc.value(req)
            st.metric("Estimate", f"${r.estimate:,.0f}", f"range ${r.lower:,.0f} – ${r.upper:,.0f}")
            st.write(f"Confidence: **{r.confidence_label}** · {r.model_version} · data as of {r.data_as_of}")
            st.dataframe(pd.DataFrame(r.top_factors), width="stretch")
