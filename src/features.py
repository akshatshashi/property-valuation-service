"""Temporal-safe feature engineering on the canonical frame.

The same functions build features for training, evaluation and live serving,
so there is exactly one definition of every feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROPERTY_TYPES = ["house", "unit", "townhouse"]
EPOCH = pd.Timestamp("2010-01-01")
SUBURB_WINDOW_DAYS = 730
MARKET_WINDOW_DAYS = 90

FEATURE_COLUMNS: list[str] = [
    "bedrooms",
    "bathrooms",
    "parking",
    "land_size_sqm",
    "building_area_sqm",
    "latitude",
    "longitude",
    "postcode_num",
    "rooms_total",
    "land_to_building",
    "property_age",
    "has_parking",
    *[f"pt_{t}" for t in PROPERTY_TYPES],
    "sale_year",
    "sale_month",
    "sale_quarter",
    "months_since_start",
    "suburb_median_price",
    "suburb_sales_count",
    "suburb_median_ppsqm",
    "market_median_price",
]


def _f(s: pd.Series) -> pd.Series:
    """Nullable/object column -> float64 with NaN, which every model accepts."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-property features that need no other rows: no leakage risk."""
    out = df.copy()
    price = _f(out["price"]) if "price" in out else pd.Series(np.nan, index=out.index)
    out["log_price"] = np.log(price.where(price > 0))
    beds, baths = _f(out["bedrooms"]), _f(out["bathrooms"])
    out["rooms_total"] = beds.fillna(0) + baths.fillna(0)
    out.loc[beds.isna() & baths.isna(), "rooms_total"] = np.nan
    land, building = _f(out["land_size_sqm"]), _f(out["building_area_sqm"])
    out["land_to_building"] = land.where(land > 0) / building.where(building > 0)
    age = pd.to_datetime(out["sale_date"]).dt.year - _f(out["year_built"])
    out["property_age"] = age.where(age >= 0)
    parking = _f(out["parking"])
    out["has_parking"] = (parking > 0).astype(float).where(parking.notna())
    ptype = out["property_type"].astype("string").str.lower()
    for t in PROPERTY_TYPES:
        out[f"pt_{t}"] = (ptype == t).fillna(False).astype(float)
    out["postcode_num"] = _f(out["postcode"])
    for col in ["bedrooms", "bathrooms", "parking", "land_size_sqm", "building_area_sqm",
                "latitude", "longitude", "year_built"]:
        out[col] = _f(out[col])
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. ``months_since_start`` counts from a fixed EPOCH so the
    same value is produced at training and serving time without stored state."""
    out = df.copy()
    d = pd.to_datetime(out["sale_date"])
    out["sale_year"] = d.dt.year.astype(float)
    out["sale_month"] = d.dt.month.astype(float)
    out["sale_quarter"] = d.dt.quarter.astype(float)
    out["months_since_start"] = ((d.dt.year - EPOCH.year) * 12 + (d.dt.month - EPOCH.month)).astype(float)
    return out


def _backward_median(
    query_dates: np.ndarray, ref_dates: np.ndarray, ref_values: np.ndarray, window: np.timedelta64
) -> tuple[np.ndarray, np.ndarray]:
    """Median and count of ``ref_values`` with ``q - window <= ref_date < q`` for each query.

    ``ref_dates`` must be sorted ascending. ``side="left"`` on the upper bound
    is what excludes same-day and future sales.
    """
    end = np.searchsorted(ref_dates, query_dates, side="left")
    start = np.searchsorted(ref_dates, query_dates - window, side="left")
    med = np.full(len(query_dates), np.nan)
    for i, (s, e) in enumerate(zip(start, end, strict=True)):
        if e > s:
            med[i] = np.median(ref_values[s:e])
    return med, (end - start).astype(float)


def add_suburb_features(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    window_days: int = SUBURB_WINDOW_DAYS,
    market_window_days: int = MARKET_WINDOW_DAYS,
) -> pd.DataFrame:
    """Suburb and market aggregates computed strictly from the past.

    The leakage trap: ``df.groupby("suburb")["price"].median()`` over the whole
    dataset lets every row "see" sales that happened after it — including its
    own price. Offline metrics then look excellent and collapse in production,
    where the future is genuinely unknown.

    How this avoids it: for each row with sale date ``d`` in suburb ``s``, only
    reference sales in ``s`` with ``d - window <= sale_date < d`` are used. This
    is a backward-looking as-of join implemented with ``searchsorted`` over
    date-sorted reference arrays; the strict ``<`` excludes same-day sales, so a
    row never contributes to its own features. Rows with no qualifying history
    get NaN (tree models route missing values natively).

    Features:
      - ``suburb_median_price``: median price in the suburb over the trailing window
      - ``suburb_sales_count``: number of those sales (a confidence signal)
      - ``suburb_median_ppsqm``: median price per sqm of building area, same window
      - ``market_median_price``: city-wide median over the last ``market_window_days``

    Coordinates are also filled from the suburb centroid when missing; location
    is not derived from price, so using all reference rows there is not leakage.
    """
    out = df.copy()
    out["sale_date"] = pd.to_datetime(out["sale_date"])
    ref = reference_df[["suburb", "sale_date", "price", "building_area_sqm", "latitude", "longitude"]].copy()
    ref["sale_date"] = pd.to_datetime(ref["sale_date"])
    ref["price"] = _f(ref["price"])
    ref = ref[ref["price"] > 0].sort_values("sale_date", kind="stable")
    ref["ppsqm"] = ref["price"] / _f(ref["building_area_sqm"]).where(lambda b: b > 0)

    window = np.timedelta64(window_days, "D")
    for col in ["suburb_median_price", "suburb_sales_count", "suburb_median_ppsqm"]:
        out[col] = np.nan

    ref_groups = {k: g for k, g in ref.groupby("suburb", sort=False)}
    for suburb, rows in out.groupby("suburb", sort=False):
        g = ref_groups.get(suburb)
        if g is None:
            out.loc[rows.index, "suburb_sales_count"] = 0.0
            continue
        q = rows["sale_date"].values
        med, cnt = _backward_median(q, g["sale_date"].values, g["price"].values, window)
        out.loc[rows.index, "suburb_median_price"] = med
        out.loc[rows.index, "suburb_sales_count"] = cnt
        gp = g[g["ppsqm"].notna()]
        ppsqm, _ = _backward_median(q, gp["sale_date"].values, gp["ppsqm"].values, window)
        out.loc[rows.index, "suburb_median_ppsqm"] = ppsqm

    market, _ = _backward_median(
        out["sale_date"].values, ref["sale_date"].values, ref["price"].values,
        np.timedelta64(market_window_days, "D"),
    )
    out["market_median_price"] = market

    centroids = reference_df.groupby("suburb")[["latitude", "longitude"]].median()
    for col in ["latitude", "longitude"]:
        out[col] = _f(out[col]).fillna(out["suburb"].map(centroids[col]).astype(float))
    return out


def build_feature_matrix(df: pd.DataFrame, reference_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Full feature pipeline -> (X with ``FEATURE_COLUMNS`` in fixed order, y = price)."""
    feats = add_suburb_features(add_temporal_features(add_property_features(df)), reference_df)
    X = feats.reindex(columns=FEATURE_COLUMNS).astype(float)
    y = _f(feats["price"]) if "price" in feats else pd.Series(np.nan, index=feats.index)
    return X, y
