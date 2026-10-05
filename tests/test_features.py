import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, add_suburb_features, build_feature_matrix


def ordered_frame() -> pd.DataFrame:
    # one sale per day in one suburb; price == 1000 * day index, so any leak is visible
    n = 20
    return pd.DataFrame({
        "sale_id": [f"s{i}" for i in range(n)],
        "sale_date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "price": [1000.0 * (i + 1) for i in range(n)],
        "suburb": ["Kew"] * n, "postcode": ["3101"] * n, "state": ["VIC"] * n,
        "property_type": ["house"] * n, "bedrooms": [3] * n, "bathrooms": [1] * n,
        "parking": [1] * n, "land_size_sqm": [400.0] * n, "building_area_sqm": [100.0] * n,
        "latitude": [-37.8] * n, "longitude": [145.0] * n, "year_built": [1950] * n,
    })


def test_suburb_features_use_only_strictly_earlier_sales():
    df = ordered_frame()
    out = add_suburb_features(df, df)
    for i in range(len(df)):
        earlier = df["price"].iloc[:i]
        assert out["suburb_sales_count"].iloc[i] == len(earlier)
        if i == 0:
            assert np.isnan(out["suburb_median_price"].iloc[0])
        else:
            assert out["suburb_median_price"].iloc[i] == earlier.median()
            # the row's own price (and anything later) must not influence it
            assert out["suburb_median_price"].iloc[i] < df["price"].iloc[i]


def test_same_day_sales_are_excluded():
    df = ordered_frame()
    df.loc[5, "sale_date"] = df.loc[4, "sale_date"]  # two sales on the same day
    out = add_suburb_features(df.sort_values("sale_date"), df)
    assert out.loc[5, "suburb_sales_count"] == out.loc[4, "suburb_sales_count"] == 4


def test_future_reference_rows_are_ignored():
    df = ordered_frame()
    future = df.copy()
    future["sale_date"] = future["sale_date"] + pd.Timedelta(days=365)
    future["price"] = 1e9
    out = add_suburb_features(df, pd.concat([df, future]))
    assert out["suburb_median_price"].max() < 1e6


def test_feature_matrix_shape_and_order():
    X, y = build_feature_matrix(ordered_frame(), ordered_frame())
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == 20
    assert X["pt_house"].eq(1).all() and X["property_age"].iloc[0] == 71
