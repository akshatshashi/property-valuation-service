import pandas as pd

from src.config import CANONICAL_SCHEMA
from src.ingest import to_canonical

COLUMN_MAP = {"Date": "sale_date", "Price": "price", "Suburb": "suburb", "Postcode": "postcode",
              "Type": "property_type", "Rooms": "bedrooms", "ID": "sale_id"}


def raw() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": ["a", "b", "c", "d"],
        "Date": ["2017-03-04", "not a date", "2017-05-06", None],
        "Price": [1_000_000, 750_000, None, 500_000],
        "Suburb": [" richmond ", "Kew", "Kew", "Kew"],
        "Postcode": [3121.0, 3101.0, 3101.0, 3101.0],
        "Type": ["h", "u", "t", "h"],
        "Rooms": [3, 2, 2, 4],
    })


def test_renames_to_canonical_schema():
    df, report = to_canonical(raw(), COLUMN_MAP)
    assert list(df.columns) == CANONICAL_SCHEMA
    row = df.iloc[0]
    assert row.suburb == "Richmond" and row.postcode == "3121" and row.property_type == "house"
    assert row.bedrooms == 3 and str(df["bedrooms"].dtype) == "Int64"
    assert report.rows_in == 4 and report.rows_out == 1


def test_bad_and_missing_rows_rejected_with_reason():
    _, report = to_canonical(raw(), COLUMN_MAP)
    assert report.rows_rejected == 3
    assert report.rejection_reasons == {"unparseable_sale_date": 1, "missing_price": 1, "missing_sale_date": 1}


def test_missing_optional_column_created_as_null():
    df, _ = to_canonical(raw(), COLUMN_MAP)
    for col in ["year_built", "land_size_sqm", "latitude"]:
        assert col in df.columns and df[col].isna().all()
