"""Ingest raw property sales from any source into one canonical schema.

Every source (Kaggle Melbourne, NSW Valuer General, ...) is described by a
``column_map`` from its own column names to canonical names. Everything
downstream only ever sees the canonical frame, so the pipeline is source-agnostic.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from pydantic import BaseModel

from src.config import CANONICAL_SCHEMA

INT_COLS = ["bedrooms", "bathrooms", "parking", "year_built"]
FLOAT_COLS = ["price", "land_size_sqm", "building_area_sqm", "latitude", "longitude"]
STR_COLS = ["sale_id", "suburb", "postcode", "state", "property_type"]

PROPERTY_TYPE_MAP = {
    "h": "house",
    "house": "house",
    "u": "unit",
    "unit": "unit",
    "apartment": "unit",
    "flat": "unit",
    "t": "townhouse",
    "townhouse": "townhouse",
}

# Kaggle "Melbourne Housing Market" (Melbourne_housing_FULL.csv).
# `Rooms` is used for bedrooms because `Bedroom2` was scraped from a second
# source and is ~24% missing; the two agree on the large majority of rows.
MELBOURNE_COLUMN_MAP: dict[str, str] = {
    "Date": "sale_date",
    "Price": "price",
    "Suburb": "suburb",
    "Postcode": "postcode",
    "Type": "property_type",
    "Rooms": "bedrooms",
    "Bathroom": "bathrooms",
    "Car": "parking",
    "Landsize": "land_size_sqm",
    "BuildingArea": "building_area_sqm",
    "Lattitude": "latitude",
    "Longtitude": "longitude",
    "YearBuilt": "year_built",
}
MELBOURNE_CONSTANTS: dict[str, str] = {"state": "VIC"}
MELBOURNE_DATE_FORMAT = "%d/%m/%Y"


class IngestReport(BaseModel):
    """What happened to the rows on the way into the canonical schema."""

    rows_in: int
    rows_out: int
    rows_rejected: int
    rejection_reasons: dict[str, int]


def load_csv(path: str, column_map: dict[str, str]) -> pd.DataFrame:
    """Read a raw CSV, keeping only the columns named in ``column_map`` that exist.

    Missing source columns are tolerated here; ``to_canonical`` creates them as null.
    """
    header = pd.read_csv(path, nrows=0).columns
    wanted = [c for c in column_map if c in header]
    return pd.read_csv(path, usecols=wanted, low_memory=False)


def _stable_id(row: pd.Series) -> str:
    """Deterministic sale id for sources that do not ship one."""
    raw = "|".join(str(v) for v in row.values)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _clean_postcode(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = num.astype("Int64").astype("string")
    # keep non-numeric postcodes as-is rather than silently nulling them
    return out.where(num.notna(), s.astype("string"))


def to_canonical(
    df: pd.DataFrame,
    column_map: dict[str, str],
    constants: dict[str, str] | None = None,
    date_format: str | None = None,
) -> tuple[pd.DataFrame, IngestReport]:
    """Map a raw frame to the canonical schema and report what was rejected.

    - Renames source columns via ``column_map``.
    - Creates any missing canonical column as all-null instead of raising:
      real property data is patchy and the pipeline must survive it.
    - Coerces types: ``sale_date`` -> datetime, ``price`` -> float, counts -> nullable Int64.
    - Rejects rows with no usable price or sale date, recording one reason per row.
    """
    rows_in = len(df)
    out = df.rename(columns=column_map).copy()
    for col, value in (constants or {}).items():
        out[col] = value
    for col in CANONICAL_SCHEMA:
        if col not in out.columns:
            out[col] = pd.NA

    # sale_id: generate a stable one when the source has none
    if out["sale_id"].isna().all():
        out["sale_id"] = df.apply(_stable_id, axis=1).values

    raw_date = out["sale_date"]
    raw_price = out["price"]
    out["sale_date"] = pd.to_datetime(raw_date, errors="coerce", format=date_format)
    out["price"] = pd.to_numeric(raw_price, errors="coerce").astype(float)
    for col in FLOAT_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")

    out["postcode"] = _clean_postcode(out["postcode"])
    out["suburb"] = out["suburb"].astype("string").str.strip().str.title()
    out["state"] = out["state"].astype("string").str.strip().str.upper()
    out["sale_id"] = out["sale_id"].astype("string")
    ptype = out["property_type"].astype("string").str.strip().str.lower()
    out["property_type"] = ptype.map(lambda v: PROPERTY_TYPE_MAP.get(v, v) if pd.notna(v) else pd.NA)
    out["property_type"] = out["property_type"].astype("string")

    # One reason per rejected row, checked in priority order.
    reasons = pd.Series(pd.NA, index=out.index, dtype="string")
    checks = [
        ("missing_price", raw_price.isna()),
        ("unparseable_price", raw_price.notna() & out["price"].isna()),
        ("missing_sale_date", raw_date.isna()),
        ("unparseable_sale_date", raw_date.notna() & out["sale_date"].isna()),
    ]
    for reason, mask in checks:
        reasons = reasons.mask(reasons.isna() & mask, reason)

    rejected = reasons.notna()
    counts = reasons[rejected].value_counts().to_dict()
    out = out.loc[~rejected, CANONICAL_SCHEMA].sort_values("sale_date", kind="stable").reset_index(drop=True)
    report = IngestReport(
        rows_in=rows_in,
        rows_out=len(out),
        rows_rejected=int(rejected.sum()),
        rejection_reasons={str(k): int(v) for k, v in counts.items()},
    )
    return out, report


def ingest_file(
    path: str,
    column_map: dict[str, str] | None = None,
    constants: dict[str, str] | None = None,
    date_format: str | None = None,
) -> tuple[pd.DataFrame, IngestReport]:
    """Load + canonicalise a file. Defaults to the Melbourne preset if the file
    looks like it (has a ``Lattitude`` column), otherwise assumes canonical names."""
    header = set(pd.read_csv(path, nrows=0).columns)
    if column_map is None:
        if "Lattitude" in header:
            column_map, constants = MELBOURNE_COLUMN_MAP, MELBOURNE_CONSTANTS
            date_format = date_format or MELBOURNE_DATE_FORMAT
        else:
            column_map = {c: c for c in CANONICAL_SCHEMA}
    raw = load_csv(path, column_map)
    return to_canonical(raw, column_map, constants=constants, date_format=date_format)
