"""Data quality and coverage: can this data be trusted?

This is a first-class module, not an afterthought. It profiles every canonical
column, runs integrity rules that encode domain knowledge about property sales,
measures coverage by suburb and month, and condenses it all into one score
that ``scripts/train.py`` uses as a hard gate before any model is trained.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel

SEVERITY_WEIGHTS: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.1}
MAX_PRICE = 100_000_000
MIN_SUBURB_SALES_FOR_IQR = 5
SPARSE_SUBURB_THRESHOLD = 20


class ColumnProfile(BaseModel):
    column: str
    dtype: str
    completeness: float  # fraction non-null
    n_unique: int
    min_value: str | None
    max_value: str | None
    n_outliers: int


class QualityIssue(BaseModel):
    severity: str  # "high" | "medium" | "low"
    rule: str
    column: str | None
    count: int
    example: str


class QualityReport(BaseModel):
    n_rows: int
    as_of: str
    profiles: list[ColumnProfile]
    issues: list[QualityIssue]
    coverage_by_suburb: dict[str, int]
    coverage_by_month: dict[str, int]
    overall_score: float  # 0..1


# ---------------------------------------------------------------- integrity rules


def _iqr_outlier_mask(values: pd.Series, k: float = 3.0) -> pd.Series:
    """True where a value lies more than ``k`` IQRs outside the quartiles."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return pd.Series(False, index=values.index)
    return ((v < q1 - k * iqr) | (v > q3 + k * iqr)).fillna(False)


def _suburb_price_outliers(df: pd.DataFrame) -> pd.Series:
    """Price outside 3 IQRs of its own suburb (suburbs with >= 5 sales only)."""
    mask = pd.Series(False, index=df.index)
    for _, grp in df.groupby("suburb", dropna=True):
        if len(grp) >= MIN_SUBURB_SALES_FOR_IQR:
            mask.loc[grp.index] = _iqr_outlier_mask(grp["price"]).values
    return mask


def _same_property_same_day(df: pd.DataFrame) -> pd.Series:
    """Same address-like key sold more than once on the same date.

    The canonical schema has no street address, so the key is
    (suburb, latitude, longitude, property_type, bedrooms). Rows already sharing a
    sale_id are excluded so exact duplicates are not double-counted.
    """
    key = ["suburb", "latitude", "longitude", "property_type", "bedrooms", "sale_date"]
    has_key = df[["latitude", "longitude"]].notna().all(axis=1)
    sub = df[has_key & ~df["sale_id"].duplicated(keep=False)]
    dup = sub.duplicated(subset=key, keep=False)
    mask = pd.Series(False, index=df.index)
    mask.loc[dup[dup].index] = True
    return mask


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC).date())


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    column: str | None
    mask: Callable[[pd.DataFrame], pd.Series]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


RULES: list[Rule] = [
    Rule("price_out_of_range", "high", "price",
         lambda d: (_num(d, "price") <= 0) | (_num(d, "price") > MAX_PRICE)),
    Rule("sale_date_in_future", "high", "sale_date",
         lambda d: pd.to_datetime(d["sale_date"]) > _now()),
    Rule("non_positive_land_size", "high", "land_size_sqm", lambda d: _num(d, "land_size_sqm") <= 0),
    Rule("non_positive_building_area", "high", "building_area_sqm",
         lambda d: _num(d, "building_area_sqm") <= 0),
    Rule("building_larger_than_land", "medium", "building_area_sqm",
         lambda d: (_num(d, "building_area_sqm") > _num(d, "land_size_sqm"))
         & (_num(d, "land_size_sqm") > 0)),
    Rule("implausible_bedrooms", "medium", "bedrooms", lambda d: _num(d, "bedrooms") > 20),
    Rule("implausible_bathrooms", "medium", "bathrooms", lambda d: _num(d, "bathrooms") > 20),
    Rule("implausible_year_built", "medium", "year_built",
         lambda d: (_num(d, "year_built") < 1800) | (_num(d, "year_built") > _now().year)),
    Rule("duplicate_sale_id", "high", "sale_id",
         lambda d: d["sale_id"].notna() & d["sale_id"].duplicated(keep="first")),
    Rule("same_property_sold_twice_same_day", "medium", None, _same_property_same_day),
    Rule("price_outlier_within_suburb", "low", "price", _suburb_price_outliers),
    Rule("suburb_postcode_mismatch_nulls", "low", "postcode",
         lambda d: d["suburb"].isna() ^ d["postcode"].isna()),
]


def _rule_mask(rule: Rule, df: pd.DataFrame) -> pd.Series:
    return rule.mask(df).fillna(False).astype(bool)


def _example(df: pd.DataFrame, mask: pd.Series, column: str | None) -> str:
    row = df[mask].iloc[0]
    value = f"{column}={row[column]}" if column else f"suburb={row['suburb']}, date={row['sale_date']}"
    return f"sale_id={row['sale_id']}: {value}"


# ---------------------------------------------------------------- public API


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    """Completeness, cardinality, range and 3-IQR outlier count for every column."""
    profiles = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_date = pd.api.types.is_datetime64_any_dtype(s)
        if len(non_null) and (is_numeric or is_date):
            lo, hi = str(non_null.min()), str(non_null.max())
        elif len(non_null):
            ordered = non_null.astype(str).sort_values()
            lo, hi = ordered.iloc[0], ordered.iloc[-1]
        else:
            lo = hi = None
        profiles.append(
            ColumnProfile(
                column=col,
                dtype=str(s.dtype),
                completeness=round(float(s.notna().mean()) if len(s) else 0.0, 4),
                n_unique=int(non_null.nunique()),
                min_value=lo,
                max_value=hi,
                n_outliers=int(_iqr_outlier_mask(s).sum()) if is_numeric else 0,
            )
        )
    return profiles


def check_integrity(df: pd.DataFrame) -> list[QualityIssue]:
    """Run every rule in ``RULES``; return one issue per rule that fired."""
    issues = []
    for rule in RULES:
        mask = _rule_mask(rule, df)
        if mask.any():
            issues.append(
                QualityIssue(
                    severity=rule.severity,
                    rule=rule.name,
                    column=rule.column,
                    count=int(mask.sum()),
                    example=_example(df, mask, rule.column),
                )
            )
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: (order[i.severity], -i.count))


COVERAGE_BY_SUBURB_SQL = """
    SELECT suburb, COUNT(*) AS n FROM sales
    WHERE suburb IS NOT NULL GROUP BY suburb ORDER BY n DESC, suburb
"""
COVERAGE_BY_MONTH_SQL = """
    SELECT strftime(sale_date, '%Y-%m') AS month, COUNT(*) AS n FROM sales
    WHERE sale_date IS NOT NULL GROUP BY month ORDER BY month
"""


def coverage(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Sales count per suburb (descending) and per calendar month (chronological), in DuckDB SQL."""
    sales = pd.DataFrame({"suburb": df["suburb"].astype(object), "sale_date": pd.to_datetime(df["sale_date"])})
    with duckdb.connect() as con:
        con.register("sales", sales)
        by_suburb = con.execute(COVERAGE_BY_SUBURB_SQL).fetchall()
        by_month = con.execute(COVERAGE_BY_MONTH_SQL).fetchall()
    return {str(s): int(n) for s, n in by_suburb}, {str(m): int(n) for m, n in by_month}


def compute_score(profiles: list[ColumnProfile], issues: list[QualityIssue], n_rows: int) -> float:
    """Overall quality score in [0, 1].

    ``score = 0.5 * completeness + 0.5 * (1 - issue_penalty)``

    - ``completeness`` = mean fraction of non-null values across all columns.
    - ``issue_penalty`` = min(1, sum(weight[severity] * count) / n_rows), with
      weights high=1.0, medium=0.5, low=0.1 — i.e. the severity-weighted share
      of rows that broke a rule.

    A perfect, complete, rule-clean dataset scores 1.0. The training gate is 0.7.
    """
    if n_rows == 0:
        return 0.0
    completeness = float(np.mean([p.completeness for p in profiles])) if profiles else 0.0
    weighted = sum(SEVERITY_WEIGHTS[i.severity] * i.count for i in issues)
    penalty = min(1.0, weighted / n_rows)
    return round(0.5 * completeness + 0.5 * (1.0 - penalty), 4)


def build_report(df: pd.DataFrame) -> QualityReport:
    profiles = profile_columns(df)
    issues = check_integrity(df)
    by_suburb, by_month = coverage(df)
    return QualityReport(
        n_rows=len(df),
        as_of=datetime.now(UTC).isoformat(timespec="seconds"),
        profiles=profiles,
        issues=issues,
        coverage_by_suburb=by_suburb,
        coverage_by_month=by_month,
        overall_score=compute_score(profiles, issues, len(df)),
    )


# High-severity rules about one optional field: null the field, keep the sale.
# (e.g. land_size 0 is how strata-titled units are recorded — the price is still real.)
FIELD_LEVEL_RULES = {"non_positive_land_size", "non_positive_building_area"}


def filter_trainable(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Prepare data for training from the high-severity rules.

    Field-level high rules null the offending field; every other high rule drops
    the row. Medium/low issues are flagged in the report, never dropped.
    Returns the cleaned frame and the number of rows dropped.
    """
    out = df.copy()
    bad = pd.Series(False, index=out.index)
    for rule in RULES:
        if rule.severity != "high":
            continue
        mask = _rule_mask(rule, out)
        if rule.name in FIELD_LEVEL_RULES and rule.column:
            out.loc[mask, rule.column] = np.nan
        else:
            bad |= mask
    return out[~bad].reset_index(drop=True), int(bad.sum())


def render_markdown(report: QualityReport) -> str:
    """Render the report as DATA_QUALITY_REPORT.md."""
    gate = "PASS" if report.overall_score >= 0.7 else "FAIL"
    lines = [
        "# Data Quality Report",
        "",
        f"_Generated {report.as_of} by `src/quality.py`. Do not edit by hand._",
        "",
        f"**Rows:** {report.n_rows:,}  ",
        f"**Overall score:** {report.overall_score:.3f} (training gate 0.70: **{gate}**)",
        "",
        "Score = 0.5 x mean completeness + 0.5 x (1 - severity-weighted issue rate); "
        "weights high=1.0, medium=0.5, low=0.1.",
        "",
        "## Integrity issues",
        "",
    ]
    if report.issues:
        lines += ["| Severity | Rule | Column | Rows | % rows | Example |", "|---|---|---|---:|---:|---|"]
        for i in report.issues:
            pct = 100 * i.count / max(report.n_rows, 1)
            lines.append(f"| {i.severity} | `{i.rule}` | {i.column or '-'} | {i.count:,} | {pct:.1f}% | {i.example} |")
    else:
        lines.append("No integrity rules fired.")
    lines += [
        "",
        "High-severity rules drop the row from training, except zero/negative land or building area, "
        "where only that field is nulled (strata units record land as 0). Medium and low are flagged but kept.",
        "",
        "## Column profiles",
        "",
        "| Column | dtype | Completeness | Unique | Min | Max | 3-IQR outliers |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for p in report.profiles:
        lines.append(
            f"| {p.column} | {p.dtype} | {p.completeness:.1%} | {p.n_unique:,} | "
            f"{p.min_value} | {p.max_value} | {p.n_outliers:,} |"
        )
    subs = report.coverage_by_suburb
    sparse = [s for s, n in subs.items() if n < SPARSE_SUBURB_THRESHOLD]
    lines += [
        "",
        "## Coverage by suburb",
        "",
        f"{len(subs):,} suburbs. {len(sparse):,} have fewer than {SPARSE_SUBURB_THRESHOLD} sales — "
        "valuations there rely on few comparables and carry wider intervals.",
        "",
        "| Busiest suburbs | Sales |",
        "|---|---:|",
    ]
    lines += [f"| {s} | {n:,} |" for s, n in list(subs.items())[:10]]
    lines += ["", "| Thinnest suburbs | Sales |", "|---|---:|"]
    lines += [f"| {s} | {n:,} |" for s, n in list(subs.items())[-10:]]
    lines += ["", "## Coverage by month", "", "| Month | Sales |", "|---|---:|"]
    lines += [f"| {m} | {n:,} |" for m, n in report.coverage_by_month.items()]
    return "\n".join(lines) + "\n"
