import numpy as np
import pandas as pd

from src.quality import build_report, check_integrity, compute_score, filter_trainable, render_markdown


def clean_frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sale_id": [f"s{i}" for i in range(n)],
        "sale_date": pd.date_range("2020-01-01", periods=n, freq="7D"),
        "price": rng.normal(900_000, 50_000, n),
        "suburb": ["Kew"] * n,
        "postcode": ["3101"] * n,
        "state": ["VIC"] * n,
        "property_type": ["house"] * n,
        "bedrooms": [3] * n,
        "bathrooms": [2] * n,
        "parking": [1] * n,
        "land_size_sqm": [500.0] * n,
        "building_area_sqm": [180.0] * n,
        "latitude": np.linspace(-37.80, -37.81, n),
        "longitude": np.linspace(145.0, 145.01, n),
        "year_built": [1960] * n,
    })


def violating_frame() -> pd.DataFrame:
    df = clean_frame()
    df.loc[0, "price"] = -5                                      # price_out_of_range (high)
    df.loc[1, "sale_date"] = pd.Timestamp("2099-01-01")          # sale_date_in_future (high)
    df.loc[2, "building_area_sqm"] = 900.0                       # building_larger_than_land (medium)
    df.loc[3, "year_built"] = 1700                               # implausible_year_built (medium)
    df.loc[4, "sale_id"] = "s5"                                  # duplicate_sale_id (high)
    df.loc[6, "postcode"] = None                                 # suburb/postcode mismatch (low)
    df.loc[7, "bedrooms"] = 25                                   # implausible_bedrooms (medium)
    return df


def test_clean_data_has_no_issues():
    assert check_integrity(clean_frame()) == []


def test_each_violation_caught_with_correct_severity():
    found = {i.rule: i for i in check_integrity(violating_frame())}
    expected = {
        "price_out_of_range": "high",
        "sale_date_in_future": "high",
        "building_larger_than_land": "medium",
        "implausible_year_built": "medium",
        "duplicate_sale_id": "high",
        "suburb_postcode_mismatch_nulls": "low",
        "implausible_bedrooms": "medium",
    }
    for rule, severity in expected.items():
        assert rule in found, f"{rule} not detected"
        assert found[rule].severity == severity
        assert found[rule].count == 1


def test_suburb_price_outlier_is_low_severity():
    df = clean_frame()
    df.loc[10, "price"] = 9_000_000
    found = {i.rule: i for i in check_integrity(df)}
    assert found["price_outlier_within_suburb"].severity == "low"


def test_score_drops_on_corrupted_data_and_report_renders():
    good, bad = build_report(clean_frame()), build_report(violating_frame())
    assert good.overall_score == 1.0
    assert bad.overall_score < good.overall_score
    corrupted = clean_frame()
    corrupted.loc[: len(corrupted) // 2, "price"] = -1
    corrupted[["bathrooms", "parking", "land_size_sqm", "building_area_sqm", "year_built"]] = None
    assert build_report(corrupted).overall_score < 0.7  # the training gate would block this
    md = render_markdown(bad)
    assert "# Data Quality Report" in md and "price_out_of_range" in md


def test_score_formula():
    report = build_report(clean_frame())
    assert compute_score(report.profiles, [], 0) == 0.0
    assert compute_score(report.profiles, [], report.n_rows) == 1.0


def test_filter_trainable_drops_rows_and_nulls_fields():
    df = violating_frame()
    df.loc[8, "land_size_sqm"] = 0.0
    out, dropped = filter_trainable(df)
    assert dropped == 3  # bad price, future date, duplicate id
    assert len(out) == len(df) - 3
    assert out["land_size_sqm"].isna().sum() == 1  # zero land nulled, row kept
