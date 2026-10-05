"""Generate the small synthetic fixture dataset committed for CI.

    python scripts/make_fixture.py --rows 4000 --out tests/fixtures/sample_sales.csv

The real Kaggle data is not committed (licence + size), so CI trains on this
fixture. Prices follow a hedonic model — suburb level x size x type x a
market trend — plus noise, with a handful of deliberate data-quality defects
so the quality module has something to find.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SUBURBS = {  # name: (postcode, base $/sqm, lat, lon)
    "Northcote": ("3070", 9000, -37.770, 144.999), "Richmond": ("3121", 10500, -37.818, 145.000),
    "Brunswick": ("3056", 8500, -37.767, 144.960), "Preston": ("3072", 7600, -37.743, 145.005),
    "Reservoir": ("3073", 6200, -37.716, 145.007), "Bentleigh East": ("3165", 8000, -37.920, 145.064),
    "Hawthorn": ("3122", 12500, -37.822, 145.035), "Footscray": ("3011", 6800, -37.800, 144.900),
    "Coburg": ("3058", 7800, -37.744, 144.966), "Glen Waverley": ("3150", 8800, -37.878, 145.163),
    "Werribee": ("3030", 4300, -37.900, 144.661), "St Kilda": ("3182", 9800, -37.860, 144.980),
    "Kew": ("3101", 13000, -37.806, 145.030), "Sunshine": ("3020", 5600, -37.788, 144.832),
    "Doncaster": ("3108", 8400, -37.785, 145.124), "Essendon": ("3040", 9200, -37.752, 144.917),
}
TYPES = {"house": 0.55, "unit": 0.30, "townhouse": 0.15}


def generate(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(SUBURBS)
    suburb = rng.choice(names, n, p=np.linspace(1.6, 0.4, len(names)) / np.linspace(1.6, 0.4, len(names)).sum())
    ptype = rng.choice(list(TYPES), n, p=list(TYPES.values()))
    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 5 * 365, n), unit="D")
    beds = np.where(ptype == "unit", rng.integers(1, 4, n), rng.integers(2, 6, n))
    baths = np.clip(beds - rng.integers(0, 2, n), 1, None)
    building = np.where(ptype == "unit", 45 + 22 * beds, 70 + 32 * beds) * rng.normal(1, 0.12, n)
    land = np.where(ptype == "unit", np.nan, building * rng.uniform(1.6, 4.5, n))
    year_built = rng.integers(1900, 2022, n)
    months = (dates.year - 2019) * 12 + dates.month
    trend = 1 + 0.006 * months - 0.00006 * months**2  # rise then plateau
    base = np.array([SUBURBS[s][1] for s in suburb])
    type_mult = pd.Series(ptype).map({"house": 1.0, "unit": 0.82, "townhouse": 0.92}).to_numpy()
    land_premium = np.where(np.isnan(land), 1.0, 1 + 0.00025 * np.nan_to_num(land))
    age_mult = 1 - 0.0008 * (2024 - year_built)
    price = base * building * type_mult * land_premium * age_mult * trend * rng.lognormal(0, 0.10, n)

    df = pd.DataFrame({
        "sale_id": [f"FX{i:06d}" for i in range(n)], "sale_date": dates.strftime("%Y-%m-%d"),
        "price": np.round(price, -3), "suburb": suburb,
        "postcode": [SUBURBS[s][0] for s in suburb], "state": "VIC", "property_type": ptype,
        "bedrooms": beds, "bathrooms": baths, "parking": rng.integers(0, 3, n),
        "land_size_sqm": np.round(land, 0), "building_area_sqm": np.round(building, 0),
        "latitude": [SUBURBS[s][2] for s in suburb] + rng.normal(0, 0.006, n),
        "longitude": [SUBURBS[s][3] for s in suburb] + rng.normal(0, 0.006, n),
        "year_built": year_built,
    })
    # realistic patchiness + a few deliberate defects for the quality module
    for col, frac in {"building_area_sqm": 0.25, "year_built": 0.35, "parking": 0.1, "latitude": 0.05}.items():
        df.loc[rng.random(n) < frac, col] = np.nan
    df.loc[df.latitude.isna(), "longitude"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "price"] = -1
    df.loc[rng.choice(n, 3, replace=False), "year_built"] = 1066
    df.loc[rng.choice(n, 2, replace=False), "bedrooms"] = 45
    df = pd.concat([df, df.sample(2, random_state=seed)], ignore_index=True)  # duplicate sale_ids
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests/fixtures/sample_sales.csv"))
    a = p.parse_args()
    generate(a.rows, a.seed).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
