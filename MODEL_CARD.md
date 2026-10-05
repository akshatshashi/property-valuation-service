# Model Card — Property Valuation Model

## What it does
Estimates the likely sale price of a residential property in metropolitan Melbourne from its
basic attributes (suburb, type, bedrooms, bathrooms, parking, land and building size, year built).
Every estimate comes with a **90% range** and the **top five factors** that drove it.

A companion model forecasts each suburb's **monthly median sale price** a few months ahead.

> **This is a decision-support estimate, not a formal valuation.** It has not inspected the
> property, does not know its condition, renovations, views, aspect or street appeal, and must not
> be used for lending, legal or tax purposes in place of a certified valuer.

## Data
- **Source:** Kaggle "Melbourne Housing Market" (public sales records, Jan 2016 – Mar 2018).
- **Volume:** 34,857 raw records; 27,229 usable sales after removing records with no price and 18
  exact duplicates. 343 suburbs.
- **Data quality score:** 0.878 / 1.0 (training is blocked below 0.70). Full detail in
  `DATA_QUALITY_REPORT.md`. Notable gaps: building area is missing for ~60% of sales and year built
  for ~55%, so the model leans on location and room counts for most properties.

## How accurate is it?
Tested the honest way: trained on sales before 18 Nov 2017, then graded on the 5,754 sales that
happened *after* that date, which the model had never seen.

| In plain terms | Result |
|---|---|
| Estimate within **10%** of the actual sale price | **49%** of the time |
| Estimate within **20%** of the actual sale price | **77%** of the time |
| Typical (median) error | **10.3%** of the sale price |
| Sale price fell inside the stated 90% range | **90.3%** of the time |
| Typical width of the 90% range | about **±30%** of the estimate |

For comparison, simply quoting each suburb's median price lands within 10% only **30%** of the time,
so the model is a substantial improvement over a suburb-median rule of thumb. LightGBM was selected
over XGBoost (48%) and Random Forest (47%).

The 90% range is honest: raw quantile models only covered 84.8% of sales, so a conformal
calibration step widens them to reach the promised 90%.

## Where it is least reliable
| Situation | Within 10% | Range coverage | Why |
|---|---:|---:|---|
| Units / apartments | 39% | 90% | Floor level, views, building quality and strata fees are not in the data |
| Most expensive 20% of homes | 43% | 86% | Prestige homes are unique; few true comparables |
| Cheapest 20% of homes | 46% | 87% | Includes distressed sales and data errors |
| Houses | 51% | 90% | Best-served segment |

- **Sparse suburbs.** Suburbs with fewer than 20 prior sales performed similarly in this test (51%),
  but their ranges rely on very few comparables and can move sharply as new sales arrive.
- **Unusual properties.** Very large blocks, heritage homes, knock-down/rebuild candidates and
  anything far outside typical size ranges are extrapolations.
- **Thin recent data.** The most recent month in the data has fewer sales than usual; suburb
  aggregates for the latest period are noisier.
- **Time.** The data ends in March 2018. Without retraining on current sales the estimates
  describe the 2018 market, not today's.

## Price forecasts
Backtested on 5 rolling 3-month windows and compared with "next month = this month" (naive).

- City-wide: SARIMA marginally beats naive (MASE 0.97); ETS does **not** (1.36).
- Busy suburbs: models beat naive in Richmond, Brunswick and Preston, but **not** in Reservoir or
  Bentleigh East, where prices were flat and naive is very hard to beat.
- Take-away: forecasts are directional guidance only; naive is shown alongside for comparison.

## Intended use
Internal decision support: triaging listings, sanity-checking price guides, portfolio monitoring.
**Not** for: lending decisions, statutory valuations, or showing to consumers without the range.

## Monitoring and retraining
Input drift (PSI > 0.25 on more than 30% of features), a drop of more than 5 points in the share of
estimates within 10%, or a model older than 90 days triggers retraining. See `RUNBOOK.md`.
