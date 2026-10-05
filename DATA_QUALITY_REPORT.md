# Data Quality Report

_Generated 2026-10-05T07:56:30+00:00 by `src/quality.py`. Do not edit by hand._

**Rows:** 27,247  
**Overall score:** 0.878 (training gate 0.70: **PASS**)

Score = 0.5 x mean completeness + 0.5 x (1 - severity-weighted issue rate); weights high=1.0, medium=0.5, low=0.1.

## Integrity issues

| Severity | Rule | Column | Rows | % rows | Example |
|---|---|---|---:|---:|---|
| high | `non_positive_land_size` | land_size_sqm | 1,942 | 7.1% | sale_id=e51059f5930b6a85: land_size_sqm=0.0 |
| high | `non_positive_building_area` | building_area_sqm | 61 | 0.2% | sale_id=8013240e697cac45: building_area_sqm=0.0 |
| high | `duplicate_sale_id` | sale_id | 18 | 0.1% | sale_id=fd4d6be50f055917: sale_id=fd4d6be50f055917 |
| medium | `building_larger_than_land` | building_area_sqm | 311 | 1.1% | sale_id=128579329666c027: building_area_sqm=115.0 |
| medium | `same_property_sold_twice_same_day` | - | 26 | 0.1% | sale_id=4cc7925362f8b9f0: suburb=Reservoir, date=2016-08-13 00:00:00 |
| medium | `implausible_year_built` | year_built | 1 | 0.0% | sale_id=3ab406ea2b8be2fa: year_built=1196 |
| low | `price_outlier_within_suburb` | price | 152 | 0.6% | sale_id=583385ff64e0fc22: price=1631000.0 |
| low | `suburb_postcode_mismatch_nulls` | postcode | 1 | 0.0% | sale_id=ba292720a26dd1b4: postcode=<NA> |

High-severity rules drop the row from training, except zero/negative land or building area, where only that field is nulled (strata units record land as 0). Medium and low are flagged but kept.

## Column profiles

| Column | dtype | Completeness | Unique | Min | Max | 3-IQR outliers |
|---|---|---:|---:|---|---|---:|
| sale_id | string | 100.0% | 27,229 | 0001f34b5cf5ecac | fff9e9e8e027e16a | 0 |
| sale_date | datetime64[us] | 100.0% | 78 | 2016-01-28 00:00:00 | 2018-03-17 00:00:00 | 0 |
| price | float64 | 100.0% | 2,871 | 85000.0 | 11200000.0 | 325 |
| suburb | string | 100.0% | 343 | Abbotsford | Yarraville | 0 |
| postcode | string | 100.0% | 209 | 3000 | 3978 | 0 |
| state | string | 100.0% | 1 | VIC | VIC | 0 |
| property_type | string | 100.0% | 3 | house | unit | 0 |
| bedrooms | Int64 | 100.0% | 12 | 1 | 16 | 3 |
| bathrooms | Int64 | 76.3% | 10 | 0 | 9 | 16 |
| parking | Int64 | 75.0% | 13 | 0 | 18 | 151 |
| land_size_sqm | float64 | 66.0% | 1,557 | 0.0 | 433014.0 | 285 |
| building_area_sqm | float64 | 39.1% | 662 | 0.0 | 44515.0 | 142 |
| latitude | float64 | 77.0% | 11,366 | -38.19043 | -37.3978 | 7 |
| longitude | float64 | 77.0% | 12,275 | 144.42379 | 145.52635 | 9 |
| year_built | Int64 | 44.4% | 151 | 1196 | 2019 | 1 |

## Coverage by suburb

343 suburbs. 107 have fewer than 20 sales — valuations there rely on few comparables and carry wider intervals.

| Busiest suburbs | Sales |
|---|---:|
| Reservoir | 727 |
| Bentleigh East | 493 |
| Richmond | 439 |
| Preston | 415 |
| Brunswick | 387 |
| Essendon | 361 |
| Northcote | 345 |
| Glenroy | 342 |
| South Yarra | 328 |
| Glen Iris | 323 |

| Thinnest suburbs | Sales |
|---|---:|
| Fawkner Lot | 1 |
| Ferny Creek | 1 |
| Kalkallo | 1 |
| Lysterfield | 1 |
| Monbulk | 1 |
| Montrose | 1 |
| Tecoma | 1 |
| Wandin North | 1 |
| Wildwood | 1 |
| Yarra Glen | 1 |

## Coverage by month

| Month | Sales |
|---|---:|
| 2016-01 | 2 |
| 2016-02 | 35 |
| 2016-04 | 401 |
| 2016-05 | 1,167 |
| 2016-06 | 962 |
| 2016-07 | 553 |
| 2016-08 | 911 |
| 2016-09 | 1,166 |
| 2016-10 | 677 |
| 2016-11 | 1,413 |
| 2016-12 | 767 |
| 2017-02 | 526 |
| 2017-03 | 841 |
| 2017-04 | 805 |
| 2017-05 | 1,453 |
| 2017-06 | 1,463 |
| 2017-07 | 1,806 |
| 2017-08 | 1,413 |
| 2017-09 | 2,053 |
| 2017-10 | 2,441 |
| 2017-11 | 1,995 |
| 2017-12 | 723 |
| 2018-01 | 647 |
| 2018-02 | 1,506 |
| 2018-03 | 1,521 |
