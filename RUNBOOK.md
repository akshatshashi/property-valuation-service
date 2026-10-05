# Runbook — Property Valuation Service

## Train
```bash
python scripts/train.py --data data/raw.csv --seed 42
```
Writes `DATA_QUALITY_REPORT.md`, logs every model to MLflow, registers the best by PPE10 and points
`models:/property-valuation@production` at it. Exit codes: `0` ok, `2` quality gate blocked
training, `3` best model did not beat the suburb-median baseline (with
`--fail-if-not-better-than-baseline`). Inspect runs with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## Deploy
```bash
docker build -t property-valuation .
docker run -p 8000:8000 -v "$PWD/mlruns:/app/mlruns" property-valuation
```
CI (`.github/workflows/ci.yml`) runs lint, tests, the baseline gate and a container smoke test on
every push. On AWS (`infra/main.tf`) push the image to ECR and update the ECS service; the
deployment circuit breaker rolls back automatically if `/health` fails.

## Roll back to a previous model
Nothing is deleted; rollback is re-pointing the alias.
```python
from src.registry import Registry
r = Registry("./mlruns")
r.list_versions("property-valuation")          # find the last good version
r.promote("property-valuation", "3", "production")
```
Then `curl -X POST localhost:8000/admin/reload` (or restart the container). Confirm with `/health`.

## `/health` fields
| Field | Meaning | Healthy |
|---|---|---|
| `status` | `ok` if a model is loaded, else `degraded` | `ok` |
| `model_version` | registry name/version serving traffic | the version you promoted |
| `model_loaded` | false means every `/value` returns 503 | `true` |
| `data_as_of` | most recent sale in the training reference data | within ~1 month of today |
| `load_error` | why loading failed, if it did | `null` |

## When things go wrong

### Drift alert fires
1. Run `detect_feature_drift(train_sample, live_features, FEATURE_COLUMNS)` and look at *which*
   features drifted. A new data feed often shifts one feature (e.g. `building_area_sqm` units).
2. If it is a pipeline bug, fix the feed — retraining on broken data bakes the bug in.
3. If the market genuinely moved (`market_median_price`, `suburb_median_price` drift), call
   `should_retrain` with the performance check; retrain if it fires.

### Quality gate blocks training (exit 2)
1. Open `DATA_QUALITY_REPORT.md`: the issues table is sorted by severity and has an example row.
2. Compare completeness with the last good report — a sudden drop is an upstream extract problem.
3. Fix at source or extend `src/ingest.py` mapping. Do **not** lower the 0.70 threshold to get a
   run through; the currently promoted model keeps serving in the meantime.

### API returns 503
The service is up but no production model is loaded. Check `/health` → `load_error`.
- `RESOURCE_DOES_NOT_EXIST` / alias not found: nothing has been promoted — run training or
  `promote(...)`.
- File/permission errors: the mounted `mlruns` volume is missing or not readable by uid 10001.
After fixing, `POST /admin/reload`.

### Forecast looks implausible
1. Check `method` in the response — `naive` means the suburb had under 6 months of history.
2. Look at `history`: a spike usually means a month with very few sales (median of 3 sales).
   `build_price_index(..., min_sales=3)` interpolates thinner months; consider raising it.
3. Compare with `reports/forecast_backtest.csv`; if MASE > 1 for that suburb, prefer naive.

## Worked incident (illustrative scenario)
**09:10 — alert:** the monitoring job reports PPE10 on last month's settled sales at 41%,
down from 49% at training (8.4 points), and `should_retrain` → *"Retrain: PPE10 dropped 8.4 points
(> 5)"*. Feature drift shows 2 of 23 features drifted — `market_median_price` (PSI 0.41) and
`suburb_median_price` (PSI 0.29); nothing else moved.

**09:20 — diagnose:** only market-level features drifted, so inputs are healthy; prices rose
faster than the model's training range. Errors are skewed (estimates too low), consistent with a
rising market rather than a data bug. The quality report for the new extract scores 0.86 — fine.

**09:35 — act:** retrain on data through last month:
`python scripts/train.py --data data/raw_latest.csv --seed 42`. New version v8 scores PPE10 0.50
on its own temporal hold-out and beats the baseline, so the pipeline promotes it.

**10:05 — verify:** `POST /admin/reload`; `/health` shows `property-valuation/v8` and the new
`data_as_of`. Spot-check five recent sales through `/value`. Keep v7 noted as the rollback target.

**Follow-up:** add a monthly scheduled retrain so model age never exceeds 90 days.
