"""End-to-end, reproducible training pipeline.

    python scripts/train.py --data data/raw.csv --seed 42

1. ingest raw data to canonical form
2. build the quality report and write DATA_QUALITY_REPORT.md
3. abort if the quality score is below the gate (a lifecycle control)
4. build temporal-safe features
5. train the suburb-median baseline + LightGBM, XGBoost, Random Forest
6. evaluate on a temporal split and print the leaderboard
7. log everything to MLflow, register the best model by PPE10, promote it
8. save the feature list / preprocessing config alongside the model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402
from src.config import settings  # noqa: E402
from src.ingest import ingest_file  # noqa: E402
from src.models.baseline import SuburbMedianBaseline  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    evaluate_intervals,
    evaluate_valuation,
    resolve_cutoff,
    temporal_split,
)
from src.models.forecast import ALL_SUBURBS, METHODS, backtest, build_price_index  # noqa: E402
from src.models.valuation import ALGOS, ValuationModel  # noqa: E402
from src.quality import build_report, filter_trainable, render_markdown  # noqa: E402
from src.registry import Registry  # noqa: E402

EXIT_QUALITY_GATE = 2
EXIT_BASELINE_GATE = 3


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_forecast_backtests(df: pd.DataFrame, horizon: int = 3, n_folds: int = 5, top_n: int = 5) -> pd.DataFrame:
    """Backtest every forecasting method on the whole market and the busiest suburbs."""
    targets = [ALL_SUBURBS, *df["suburb"].value_counts().head(top_n).index]
    rows = []
    for target in targets:
        series = build_price_index(df, target, min_sales=3)
        for method in METHODS:
            try:
                rows.append({"series": target, "n_months": len(series), **backtest(series, method, horizon, n_folds)})
            except ValueError as exc:
                print(f"  skip forecast backtest {target}/{method}: {exc}")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "raw.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cutoff", default=None, help="temporal test cutoff (YYYY-MM-DD)")
    p.add_argument("--mlflow-uri", default=settings.mlflow_uri)
    p.add_argument("--reports-dir", default=str(ROOT / "reports"))
    p.add_argument("--quality-report", default=str(ROOT / settings.quality_report_path))
    p.add_argument("--algos", default=",".join(ALGOS))
    p.add_argument("--fail-if-not-better-than-baseline", action="store_true",
                   help="exit non-zero if the best model does not beat the baseline on PPE10 (CI gate)")
    args = p.parse_args(argv)
    set_seeds(args.seed)
    t0 = time.time()
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # 1. ingest
    df, ingest_report = ingest_file(args.data)
    print(f"[1/8] ingest: {ingest_report.model_dump()}")

    # 2. quality report
    quality = build_report(df)
    Path(args.quality_report).write_text(render_markdown(quality))
    (reports / "quality_report.json").write_text(quality.model_dump_json(indent=2))
    print(f"[2/8] quality score {quality.overall_score:.3f}; {len(quality.issues)} rules fired; "
          f"report -> {args.quality_report}")

    # 3. quality gate
    if quality.overall_score < settings.quality_gate_threshold:
        print(f"ABORT: data quality score {quality.overall_score:.3f} is below the gate "
              f"{settings.quality_gate_threshold}. Fix the input (see {args.quality_report}) — "
              "no model was trained.", file=sys.stderr)
        return EXIT_QUALITY_GATE
    clean, n_dropped = filter_trainable(df)
    print(f"[3/8] quality gate passed; dropped {n_dropped} rows breaking high-severity rules")

    # 4. features (suburb aggregates are strictly backward-looking per row)
    cutoff = resolve_cutoff(clean, args.cutoff or settings.test_cutoff_date)
    X, y = F.build_feature_matrix(clean, clean)
    X["suburb"] = clean["suburb"].astype(object).values  # used only by the baseline
    is_test = pd.to_datetime(clean["sale_date"]) >= pd.Timestamp(cutoff)
    train_df, test_df = temporal_split(clean, cutoff)
    X_train, y_train, X_test, y_test = X[~is_test], y[~is_test], X[is_test], y[is_test]
    print(f"[4/8] features: {len(F.FEATURE_COLUMNS)} cols; cutoff {cutoff}: "
          f"train {len(train_df):,} ({train_df.sale_date.min():%Y-%m}..{train_df.sale_date.max():%Y-%m}), "
          f"test {len(test_df):,} ({test_df.sale_date.min():%Y-%m}..{test_df.sale_date.max():%Y-%m})")

    # 5 + 6. train + evaluate
    alpha = settings.interval_alpha
    models: dict[str, object] = {"suburb_median_baseline": SuburbMedianBaseline()}
    for algo in args.algos.split(","):
        models[algo] = ValuationModel(algo, interval_alpha=alpha, random_state=args.seed)
    rows, preds = [], {}
    for name, model in models.items():
        t = time.time()
        model.fit(X_train, y_train)
        row = {"model": name, **evaluate_valuation(y_test, model.predict(X_test))}
        if isinstance(model, ValuationModel):
            lo_raw, _, hi_raw = model.predict_interval(X_test, alpha, calibrated=False)
            lo, pt, hi = model.predict_interval(X_test, alpha)
            raw = evaluate_intervals(y_test, lo_raw, hi_raw, 1 - alpha)
            cal = evaluate_intervals(y_test, lo, hi, 1 - alpha)
            row.update(coverage_raw=raw["coverage"], coverage=cal["coverage"],
                       median_relative_width=cal["median_relative_width"])
            preds[name] = (lo, pt, hi, lo_raw, hi_raw)
        row["fit_seconds"] = round(time.time() - t, 1)
        rows.append(row)
    board = pd.DataFrame(rows).sort_values("ppe10", ascending=False).reset_index(drop=True)
    board.to_csv(reports / "leaderboard.csv", index=False)
    print("[5-6/8] leaderboard (temporal test set):")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
        print(board.to_string(index=False))

    baseline_ppe10 = float(board.loc[board.model == "suburb_median_baseline", "ppe10"].iloc[0])
    best = board[board.model != "suburb_median_baseline"].iloc[0]
    beats_baseline = bool(best.ppe10 > baseline_ppe10)

    lo, pt, hi, lo_raw, hi_raw = preds[best.model]
    pd.DataFrame({
        "sale_date": test_df["sale_date"].values, "suburb": test_df["suburb"].astype(object).values,
        "property_type": test_df["property_type"].astype(object).values,
        "y_true": y_test.values, "y_pred": pt, "lower": lo, "upper": hi, "lower_raw": lo_raw, "upper_raw": hi_raw,
    }).to_parquet(reports / "test_predictions.parquet", index=False)

    fc = run_forecast_backtests(clean)
    fc.to_csv(reports / "forecast_backtest.csv", index=False)
    print("[6/8] forecast backtest (horizon 3 months, 5 folds; MASE < 1 beats naive):")
    print(fc[["series", "method", "mae", "mape", "mase", "interval_coverage"]].to_string(index=False))

    # 7 + 8. log to MLflow with the preprocessing config, register best by PPE10
    data_as_of = pd.to_datetime(clean["sale_date"]).max().strftime("%Y-%m-%d")
    training_id = f"train-{int(time.time())}"
    feature_config = {
        "feature_columns": F.FEATURE_COLUMNS, "epoch": str(F.EPOCH.date()),
        "suburb_window_days": F.SUBURB_WINDOW_DAYS, "market_window_days": F.MARKET_WINDOW_DAYS,
        "property_types": F.PROPERTY_TYPES, "interval_alpha": alpha, "cutoff": cutoff,
        "data_as_of": data_as_of, "seed": args.seed, "data_path": str(args.data),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    (reports / "feature_config.json").write_text(json.dumps(feature_config, indent=2))
    ref_path = reports / "reference.parquet"
    clean.to_parquet(ref_path, index=False)
    X_train[F.FEATURE_COLUMNS].sample(min(5000, len(X_train)), random_state=args.seed).to_parquet(
        reports / "train_feature_sample.parquet", index=False)

    registry = Registry(args.mlflow_uri)
    for _, r in board.iterrows():
        name = r["model"]
        metrics = {k: v for k, v in r.items() if k != "model"}
        model = models[name]
        params = {"algo": name, "seed": args.seed, "cutoff": cutoff,
                  **(getattr(model, "params", {}) or {})}
        cfg = {**feature_config, "algo": name, "test_metrics": metrics}
        cfg_path = reports / f"model_config_{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, default=str))
        artifacts = {"config": str(cfg_path), "reports": str(reports / "leaderboard.csv"),
                     "quality": str(reports / "quality_report.json"), "reference": str(ref_path),
                     "monitoring": str(reports / "train_feature_sample.parquet")}
        registry.log_run(model, params, metrics, artifacts,
                         model_name=settings.model_name if name != "suburb_median_baseline" else "baseline",
                         tags={"training_id": training_id, "algo": name})

    if not beats_baseline:
        print(f"[7/8] best model {best.model} PPE10 {best.ppe10:.3f} does NOT beat the baseline "
              f"{baseline_ppe10:.3f}; not promoting.", file=sys.stderr)
        return EXIT_BASELINE_GATE if args.fail_if_not_better_than_baseline else 0

    version = registry.register_best(settings.model_name, "ppe10", higher_is_better=True,
                                     training_id=training_id)
    registry.promote(settings.model_name, version, "production")
    summary = {"model": best.model, "version": version, "ppe10": float(best.ppe10),
               "baseline_ppe10": baseline_ppe10, "coverage": float(best.coverage),
               "coverage_raw": float(best.coverage_raw), "quality_score": quality.overall_score,
               "cutoff": cutoff, "data_as_of": data_as_of, "seconds": round(time.time() - t0, 1)}
    (reports / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[7-8/8] registered {settings.model_name} v{version} ({best.model}) -> @production. "
          f"PPE10 {best.ppe10:.3f} vs baseline {baseline_ppe10:.3f}; "
          f"90% interval coverage {best.coverage:.3f} (raw quantile {best.coverage_raw:.3f}). "
          f"Done in {summary['seconds']}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
