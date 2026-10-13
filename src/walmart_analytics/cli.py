"""Command line entry point: walmart-analytics {download,build,all}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import forecast, inventory, marts, pricing, report
from .data import build_warehouse, connect, download


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="walmart-analytics")
    ap.add_argument("step", choices=["download", "build", "all"])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    a = ap.parse_args(argv)
    raw, db = a.data / "raw", a.data / "warehouse.duckdb"

    if a.step in ("download", "all"):
        download(raw)
    if a.step in ("build", "all"):
        print("Building warehouse ...")
        build_warehouse(raw, db)
    if a.step != "all":
        return

    con = connect(db)
    print("Running SQL marts ...")
    m = marts.run_all(con)
    print("Backtesting forecasts (a few minutes) ...")
    series = forecast.add_features(forecast.load_series(con))
    preds = forecast.backtest(series, forecast.make_folds(series["date"].max()))
    preds.to_parquet(a.data / "backtest.parquet")
    scores = forecast.score(preds)
    print("Estimating price elasticities ...")
    el = pricing.elasticities(pricing.load_weekly(con))
    inv = inventory.compare(preds, models=("Seasonal naive", "Weekday profile", scores["wape"].idxmin()))
    res = report.write(a.reports, m, scores, preds, el, inv)
    print(json.dumps(res["forecast"], indent=2))
    print(json.dumps(res["inventory"], indent=2))
    pd.set_option("display.width", 120)


if __name__ == "__main__":
    main()
