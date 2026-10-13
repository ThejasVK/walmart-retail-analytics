"""What better forecasts are worth in inventory.

A store that orders to a forecast holds safety stock on top of it to cover
forecast misses. For a target service level, the buffer is the matching
quantile of past errors. A more accurate forecast needs a smaller buffer for
the same service, and the difference is inventory value that does not have to
sit on the shelf.

The buffer for each store and department is calibrated on the earlier backtest
folds and then tested, out of sample, on the last fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safety_stock_test(
    preds: pd.DataFrame, model: str, service: float = 0.95, calib_folds=(1, 2), test_fold: int = 3
) -> dict:
    p = preds[preds["model"] == model].copy()
    p["err"] = p["units"] - p["pred"]
    cal = p[p["fold"].isin(calib_folds)]
    buffer = cal.groupby(["store_id", "dept_id"])["err"].quantile(service).clip(lower=0)
    t = p[p["fold"] == test_fold].copy()
    t["buffer"] = buffer.reindex(pd.MultiIndex.from_frame(t[["store_id", "dept_id"]])).to_numpy()
    t["stock"] = t["pred"] + t["buffer"]
    covered = (t["units"] <= t["stock"]).mean()
    unit_fill = np.minimum(t["units"], t["stock"]).sum() / t["units"].sum()
    days = t["date"].nunique()
    return {
        "model": model,
        "target_service": service,
        "days_covered": float(covered),
        "unit_fill_rate": float(unit_fill),
        "avg_daily_buffer_units": float(t["buffer"].sum() / days),
        "avg_daily_buffer_value": float((t["buffer"] * t["avg_price"]).sum() / days),
        "avg_daily_excess_value": float((np.clip(t["stock"] - t["units"], 0, None) * t["avg_price"]).sum() / days),
    }


def compare(
    preds: pd.DataFrame, models=("Seasonal naive", "Weekday profile", "LightGBM"), service: float = 0.95
) -> pd.DataFrame:
    return pd.DataFrame([safety_stock_test(preds, m, service) for m in models]).set_index("model")
