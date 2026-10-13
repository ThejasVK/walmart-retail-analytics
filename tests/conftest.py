"""A tiny M5-shaped dataset: 2 stores, 3 items, 70 days."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    days = 70
    dates = pd.date_range("2011-01-29", periods=days, freq="D")
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + np.arange(days) // 7,
            "weekday": dates.day_name(),
            "wday": np.arange(days) % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": ["SuperBowl" if i == 8 else "NA" for i in range(days)],
            "event_type_1": ["Sporting" if i == 8 else "NA" for i in range(days)],
            "event_name_2": "NA",
            "event_type_2": "NA",
            "snap_CA": (np.arange(days) % 30 < 10).astype(int),
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    cal.to_csv(tmp_path / "calendar.csv", index=False)
    items = [
        ("FOODS_1_001", "FOODS_1", "FOODS"),
        ("FOODS_1_002", "FOODS_1", "FOODS"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"),
    ]
    stores = [("CA_1", "CA"), ("TX_1", "TX")]
    rows, prices = [], []
    for store, state in stores:
        for item, dept, cat in items:
            # HOBBIES_1_001 only goes on sale in week 3 in TX_1.
            first_week = 2 if (store == "TX_1" and item.startswith("HOBBIES")) else 0
            units = rng.poisson(5, days)
            units[: first_week * 7] = 0
            rows.append(
                {
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                    **{f"d_{i + 1}": int(u) for i, u in enumerate(units)},
                }
            )
            for w in range(first_week, days // 7):
                prices.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": 11101 + w,
                        "sell_price": 2.0 if w % 2 == 0 else 2.5,
                    }
                )
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices).to_csv(tmp_path / "sell_prices.csv", index=False)
    return tmp_path


@pytest.fixture
def warehouse(raw_dir, tmp_path):
    from walmart_analytics.data import build_warehouse, connect

    db = tmp_path / "w.duckdb"
    build_warehouse(raw_dir, db)
    return connect(db)
