import numpy as np
import pandas as pd

from walmart_analytics.inventory import compare


def test_better_forecast_needs_less_buffer():
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2016-01-01", periods=84)
    y = rng.poisson(100, 84).astype(float)
    for model, noise in (("Seasonal naive", 20), ("LightGBM", 5)):
        pred = y + rng.normal(0, noise, 84)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "store_id": "CA_1",
                    "dept_id": "FOODS_1",
                    "date": dates,
                    "units": y,
                    "pred": pred,
                    "avg_price": 3.0,
                    "fold": np.repeat([1, 2, 3], 28),
                }
            )
        )
    out = compare(pd.concat(rows), models=("Seasonal naive", "LightGBM"))
    assert out.loc["LightGBM", "avg_daily_buffer_value"] < out.loc["Seasonal naive", "avg_daily_buffer_value"]
