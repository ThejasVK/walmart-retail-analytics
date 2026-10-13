import numpy as np
import pandas as pd

from walmart_analytics import forecast as f


def _toy(days=120):
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    wk = np.tile([10, 10, 10, 10, 12, 20, 20], days // 7 + 1)[:days].astype(float)
    df = pd.DataFrame(
        {
            "store_id": "CA_1",
            "dept_id": "FOODS_1",
            "date": dates,
            "units": wk,
            "revenue": wk * 2,
            "avg_price": 2.0,
            "snap": 0,
            "event": 0,
        }
    )
    return f.add_features(df)


def test_wape():
    assert f.wape(np.array([10, 10]), np.array([8, 12])) == 0.2


def test_features_never_look_inside_the_horizon():
    df = _toy()
    row = df.iloc[100]
    assert row["lag_28"] == df.iloc[72]["units"]
    assert row["roll7_28"] == df.iloc[66:73]["units"].mean()


def test_seasonal_naive_is_exact_on_a_pure_weekly_pattern():
    df = _toy()
    fold = f.make_folds(df["date"].max(), n=1)[0]
    train = df[df["date"] <= fold.origin]
    test = df[df["date"] > fold.origin]
    pred = f.seasonal_naive(train, test)
    assert f.wape(test["units"].to_numpy(), pred) == 0


def test_folds_tile_the_end_of_the_data():
    last = pd.Timestamp("2016-05-22")
    folds = f.make_folds(last, n=3)
    assert folds[-1].test_days[1] == last
    assert folds[1].test_days[0] == folds[0].test_days[1] + pd.Timedelta(days=1)
