"""28-day demand forecasts for every store and department.

Each forecast is made 28 days ahead from a fixed origin, so every feature uses
data that is at least 28 days old at prediction time. Models are scored with a
rolling-origin backtest over the last three 28-day windows of the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZON = 28
LAGS = (28, 35, 42, 49, 364)
NUMERIC = [f"lag_{k}" for k in LAGS] + [
    "roll7_28",
    "roll28_28",
    "roll91_28",
    "price_idx",
    "weekday",
    "month",
    "mday",
    "snap",
    "event",
    "event_next",
]
CATEGORICAL = ["store_id", "dept_id"]


def load_series(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Daily units, revenue and average shelf price per store and department."""
    return con.sql("""
        SELECT s.store_id, i.dept_id, s.date,
               SUM(s.units) AS units,
               SUM(s.revenue) AS revenue,
               AVG(s.sell_price) AS avg_price,
               ANY_VALUE(CASE st.state_id WHEN 'CA' THEN c.snap_CA
                                          WHEN 'TX' THEN c.snap_TX
                                          ELSE c.snap_WI END) AS snap,
               ANY_VALUE((c.event_name_1 IS NOT NULL)::INT) AS event,
               ANY_VALUE(c.weekday) AS weekday_name
        FROM sales s
        JOIN items i USING (item_id)
        JOIN stores st USING (store_id)
        JOIN calendar c ON c.date = s.date
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).df()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "dept_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby(["store_id", "dept_id"], sort=False)
    for k in LAGS:
        df[f"lag_{k}"] = g["units"].shift(k)
    shifted = g["units"].shift(HORIZON)
    key = [df["store_id"], df["dept_id"]]
    for w in (7, 28, 91):
        df[f"roll{w}_28"] = shifted.groupby(key).transform(lambda s, w=w: s.rolling(w).mean())
    ref = g["avg_price"].transform(lambda s: s.shift(HORIZON).rolling(364, min_periods=28).mean())
    df["price_idx"] = df["avg_price"] / ref
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["mday"] = df["date"].dt.day
    df["event_next"] = g["event"].shift(-1).fillna(0)
    return df


@dataclass
class Fold:
    origin: pd.Timestamp  # last day of training data

    @property
    def test_days(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.origin + pd.Timedelta(days=1), self.origin + pd.Timedelta(days=HORIZON)


def make_folds(last_day: pd.Timestamp, n: int = 3) -> list[Fold]:
    return [Fold(last_day - pd.Timedelta(days=HORIZON * k)) for k in range(n, 0, -1)]


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Repeat the last observed week, matched on weekday."""
    last_week = train[train["date"] > train["date"].max() - pd.Timedelta(days=7)]
    m = last_week.set_index(["store_id", "dept_id", "weekday"])["units"]
    idx = pd.MultiIndex.from_frame(test[["store_id", "dept_id", "weekday"]])
    return m.reindex(idx).to_numpy(dtype=float)


def weekday_profile(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Mean of the last 28 days, scaled by each series' weekday pattern."""
    recent = train[train["date"] > train["date"].max() - pd.Timedelta(days=HORIZON)]
    year = train[train["date"] > train["date"].max() - pd.Timedelta(days=364)]
    level = recent.groupby(["store_id", "dept_id"])["units"].mean()
    prof = (
        year.groupby(["store_id", "dept_id", "weekday"])["units"].mean()
        / year.groupby(["store_id", "dept_id"])["units"].mean()
    )
    t = test[["store_id", "dept_id", "weekday"]]
    lv = level.reindex(pd.MultiIndex.from_frame(t[["store_id", "dept_id"]])).to_numpy()
    pf = prof.reindex(pd.MultiIndex.from_frame(t)).to_numpy()
    return lv * pf


def _xy(df: pd.DataFrame):
    return df[NUMERIC + CATEGORICAL], df["units"].to_numpy(dtype=float)


def ridge_model():
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return make_pipeline(pre, Ridge(alpha=1.0))


def fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, seed: int = 7) -> np.ndarray:
    if name == "Seasonal naive":
        return seasonal_naive(train, test)
    if name == "Weekday profile":
        return weekday_profile(train, test)
    tr = train.dropna(subset=NUMERIC)
    X, y = _xy(tr)
    Xt, _ = _xy(test)
    if name == "Linear regression":
        m = ridge_model().fit(X, y)
        return m.predict(Xt)
    Xc, Xtc = X.copy(), Xt.copy()
    for c in CATEGORICAL:
        cats = pd.Categorical(train[c]).categories
        Xc[c] = pd.Categorical(Xc[c], categories=cats)
        Xtc[c] = pd.Categorical(Xtc[c], categories=cats)
    if name == "Random forest":
        Xc = pd.get_dummies(Xc, columns=CATEGORICAL)
        Xtc = pd.get_dummies(Xtc, columns=CATEGORICAL).reindex(columns=Xc.columns, fill_value=0)
        m = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed
        ).fit(Xc, y)
        return m.predict(Xtc)
    if name == "LightGBM":
        m = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.1,
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        m.fit(Xc, y)
        return m.predict(Xtc)
    raise ValueError(name)


MODELS = ["Seasonal naive", "Weekday profile", "Linear regression", "Random forest", "LightGBM"]


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def backtest(df: pd.DataFrame, folds: list[Fold], models=MODELS) -> pd.DataFrame:
    """Predictions for every model and fold, one row per series-day."""
    out = []
    for i, fold in enumerate(folds, start=1):
        start, end = fold.test_days
        train = df[df["date"] <= fold.origin]
        test = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        base = test[["store_id", "dept_id", "date", "units", "revenue", "avg_price"]].copy()
        base["fold"] = i
        for name in models:
            pred = np.clip(fit_predict(name, train, test), 0, None)
            out.append(base.assign(model=name, pred=pred))
    return pd.concat(out, ignore_index=True)


def score(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold), g in preds.groupby(["model", "fold"]):
        rows.append({"model": model, "fold": fold, "wape": wape(g["units"].to_numpy(), g["pred"].to_numpy())})
    s = pd.DataFrame(rows).pivot(index="model", columns="fold", values="wape")
    s.columns = [f"fold_{c}" for c in s.columns]
    s["wape"] = s.mean(axis=1)
    s["accuracy"] = 1 - s["wape"]
    return s.sort_values("wape")
