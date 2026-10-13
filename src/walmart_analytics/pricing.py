"""Price elasticity by department.

Weekly units and shelf price per item and store. The estimate uses two sets of
fixed effects, removed by alternating demeaning:

* item x store, so the comparison is always the same product in the same store
  at a different price, not cheap products against expensive ones;
* store x department x week, so seasonality, holidays and store traffic in a
  given week do not pass for price effects.

What is left is how unit sales move when one item's price moves relative to
its own history and to the rest of its department that week.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd


def load_weekly(con: duckdb.DuckDBPyConnection, min_avg_units: float = 3.0) -> pd.DataFrame:
    """Item-store-weeks for items that sell often enough to measure."""
    return con.sql(f"""
        WITH w AS (
            SELECT s.store_id, s.item_id, i.dept_id, s.wm_yr_wk,
                   SUM(s.units) AS units, ANY_VALUE(s.sell_price) AS price,
                   COUNT(*) AS n_days
            FROM sales s JOIN items i USING (item_id)
            GROUP BY 1, 2, 3, 4
        )
        SELECT * FROM w
        WHERE n_days = 7
          AND (store_id, item_id) IN (
              SELECT store_id, item_id FROM w GROUP BY 1, 2
              HAVING AVG(units) >= {min_avg_units} AND COUNT(DISTINCT price) > 1)
    """).df()


def _demean(
    df: pd.DataFrame, cols: list[str], groups: list[list[str]], iters: int = 50, tol: float = 1e-8
) -> pd.DataFrame:
    """Remove several sets of fixed effects by alternating projections."""
    codes = [df.groupby(g, sort=False).ngroup().to_numpy() for g in groups]
    counts = [np.bincount(c) for c in codes]
    out = {}
    for col in cols:
        v = df[col].to_numpy(dtype=float).copy()
        for _ in range(iters):
            shift = 0.0
            for c, n in zip(codes, counts):
                m = np.bincount(c, weights=v) / n
                v -= m[c]
                shift = max(shift, float(np.abs(m).max()))
            if shift < tol:
                break
        out[col] = v
    return pd.DataFrame(out, index=df.index)


def elasticities(weekly: pd.DataFrame) -> pd.DataFrame:
    d = weekly.copy()
    d["ly"] = np.log1p(d["units"])
    d["lp"] = np.log(d["price"])
    dm = _demean(d, ["ly", "lp"], [["store_id", "item_id"], ["store_id", "dept_id", "wm_yr_wk"]])
    d["y"], d["x"] = dm["ly"], dm["lp"]
    d = d.sort_values(["store_id", "item_id", "wm_yr_wk"])
    d["moved"] = d.groupby(["store_id", "item_id"])["price"].diff().abs().gt(0)
    rows = []
    for dept, g in d.groupby("dept_id"):
        sxx = float((g["x"] ** 2).sum())
        if sxx == 0:
            continue
        b = float((g["x"] * g["y"]).sum() / sxx)
        resid = g["y"] - b * g["x"]
        # Standard error clustered by item-store, since weeks of one item are not independent.
        cl = (g["x"] * resid).groupby([g["store_id"], g["item_id"]]).sum()
        se = float(np.sqrt((cl**2).sum()) / sxx)
        rows.append(
            {
                "dept_id": dept,
                "elasticity": b,
                "se": se,
                "item_stores": g.groupby(["store_id", "item_id"]).ngroups,
                "weeks": len(g),
                "price_changes": int(g["moved"].sum()),
            }
        )
    r = pd.DataFrame(rows).set_index("dept_id").sort_values("elasticity")
    r["units_10pct_off"] = 0.9 ** r["elasticity"] - 1
    r["revenue_10pct_off"] = 0.9 ** (1 + r["elasticity"]) - 1
    return r
