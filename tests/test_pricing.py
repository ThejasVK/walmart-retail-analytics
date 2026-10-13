import numpy as np
import pandas as pd

from walmart_analytics.pricing import _demean, elasticities


def test_demean_removes_both_fixed_effects():
    df = pd.DataFrame({"a": list("xxyy"), "b": list("pqpq"), "v": [1.0, 3.0, 5.0, 7.0]})
    out = _demean(df, ["v"], [["a"], ["b"]])
    assert np.allclose(out["v"], 0)


def test_recovers_a_known_elasticity_despite_confounders():
    rng = np.random.default_rng(1)
    rows = []
    for item in range(60):
        base = rng.uniform(1, 4)  # item-level demand
        p0 = rng.uniform(1, 10)  # item-level price (correlated with nothing)
        for week in range(40):
            season = 0.5 * np.sin(week / 5)  # shared weekly shock
            price = p0 * rng.choice([0.8, 0.9, 1.0, 1.0, 1.1])
            mu = np.exp(base + season - 1.5 * np.log(price / p0))
            rows.append(
                {
                    "store_id": "CA_1",
                    "item_id": f"i{item}",
                    "dept_id": "FOODS_1",
                    "wm_yr_wk": week,
                    "units": np.expm1(np.log1p(mu)),
                    "price": price,
                }
            )
    est = elasticities(pd.DataFrame(rows))
    assert abs(est.loc["FOODS_1", "elasticity"] - (-1.5)) < 0.15
