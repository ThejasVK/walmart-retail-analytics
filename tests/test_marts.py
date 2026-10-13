from walmart_analytics import marts


def test_every_named_query_runs(warehouse):
    out = marts.run_all(warehouse)
    assert {"revenue_by_year", "category_mix", "store_scorecard", "snap_effect", "department_mix"} <= set(out)
    mix = out["category_mix"]
    assert abs(mix["share"].sum() - 1) < 1e-9
