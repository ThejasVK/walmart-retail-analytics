import pandas as pd


def test_unpivot_keeps_only_priced_days(warehouse, raw_dir):
    n = warehouse.sql("SELECT COUNT(*) FROM sales").fetchone()[0]
    # 6 item-stores x 70 days, minus 14 unpriced days for one of them.
    assert n == 6 * 70 - 14


def test_units_and_revenue_match_raw(warehouse, raw_dir):
    wide = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    raw_units = wide.filter(like="d_").to_numpy().sum()
    got = warehouse.sql("SELECT SUM(units) FROM sales").fetchone()[0]
    assert got == raw_units  # unpriced days had zero units in the fixture
    bad = warehouse.sql("SELECT COUNT(*) FROM sales WHERE abs(revenue - units * sell_price) > 1e-9").fetchone()[0]
    assert bad == 0


def test_calendar_day_index(warehouse):
    first = warehouse.sql("SELECT d, date FROM calendar ORDER BY date LIMIT 1").fetchone()
    assert first[0] == "d_1" and str(first[1]) == "2011-01-29"
