"""Download the M5 data and load it into a local DuckDB warehouse.

The M5 dataset is daily unit sales for 3,049 Walmart products in 10 stores
across California, Texas and Wisconsin (2011-01-29 to 2016-05-22), with weekly
shelf prices and a calendar of holidays, sporting events and SNAP days.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import duckdb

M5_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
RAW_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")


def download(raw_dir: Path) -> None:
    """Fetch and unzip the M5 files into raw_dir unless they are already there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / f).exists() for f in RAW_FILES):
        return
    archive = raw_dir / "m5.zip"
    if not archive.exists():
        print(f"Downloading {M5_URL} ...")
        urllib.request.urlretrieve(M5_URL, archive)
    with zipfile.ZipFile(archive) as z:
        for name in RAW_FILES:
            z.extract(name, raw_dir)


def build_warehouse(raw_dir: Path, db_path: Path) -> None:
    """Load the raw CSVs into DuckDB and build the long sales fact table.

    sales_train_evaluation.csv is wide (one column per day). It is unpivoted
    into one row per item, store and day, joined to the calendar and to the
    week's shelf price so every row carries revenue. Days before an item was
    first priced in a store are dropped: the item was not on the shelf yet.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("SET enable_progress_bar = false")
    raw = raw_dir.as_posix()
    con.execute(f"""
        CREATE OR REPLACE TABLE calendar AS
        SELECT CAST(date AS DATE) AS date, wm_yr_wk, weekday, month, year,
               'd_' || CAST(row_number() OVER (ORDER BY date) AS VARCHAR) AS d,
               event_name_1, event_type_1, event_name_2, event_type_2,
               snap_CA, snap_TX, snap_WI
        FROM read_csv_auto('{raw}/calendar.csv', nullstr='NA');

        CREATE OR REPLACE TABLE prices AS
        SELECT * FROM read_csv_auto('{raw}/sell_prices.csv');

        CREATE OR REPLACE TABLE items AS
        SELECT DISTINCT item_id, dept_id, cat_id
        FROM read_csv_auto('{raw}/sales_train_evaluation.csv');

        CREATE OR REPLACE TABLE stores AS
        SELECT DISTINCT store_id, state_id
        FROM read_csv_auto('{raw}/sales_train_evaluation.csv');
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE sales AS
        WITH wide AS (
            SELECT * EXCLUDE (dept_id, cat_id, state_id)
            FROM read_csv_auto('{raw}/sales_train_evaluation.csv')
        ),
        long AS (
            UNPIVOT wide ON COLUMNS('^d_') INTO NAME d VALUE units
        )
        SELECT l.store_id, l.item_id, c.date, c.wm_yr_wk,
               CAST(l.units AS INTEGER) AS units,
               p.sell_price,
               l.units * p.sell_price AS revenue
        FROM long l
        JOIN calendar c USING (d)
        JOIN prices p
          ON p.store_id = l.store_id AND p.item_id = l.item_id
         AND p.wm_yr_wk = c.wm_yr_wk
    """)
    con.close()


def connect(db_path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("SET enable_progress_bar = false")
    return con
