"""Run the named SQL queries in sql/marts.sql."""

from __future__ import annotations

import re
from importlib import resources

import duckdb
import pandas as pd


def queries() -> dict[str, str]:
    text = resources.files("walmart_analytics").joinpath("sql/marts.sql").read_text()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def run_all(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    return {name: con.sql(sql).df() for name, sql in queries().items()}
