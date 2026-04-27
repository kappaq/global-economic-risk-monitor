"""DuckDB read/write layer. All database access goes through this module."""

import os
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "economic_risk.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(SCHEMA_PATH.read_text())
    return conn


def upsert_indicators(df: pd.DataFrame) -> None:
    """Insert or replace indicator rows. df must have: country_code, series_id, date, value."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO indicators (country_code, series_id, date, value)
            SELECT country_code, series_id, date, value FROM df
        """)


def upsert_model_outputs(df: pd.DataFrame) -> None:
    """Insert or replace model output rows."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_outputs
                (country_code, model_name, date, recession_prob, inflation_state, inflation_probs, composite_risk)
            SELECT country_code, model_name, date, recession_prob, inflation_state, inflation_probs, composite_risk
            FROM df
        """)


def read_indicators(country_code: str = None, series_id: str = None) -> pd.DataFrame:
    query = "SELECT * FROM indicators WHERE 1=1"
    params = []
    if country_code:
        query += " AND country_code = ?"
        params.append(country_code)
    if series_id:
        query += " AND series_id = ?"
        params.append(series_id)
    query += " ORDER BY date"
    with get_connection() as conn:
        return conn.execute(query, params).df()


def read_model_outputs(country_code: str = None, model_name: str = None) -> pd.DataFrame:
    query = "SELECT * FROM model_outputs WHERE 1=1"
    params = []
    if country_code:
        query += " AND country_code = ?"
        params.append(country_code)
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    query += " ORDER BY date"
    with get_connection() as conn:
        return conn.execute(query, params).df()


def latest_indicator_date(country_code: str = "USA") -> pd.Timestamp | None:
    with get_connection() as conn:
        result = conn.execute(
            "SELECT MAX(date) FROM indicators WHERE country_code = ?", [country_code]
        ).fetchone()
    val = result[0] if result else None
    return pd.Timestamp(val) if val else None


def data_is_stale(max_age_hours: int = 24) -> bool:
    latest = latest_indicator_date()
    if latest is None:
        return True
    age = pd.Timestamp.now() - latest
    return age.total_seconds() / 3600 > max_age_hours
