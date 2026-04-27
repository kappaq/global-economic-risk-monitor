"""DuckDB read/write layer. All database access goes through this module."""

import time
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "economic_risk.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_SCHEMA_SQL: str = SCHEMA_PATH.read_text()


def get_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(_SCHEMA_SQL)
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


def read_indicators_multi(country_code: str, series_ids: list[str]) -> dict[str, pd.Series]:
    """Fetch multiple series in one query and return as {series_id: pd.Series}."""
    placeholders = ", ".join("?" for _ in series_ids)
    query = f"""
        SELECT series_id, date, value FROM indicators
        WHERE country_code = ? AND series_id IN ({placeholders})
        ORDER BY date
    """
    with get_connection() as conn:
        df = conn.execute(query, [country_code] + series_ids).df()
    result = {}
    for sid, grp in df.groupby("series_id"):
        s = grp.set_index("date")["value"]
        s.index = pd.to_datetime(s.index)
        result[sid] = s.sort_index()
    return result


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
    if not DB_PATH.exists():
        return True
    age_seconds = time.time() - DB_PATH.stat().st_mtime
    return age_seconds / 3600 > max_age_hours
