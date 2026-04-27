"""Unit tests for models/composite.py — composite risk scorer."""

import pytest
import pandas as pd
import numpy as np

from models.composite import _zscore_to_risk, score_country
import data.store as store_module


# ── _zscore_to_risk ───────────────────────────────────────────────────────────

def test_zscore_to_risk_output_range():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _zscore_to_risk(s)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_zscore_to_risk_monotone():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _zscore_to_risk(s)
    assert result.iloc[-1] > result.iloc[0]


def test_zscore_to_risk_inverted_reverses_order():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    normal = _zscore_to_risk(s, invert=False)
    inverted = _zscore_to_risk(s, invert=True)
    assert inverted.iloc[-1] < inverted.iloc[0]
    assert abs(normal.iloc[0] - inverted.iloc[-1]) < 1e-9


def test_zscore_to_risk_constant_series_returns_half():
    s = pd.Series([7.0, 7.0, 7.0])
    result = _zscore_to_risk(s)
    assert all(result == 0.5)


def test_zscore_to_risk_no_nan_in_output():
    s = pd.Series([np.nan, 1.0, 2.0, 3.0])
    result = _zscore_to_risk(s.dropna())
    assert not result.isna().any()


# ── score_country ─────────────────────────────────────────────────────────────

def _gbr_indicators(gdp_vals, cpi_vals, unemp_vals):
    """Build a 3-year GBR indicator DataFrame (minimum needed by score_country)."""
    years = ["2019-01-01", "2020-01-01", "2021-01-01"]
    rows = []
    for sid, vals in [
        ("NY.GDP.MKTP.KD.ZG", gdp_vals),
        ("FP.CPI.TOTL.ZG", cpi_vals),
        ("SL.UEM.TOTL.ZS", unemp_vals),
    ]:
        for date, val in zip(years, vals):
            rows.append({"country_code": "GBR", "series_id": sid,
                         "date": pd.Timestamp(date), "value": val})
    return pd.DataFrame(rows)


def test_score_country_returns_expected_columns(temp_db):
    from data.store import upsert_indicators
    upsert_indicators(_gbr_indicators([2.1, 1.5, 0.8], [3.2, 5.1, 2.4], [4.0, 4.2, 5.1]))

    result = score_country("GBR")
    assert not result.empty
    required_cols = {"country_code", "composite_risk", "recession_prob", "inflation_state"}
    assert required_cols.issubset(result.columns)


def test_score_country_composite_in_unit_interval(temp_db):
    from data.store import upsert_indicators
    upsert_indicators(_gbr_indicators([2.0, -4.0, 1.0], [1.5, 6.0, 3.2], [3.1, 5.2, 4.0]))

    result = score_country("GBR")
    assert (result["composite_risk"] >= 0).all()
    assert (result["composite_risk"] <= 1).all()


def test_score_country_empty_when_data_missing(temp_db):
    result = score_country("JPN")
    assert result.empty


def test_score_country_inflation_state_labels(temp_db):
    from data.store import upsert_indicators
    # CPI: 1.0 (low), 3.0 (moderate), 8.0 (high) — all three regimes present
    upsert_indicators(_gbr_indicators([2.0, 1.5, 1.2], [1.0, 3.0, 8.0], [4.0, 4.2, 4.5]))

    result = score_country("GBR")
    valid_labels = {"low", "moderate", "high"}
    assert set(result["inflation_state"].unique()).issubset(valid_labels)
