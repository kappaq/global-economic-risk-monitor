"""Unit tests for data/store.py — DB read/write layer."""

import os
import time
import pytest
import pandas as pd

import data.store as store_module
from data.store import (
    upsert_indicators, read_indicators, read_indicators_multi,
    upsert_model_outputs, read_model_outputs, data_is_stale,
)


# ── data_is_stale ─────────────────────────────────────────────────────────────

def test_stale_when_db_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "no_such.db")
    assert data_is_stale() is True


def test_not_stale_when_db_just_written(monkeypatch, tmp_path):
    db = tmp_path / "fresh.db"
    db.touch()
    monkeypatch.setattr(store_module, "DB_PATH", db)
    assert data_is_stale() is False


def test_stale_when_db_is_old(monkeypatch, tmp_path):
    db = tmp_path / "old.db"
    db.touch()
    old_mtime = time.time() - 25 * 3600
    os.utime(db, (old_mtime, old_mtime))
    monkeypatch.setattr(store_module, "DB_PATH", db)
    assert data_is_stale() is True


# ── indicators round-trip ─────────────────────────────────────────────────────

def test_upsert_and_read_indicators(temp_db):
    df = pd.DataFrame({
        "country_code": ["USA", "USA"],
        "series_id":    ["T10Y2Y", "T10Y2Y"],
        "date":         [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
        "value":        [0.5, 0.8],
    })
    upsert_indicators(df)
    result = read_indicators(country_code="USA", series_id="T10Y2Y")
    assert len(result) == 2
    assert set(result["value"].tolist()) == {0.5, 0.8}


def test_upsert_is_idempotent(temp_db):
    df = pd.DataFrame({
        "country_code": ["USA"],
        "series_id":    ["UNRATE"],
        "date":         [pd.Timestamp("2023-01-01")],
        "value":        [3.5],
    })
    upsert_indicators(df)
    upsert_indicators(df)  # second insert should replace, not duplicate
    result = read_indicators(country_code="USA", series_id="UNRATE")
    assert len(result) == 1


def test_read_indicators_filters_by_country(seeded_db):
    result = read_indicators(country_code="USA")
    assert all(result["country_code"] == "USA")


def test_read_indicators_multi_batch(seeded_db):
    series = read_indicators_multi("USA", ["T10Y2Y", "UNRATE"])
    assert "T10Y2Y" in series
    assert "UNRATE" in series
    assert len(series["T10Y2Y"]) == 2
    assert len(series["UNRATE"]) == 1


def test_read_indicators_multi_missing_series_absent(seeded_db):
    series = read_indicators_multi("USA", ["T10Y2Y", "NONEXISTENT"])
    assert "T10Y2Y" in series
    assert "NONEXISTENT" not in series


# ── model_outputs round-trip ──────────────────────────────────────────────────

def test_upsert_and_read_model_outputs(temp_db):
    df = pd.DataFrame({
        "country_code":   ["USA"],
        "model_name":     ["recession"],
        "date":           [pd.Timestamp("2023-06-01")],
        "recession_prob": [0.62],
        "inflation_state": [None],
        "inflation_probs": [None],
        "composite_risk": [0.62],
    })
    upsert_model_outputs(df)
    result = read_model_outputs(country_code="USA", model_name="recession")
    assert len(result) == 1
    assert abs(result.iloc[0]["recession_prob"] - 0.62) < 1e-6


def test_model_outputs_upsert_replaces_existing(temp_db):
    base = pd.DataFrame({
        "country_code": ["USA"], "model_name": ["recession"],
        "date": [pd.Timestamp("2023-01-01")],
        "recession_prob": [0.3], "inflation_state": [None],
        "inflation_probs": [None], "composite_risk": [0.3],
    })
    updated = base.copy()
    updated["recession_prob"] = 0.9
    upsert_model_outputs(base)
    upsert_model_outputs(updated)
    result = read_model_outputs(country_code="USA", model_name="recession")
    assert len(result) == 1
    assert abs(result.iloc[0]["recession_prob"] - 0.9) < 1e-6
