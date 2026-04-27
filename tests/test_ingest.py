"""Unit tests for data/ingest.py — World Bank response parsing."""

import pytest
import pandas as pd
import data.ingest as ingest_module
from data.ingest import fetch_worldbank_indicator


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _make_entry(country_id, iso3, date, value):
    return {
        "country": {"id": country_id},
        "countryiso3code": iso3,
        "date": date,
        "value": value,
    }


# ── fetch_worldbank_indicator ─────────────────────────────────────────────────

def test_parses_valid_entries(monkeypatch):
    payload = [
        {"page": 1, "pages": 1, "per_page": 500, "total": 2},
        [
            _make_entry("US", "USA", "2022", 2.1),
            _make_entry("GB", "GBR", "2022", 1.8),
        ],
    ]
    monkeypatch.setattr(ingest_module.requests, "get",
                        lambda url, timeout: _MockResponse(payload))

    df = fetch_worldbank_indicator("FP.CPI.TOTL.ZG", ["USA", "GBR"])
    assert len(df) == 2
    assert set(df["country_code"].tolist()) == {"USA", "GBR"}
    assert df["series_id"].iloc[0] == "FP.CPI.TOTL.ZG"


def test_skips_null_values(monkeypatch):
    payload = [
        {"page": 1},
        [
            _make_entry("US", "USA", "2022", 2.1),
            _make_entry("US", "USA", "2021", None),  # should be skipped
        ],
    ]
    monkeypatch.setattr(ingest_module.requests, "get",
                        lambda url, timeout: _MockResponse(payload))

    df = fetch_worldbank_indicator("FP.CPI.TOTL.ZG", ["USA"])
    assert len(df) == 1
    assert df.iloc[0]["value"] == 2.1


def test_returns_empty_df_on_empty_payload(monkeypatch):
    payload = [{"page": 1}, []]
    monkeypatch.setattr(ingest_module.requests, "get",
                        lambda url, timeout: _MockResponse(payload))

    df = fetch_worldbank_indicator("FP.CPI.TOTL.ZG", ["USA"])
    assert df.empty


def test_returns_empty_df_on_malformed_payload(monkeypatch):
    monkeypatch.setattr(ingest_module.requests, "get",
                        lambda url, timeout: _MockResponse([{}]))

    df = fetch_worldbank_indicator("FP.CPI.TOTL.ZG", ["USA"])
    assert df.empty


def test_date_parsed_as_timestamp(monkeypatch):
    payload = [
        {"page": 1},
        [_make_entry("US", "USA", "2020", 1.5)],
    ]
    monkeypatch.setattr(ingest_module.requests, "get",
                        lambda url, timeout: _MockResponse(payload))

    df = fetch_worldbank_indicator("NY.GDP.MKTP.KD.ZG", ["USA"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.iloc[0]["date"] == pd.Timestamp("2020-01-01")
