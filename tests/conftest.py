import json
import pytest
import pandas as pd
import data.store as store_module


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Redirect all DB access to a throwaway temp file."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(store_module, "DB_PATH", db_path)
    return db_path


@pytest.fixture
def seeded_db(temp_db):
    """Temp DB pre-populated with a minimal set of indicators and model outputs."""
    from data.store import upsert_indicators, upsert_model_outputs

    indicators = pd.DataFrame({
        "country_code": ["USA", "USA", "USA", "USA", "GBR", "GBR"],
        "series_id":    ["T10Y2Y", "T10Y2Y", "UNRATE", "USREC",
                         "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG"],
        "date": [
            pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01"),
            pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01"),
        ],
        "value": [0.5, 0.8, 3.5, 0.0, 2.1, 4.2],
    })
    upsert_indicators(indicators)

    model_outputs = pd.DataFrame({
        "country_code":   ["USA", "USA"],
        "model_name":     ["recession", "recession"],
        "date":           [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
        "recession_prob": [0.30, 0.45],
        "inflation_state": [None, None],
        "inflation_probs": [None, None],
        "composite_risk": [0.30, 0.45],
    })
    upsert_model_outputs(model_outputs)
    return temp_db
