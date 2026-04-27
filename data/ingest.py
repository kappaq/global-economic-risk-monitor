"""
Data ingestion pipeline.
Sources: FRED API (US macro indicators) + World Bank API (multi-country).
Pipeline is idempotent — safe to re-run at any time.
"""

import os
import requests
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from fredapi import Fred

from data.store import upsert_indicators

load_dotenv()

FRED_START = "1980-01-01"

FRED_SERIES = {
    "USREC":    "NBER Recession Indicator",
    "T10Y2Y":   "10Y-2Y Treasury Yield Spread",
    "T10Y3M":   "10Y-3M Treasury Yield Spread",
    "UNRATE":   "Civilian Unemployment Rate",
    "INDPRO":   "Industrial Production Index",
    "PAYEMS":   "Nonfarm Payrolls",
    "UMCSENT":  "UMich Consumer Sentiment",
    "CPIAUCSL": "CPI All Urban Consumers",
    "CPILFESL": "Core CPI (ex food & energy)",
    "PCEPI":    "PCE Price Index",
    "MICH":     "1Y Inflation Expectations",
    "VIXCLS":   "CBOE VIX",
}

WORLDBANK_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP Growth Rate",
    "FP.CPI.TOTL.ZG":    "CPI Inflation Rate",
    "SL.UEM.TOTL.ZS":    "Unemployment Rate",
}

COUNTRIES = ["USA", "GBR", "DEU", "JPN"]


def fetch_fred_series() -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set. Copy .env.example to .env and add your key.")
    fred = Fred(api_key=api_key)
    rows = []
    for series_id in FRED_SERIES:
        print(f"  Fetching FRED: {series_id}")
        try:
            s = fred.get_series(series_id, observation_start=FRED_START)
            s = s.dropna().reset_index()
            s.columns = ["date", "value"]
            s["country_code"] = "USA"
            s["series_id"] = series_id
            rows.append(s[["country_code", "series_id", "date", "value"]])
        except Exception as e:
            print(f"  WARNING: Could not fetch {series_id}: {e}")
    return pd.concat(rows, ignore_index=True)


def fetch_worldbank_indicator(indicator_code: str, countries: list[str]) -> pd.DataFrame:
    iso2_map = {"USA": "US", "GBR": "GB", "DEU": "DE", "JPN": "JP"}
    iso2_codes = ";".join(iso2_map[c] for c in countries)
    url = (
        f"https://api.worldbank.org/v2/country/{iso2_codes}/indicator/{indicator_code}"
        f"?format=json&per_page=500&mrv=50"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if len(payload) < 2 or not payload[1]:
        return pd.DataFrame()

    iso3_map = {"US": "USA", "GB": "GBR", "DE": "DEU", "JP": "JPN"}
    rows = []
    for entry in payload[1]:
        if entry["value"] is None:
            continue
        iso2 = entry["countryiso3code"][:2] if len(entry["countryiso3code"]) == 3 else entry["country"]["id"]
        country_code = iso3_map.get(entry["country"]["id"], entry["countryiso3code"])
        rows.append({
            "country_code": country_code,
            "series_id": indicator_code,
            "date": pd.Timestamp(f"{entry['date']}-01-01"),
            "value": float(entry["value"]),
        })
    return pd.DataFrame(rows)


def fetch_worldbank_all() -> pd.DataFrame:
    frames = []
    for indicator_code in WORLDBANK_INDICATORS:
        print(f"  Fetching World Bank: {indicator_code}")
        try:
            df = fetch_worldbank_indicator(indicator_code, COUNTRIES)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not fetch {indicator_code}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_pipeline() -> None:
    print("=== Data Ingestion Pipeline ===")
    print("\n[1/2] Fetching US indicators from FRED...")
    fred_df = fetch_fred_series()
    upsert_indicators(fred_df)
    print(f"  Stored {len(fred_df):,} FRED rows.")

    print("\n[2/2] Fetching multi-country indicators from World Bank...")
    wb_df = fetch_worldbank_all()
    if not wb_df.empty:
        upsert_indicators(wb_df)
        print(f"  Stored {len(wb_df):,} World Bank rows.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
