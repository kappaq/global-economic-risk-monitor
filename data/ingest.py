"""
Data ingestion pipeline.
Sources: FRED API (US macro indicators) + World Bank API (multi-country).
Pipeline is idempotent — safe to re-run at any time.
"""

import logging
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from data.store import upsert_indicators

load_dotenv()

logger = logging.getLogger(__name__)

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

_WB_ISO2 = {"USA": "US", "GBR": "GB", "DEU": "DE", "JPN": "JP"}
_WB_ISO3 = {"US": "USA", "GB": "GBR", "DE": "DEU", "JP": "JPN"}


def _to_timestamp(val) -> pd.Timestamp:
    """Normalise any date-like value to a timezone-naive pd.Timestamp at midnight."""
    ts = pd.Timestamp(val)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()  # strips time component → midnight


def fetch_fred_series(retries: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Fetch all FRED series. Returns (DataFrame, list_of_failed_series_ids)."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set. Copy .env.example to .env and add your key.")
    fred = Fred(api_key=api_key)
    rows: list[pd.DataFrame] = []
    failed: list[str] = []
    for series_id in FRED_SERIES:
        for attempt in range(retries):
            try:
                logger.info("Fetching FRED: %s (attempt %d/%d)", series_id, attempt + 1, retries)
                s = fred.get_series(series_id, observation_start=FRED_START)
                s = s.dropna().reset_index()
                s.columns = ["date", "value"]
                s["date"] = s["date"].apply(_to_timestamp)
                s["country_code"] = "USA"
                s["series_id"] = series_id
                rows.append(s[["country_code", "series_id", "date", "value"]])
                break
            except Exception as exc:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning("FRED %s failed (attempt %d/%d): %s — retrying in %ds",
                                   series_id, attempt + 1, retries, exc, wait)
                    time.sleep(wait)
                else:
                    logger.warning("FRED %s failed after %d attempts: %s", series_id, retries, exc)
                    failed.append(series_id)
    if not rows:
        raise RuntimeError(
            "Failed to fetch any FRED series. Check FRED_API_KEY and network connectivity."
        )
    return pd.concat(rows, ignore_index=True), failed


def fetch_worldbank_indicator(
    indicator_code: str, countries: list[str], retries: int = 3
) -> pd.DataFrame:
    """Fetch one World Bank indicator for a list of countries. Retries on 429/transient errors."""
    iso2_codes = ";".join(_WB_ISO2[c] for c in countries)
    url = (
        f"https://api.worldbank.org/v2/country/{iso2_codes}/indicator/{indicator_code}"
        f"?format=json&per_page=500&mrv=50"
    )
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429 and attempt < retries - 1:
            wait = 2 ** attempt
            logger.warning("World Bank rate-limited (429); retrying in %ds...", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break

    payload = resp.json()
    if len(payload) < 2 or not payload[1]:
        return pd.DataFrame()

    rows = []
    for entry in payload[1]:
        if entry["value"] is None:
            continue
        country_code = _WB_ISO3.get(entry["country"]["id"], entry["countryiso3code"])
        rows.append({
            "country_code": country_code,
            "series_id":    indicator_code,
            "date":         _to_timestamp(f"{entry['date']}-01-01"),
            "value":        float(entry["value"]),
        })
    return pd.DataFrame(rows)


def fetch_worldbank_all() -> pd.DataFrame:
    frames = []
    for indicator_code in WORLDBANK_INDICATORS:
        logger.info("Fetching World Bank: %s", indicator_code)
        try:
            df = fetch_worldbank_indicator(indicator_code, COUNTRIES)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", indicator_code, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_pipeline() -> list[str]:
    """Run the full ingestion pipeline. Returns list of FRED series IDs that failed to fetch."""
    logger.info("=== Data Ingestion Pipeline ===")

    logger.info("[1/2] Fetching US indicators from FRED...")
    fred_df, failed_series = fetch_fred_series()
    upsert_indicators(fred_df)
    logger.info("Stored %d FRED rows.", len(fred_df))
    if failed_series:
        logger.warning("Missing FRED series: %s", ", ".join(failed_series))

    logger.info("[2/2] Fetching multi-country indicators from World Bank...")
    wb_df = fetch_worldbank_all()
    if not wb_df.empty:
        upsert_indicators(wb_df)
        logger.info("Stored %d World Bank rows.", len(wb_df))

    logger.info("Pipeline complete.")
    return failed_series


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failed = run_pipeline()
    if failed:
        print(f"\nWARNING: {len(failed)} series failed: {', '.join(failed)}")
    else:
        print("\nAll series fetched successfully.")
