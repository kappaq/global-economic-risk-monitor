"""
Multi-country composite risk scorer.
For UK, Germany, Japan: uses World Bank GDP/CPI/unemployment data
to produce a [0,1] risk scalar suitable for the choropleth map.
Each indicator is z-score normalized against the country's own history.
"""

import numpy as np
import pandas as pd

from data.store import read_indicators, upsert_model_outputs

COUNTRIES = ["GBR", "DEU", "JPN"]
MODEL_NAME = "composite"

WEIGHTS = {"recession_proxy": 0.4, "inflation_proxy": 0.4, "unemployment_proxy": 0.2}


def _zscore_to_risk(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize a series to [0,1] via min-max on its z-score."""
    std = series.std()
    z = (series - series.mean()) / (std if std != 0 else 1)
    if invert:
        z = -z
    return ((z - z.min()) / (z.max() - z.min())).fillna(0.5)


def score_country(country_code: str) -> pd.DataFrame:
    def _get(series_id: str) -> pd.Series:
        df = read_indicators(country_code=country_code, series_id=series_id)
        if df.empty:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"].sort_index()

    gdp   = _get("NY.GDP.MKTP.KD.ZG")
    cpi   = _get("FP.CPI.TOTL.ZG")
    unemp = _get("SL.UEM.TOTL.ZS")

    if gdp.empty or cpi.empty or unemp.empty:
        return pd.DataFrame()

    common_idx = gdp.index.intersection(cpi.index).intersection(unemp.index)
    if len(common_idx) < 3:
        return pd.DataFrame()

    gdp   = gdp.reindex(common_idx)
    cpi   = cpi.reindex(common_idx)
    unemp = unemp.reindex(common_idx)

    # Low/negative GDP growth → high recession risk
    recession_proxy   = _zscore_to_risk(gdp, invert=True)
    # High CPI → high inflation risk
    inflation_proxy   = _zscore_to_risk(cpi, invert=False)
    # High unemployment → additional risk
    unemployment_proxy = _zscore_to_risk(unemp, invert=False)

    composite = (
        WEIGHTS["recession_proxy"]   * recession_proxy
        + WEIGHTS["inflation_proxy"]   * inflation_proxy
        + WEIGHTS["unemployment_proxy"] * unemployment_proxy
    )

    return pd.DataFrame({
        "country_code":   country_code,
        "model_name":     MODEL_NAME,
        "date":           common_idx,
        "recession_prob":  recession_proxy.values,
        "inflation_state": None,
        "inflation_probs": None,
        "composite_risk":  composite.values,
    })


def run() -> None:
    print("  Scoring multi-country composite risk...")
    frames = []
    for country in COUNTRIES:
        df = score_country(country)
        if not df.empty:
            frames.append(df)
            print(f"  {country}: {len(df)} annual data points scored.")
    if frames:
        upsert_model_outputs(pd.concat(frames, ignore_index=True))
    print("  Multi-country scoring complete.")


if __name__ == "__main__":
    run()
