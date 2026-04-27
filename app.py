"""
Global Economic Risk Monitor — Main Dashboard
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.store import data_is_stale, read_model_outputs, read_indicators
from data.ingest import run_pipeline
from models.recession import RecessionModel
from models.inflation import InflationModel
from models.composite import run as run_composite

st.set_page_config(
    page_title="Global Economic Risk Monitor",
    page_icon="🌐",
    layout="wide",
)

COUNTRY_META = {
    "USA": {"name": "United States", "iso_alpha": "USA", "flag": "🇺🇸"},
    "GBR": {"name": "United Kingdom", "iso_alpha": "GBR", "flag": "🇬🇧"},
    "DEU": {"name": "Germany",        "iso_alpha": "DEU", "flag": "🇩🇪"},
    "JPN": {"name": "Japan",          "iso_alpha": "JPN", "flag": "🇯🇵"},
}


# ── Data & model loading ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_recession_outputs() -> pd.DataFrame:
    return read_model_outputs(country_code="USA", model_name="recession")

@st.cache_data(ttl=3600, show_spinner=False)
def load_inflation_outputs() -> pd.DataFrame:
    return read_model_outputs(country_code="USA", model_name="inflation")

@st.cache_data(ttl=3600, show_spinner=False)
def load_map_data() -> pd.DataFrame:
    """Latest composite risk score per country for the choropleth."""
    rows = []
    for code in COUNTRY_META:
        if code == "USA":
            rec = read_model_outputs(country_code="USA", model_name="recession")
            inf = read_model_outputs(country_code="USA", model_name="inflation")
            if rec.empty or inf.empty:
                continue
            rec_prob = rec.sort_values("date").iloc[-1]["recession_prob"]
            inf_state = inf.sort_values("date").iloc[-1]["inflation_state"]
            inf_probs = json.loads(inf.sort_values("date").iloc[-1]["inflation_probs"])
            composite = 0.5 * rec_prob + 0.5 * inf_probs.get("high", 0)
            rows.append({
                "country_code": code,
                "name": COUNTRY_META[code]["name"],
                "composite_risk": composite,
                "recession_prob": rec_prob,
                "inflation_state": inf_state,
            })
        else:
            df = read_model_outputs(country_code=code, model_name="composite")
            if df.empty:
                continue
            latest = df.sort_values("date").iloc[-1]
            rows.append({
                "country_code": code,
                "name": COUNTRY_META[code]["name"],
                "composite_risk": latest["composite_risk"],
                "recession_prob": latest["recession_prob"],
                "inflation_state": "N/A",
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def load_nber_recessions() -> list[dict]:
    df = read_indicators(country_code="USA", series_id="USREC")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["in_recession"] = df["value"] == 1
    periods = []
    start = None
    for _, row in df.iterrows():
        if row["in_recession"] and start is None:
            start = row["date"]
        elif not row["in_recession"] and start is not None:
            periods.append({"start": start, "end": row["date"]})
            start = None
    return periods


def refresh_all_data():
    with st.spinner("Fetching latest data from FRED and World Bank..."):
        run_pipeline()
    with st.spinner("Running recession model..."):
        RecessionModel().run()
    with st.spinner("Running inflation model..."):
        InflationModel().run()
    with st.spinner("Scoring multi-country composite risk..."):
        run_composite()
    st.cache_data.clear()
    st.success("Data refreshed successfully.")
    st.rerun()


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🌐 Global Economic Risk Monitor")
st.caption("Probabilistic recession and inflation risk — United States, United Kingdom, Germany, Japan")

col_refresh, col_stale = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Data", use_container_width=True):
        refresh_all_data()
with col_stale:
    if data_is_stale():
        st.warning("Data may be outdated. Click Refresh to fetch latest indicators.")

st.divider()

# ── Map + sidebar ─────────────────────────────────────────────────────────────

map_data = load_map_data()
recession_df = load_recession_outputs()
inflation_df = load_inflation_outputs()

left, right = st.columns([3, 2])

with left:
    st.subheader("Risk Map — Click a country to drill down")
    if map_data.empty:
        st.info("No model outputs yet. Click Refresh Data to run the pipeline.")
    else:
        fig_map = px.choropleth(
            map_data,
            locations="country_code",
            locationmode="ISO-3",
            color="composite_risk",
            hover_name="name",
            hover_data={
                "country_code": False,
                "composite_risk": ":.0%",
                "recession_prob": ":.0%",
                "inflation_state": True,
            },
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            labels={
                "composite_risk": "Composite Risk",
                "recession_prob": "Recession Prob",
                "inflation_state": "Inflation Regime",
            },
        )
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            coloraxis_colorbar=dict(title="Risk", tickformat=".0%"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=380,
        )
        click = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map")
        selected_country = "USA"
        if click and click.get("selection", {}).get("points"):
            point = click["selection"]["points"][0]
            selected_country = point.get("location", "USA")

with right:
    st.subheader("Current Risk Snapshot")
    if not map_data.empty:
        for _, row in map_data.iterrows():
            meta = COUNTRY_META.get(row["country_code"], {})
            risk = row["composite_risk"]
            color = "🔴" if risk > 0.65 else ("🟡" if risk > 0.35 else "🟢")
            st.metric(
                label=f"{meta.get('flag','')} {row['name']}",
                value=f"{risk:.0%} risk",
                delta=f"Recession: {row['recession_prob']:.0%} | Inflation: {row['inflation_state']}",
                delta_color="off",
            )

st.divider()

# ── Time series: Recession ────────────────────────────────────────────────────

st.subheader("📉 Recession Probability — United States (6-month horizon)")

if not recession_df.empty:
    recession_df["date"] = pd.to_datetime(recession_df["date"])
    recession_df = recession_df.sort_values("date")
    nber = load_nber_recessions()

    fig_rec = go.Figure()

    for period in nber:
        fig_rec.add_vrect(
            x0=period["start"], x1=period["end"],
            fillcolor="grey", opacity=0.15, line_width=0,
        )

    # Single legend entry for NBER shading
    fig_rec.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="grey", opacity=0.4, symbol="square"),
        name="NBER Recession",
    ))

    fig_rec.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5,
                      annotation_text="50% threshold", annotation_position="bottom right")

    fig_rec.add_trace(go.Scatter(
        x=recession_df["date"],
        y=recession_df["recession_prob"],
        mode="lines",
        name="Recession Probability",
        line=dict(color="#E63946", width=2),
        fill="tozeroy",
        fillcolor="rgba(230,57,70,0.1)",
    ))

    fig_rec.update_layout(
        yaxis=dict(title="P(Recession next 6 months)", tickformat=".0%", range=[0, 1]),
        xaxis_title="Date",
        height=320,
        margin=dict(l=0, r=20, t=10, b=0),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_rec, use_container_width=True,
                    config={"displayModeBar": False})

    latest_rec = recession_df.iloc[-1]
    st.caption(
        f"Latest estimate ({latest_rec['date'].strftime('%b %Y')}): "
        f"**{latest_rec['recession_prob']:.1%}** probability of recession within 6 months. "
        f"Model: Logistic Regression trained on 1985–2019 NBER recession data."
    )

st.divider()

# ── Time series: Inflation regime ─────────────────────────────────────────────

st.subheader("📈 Inflation Regime — United States")

if not inflation_df.empty:
    inflation_df["date"] = pd.to_datetime(inflation_df["date"])
    inflation_df = inflation_df.sort_values("date")

    prob_cols = {"low": [], "moderate": [], "high": []}
    for probs_json in inflation_df["inflation_probs"]:
        probs = json.loads(probs_json)
        for k in prob_cols:
            prob_cols[k].append(probs.get(k, 0.0))

    fig_inf = go.Figure()
    colors = {"low": "#2DC653", "moderate": "#F4A261", "high": "#E63946"}

    for regime, color in colors.items():
        fig_inf.add_trace(go.Scatter(
            x=inflation_df["date"],
            y=prob_cols[regime],
            mode="lines",
            name=regime.capitalize(),
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.6)" if False else color,
        ))

    fig_inf.update_layout(
        yaxis=dict(title="Regime Probability", tickformat=".0%", range=[0, 1]),
        xaxis_title="Date",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_inf, use_container_width=True)

    latest_inf = inflation_df.iloc[-1]
    probs_latest = json.loads(latest_inf["inflation_probs"])
    st.caption(
        f"Latest estimate ({latest_inf['date'].strftime('%b %Y')}): "
        f"Regime = **{latest_inf['inflation_state'].capitalize()}** | "
        f"Low: {probs_latest.get('low', 0):.0%} · "
        f"Moderate: {probs_latest.get('moderate', 0):.0%} · "
        f"High: {probs_latest.get('high', 0):.0%}. "
        f"Model: Gaussian HMM (3 states) on CPI, Core CPI, PCE, expectations."
    )
