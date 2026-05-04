"""
Global Economic Risk Monitor — Main Dashboard
"""

import logging
import os
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from data.store import data_is_stale, read_model_outputs, read_indicators_multi
from data.ingest import run_pipeline
from models.recession import RecessionModel
from models.inflation import InflationModel
from models.composite import run as run_composite

load_dotenv()

st.set_page_config(
    page_title="Global Economic Risk Monitor",
    page_icon=":material/public:",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

st.session_state.setdefault("selected_country", "USA")

COUNTRY_META = {
    "USA": {"name": "United States", "iso_alpha": "USA", "flag": "🇺🇸"},
    "GBR": {"name": "United Kingdom", "iso_alpha": "GBR", "flag": "🇬🇧"},
    "DEU": {"name": "Germany",        "iso_alpha": "DEU", "flag": "🇩🇪"},
    "JPN": {"name": "Japan",          "iso_alpha": "JPN", "flag": "🇯🇵"},
}

REGIME_COLORS = {"low": "#2DC653", "moderate": "#F4A261", "high": "#E63946"}
REGIME_RGBA  = {"low": "rgba(45,198,83,0.6)", "moderate": "rgba(244,162,97,0.6)", "high": "rgba(230,57,70,0.6)"}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_recession_outputs() -> pd.DataFrame:
    return read_model_outputs(country_code="USA", model_name="recession")

@st.cache_data(ttl=3600, show_spinner=False)
def load_inflation_outputs() -> pd.DataFrame:
    return read_model_outputs(country_code="USA", model_name="inflation")

@st.cache_data(ttl=3600, show_spinner=False)
def load_map_data() -> pd.DataFrame:
    rows = []
    for code in COUNTRY_META:
        if code == "USA":
            rec = read_model_outputs(country_code="USA", model_name="recession")
            inf = read_model_outputs(country_code="USA", model_name="inflation")
            if rec.empty or inf.empty:
                continue
            rec_prob  = rec.sort_values("date").iloc[-1]["recession_prob"]
            inf_state = inf.sort_values("date").iloc[-1]["inflation_state"]
            inf_probs = json.loads(inf.sort_values("date").iloc[-1]["inflation_probs"])
            composite = 0.5 * rec_prob + 0.5 * inf_probs.get("high", 0)
            rows.append({"country_code": code, "name": COUNTRY_META[code]["name"],
                         "composite_risk": composite, "recession_prob": rec_prob,
                         "inflation_state": inf_state})
        else:
            df = read_model_outputs(country_code=code, model_name="composite")
            if df.empty:
                continue
            latest = df.sort_values("date").iloc[-1]
            rows.append({"country_code": code, "name": COUNTRY_META[code]["name"],
                         "composite_risk": latest["composite_risk"],
                         "recession_prob": latest["recession_prob"],
                         "inflation_state": latest["inflation_state"] or "N/A"})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def load_country_indicators(country_code: str) -> dict:
    wb_ids = ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"]
    raw = read_indicators_multi(country_code, wb_ids)
    return {
        "GDP Growth":    raw.get("NY.GDP.MKTP.KD.ZG"),
        "CPI Inflation": raw.get("FP.CPI.TOTL.ZG"),
        "Unemployment":  raw.get("SL.UEM.TOTL.ZS"),
    }

# ── Actions ───────────────────────────────────────────────────────────────────

def refresh_all_data():
    with st.status("Refreshing data...", expanded=True) as status:
        try:
            st.write(":material/download: Fetching from FRED and World Bank...")
            failed_series = run_pipeline()
            if failed_series:
                st.warning(
                    f"Could not fetch {len(failed_series)} FRED series: "
                    f"{', '.join(failed_series)}. Results may be incomplete.",
                    icon=":material/warning:",
                )
            st.write(":material/psychology: Running recession model...")
            RecessionModel().run()
            st.write(":material/bar_chart: Running inflation model...")
            InflationModel().run()
            st.write(":material/public: Scoring multi-country composite risk...")
            run_composite()
            status.update(label="Refresh complete!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label=f"Refresh failed: {exc}", state="error", expanded=True)
            st.exception(exc)
            return
    st.cache_data.clear()
    st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.title(":material/public: Global Economic Risk Monitor")
st.caption("Probabilistic recession and inflation risk — United States, United Kingdom, Germany, Japan")

col_btn, col_warn, _ = st.columns([1, 2, 3])
with col_btn:
    if st.button(":material/refresh: Refresh Data", use_container_width=True, key="refresh_btn"):
        refresh_all_data()
with col_warn:
    if data_is_stale():
        st.warning("Data may be outdated — click Refresh.", icon=":material/warning:")

st.divider()

# ── Map + Risk Snapshot ───────────────────────────────────────────────────────

map_data      = load_map_data()
recession_df  = load_recession_outputs()
inflation_df  = load_inflation_outputs()

left, right = st.columns([3, 2])

with left:
    st.subheader(":material/map: Risk Map")
    st.caption("Composite score = 50% recession probability + 50% inflation risk")
    with st.container(border=True):
        if map_data.empty:
            st.info("No model outputs yet — click Refresh Data to run the pipeline.",
                    icon=":material/info:")
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
                height=370,
            )
            click = st.plotly_chart(fig_map, use_container_width=True,
                                    on_select="rerun", key="map")
            if click and click.get("selection", {}).get("points"):
                st.session_state.selected_country = (
                    click["selection"]["points"][0].get("location", "USA")
                )

with right:
    st.subheader(":material/crisis_alert: Risk Snapshot")
    st.caption("Latest model outputs per country")
    if not map_data.empty:
        for _, row in map_data.iterrows():
            meta = COUNTRY_META.get(row["country_code"], {})
            risk = row["composite_risk"]
            icon = ":material/dangerous:" if risk > 0.65 else (
                   ":material/warning:"   if risk > 0.35 else
                   ":material/check_circle:")
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"###  {row['country_code']}")
                    st.caption(meta.get("name", ""))
                with c2:
                    _is_usa = row["country_code"] == "USA"
                    _comp_help = (
                        "0.5 × P(recession next 6 months) + 0.5 × P(high inflation regime). "
                        "Mixed horizon: recession component is 6-month forward; inflation component is current-state. "
                        "Not independently calibrated."
                        if _is_usa else
                        "Normalized stress index: 0.4 × GDP stress + 0.4 × CPI stress + 0.2 × unemployment stress. "
                        "Each component z-scored against own history, min-max scaled to [0–1]. Not a probability."
                    )
                    st.metric("Composite Risk", f"{risk:.0%}", help=_comp_help)
                    _rec_label = "Recession" if _is_usa else "Recession Stress"
                    st.caption(
                        f"{_rec_label}: **{row['recession_prob']:.0%}** · "
                        f"Inflation: **{str(row['inflation_state']).capitalize()}**"
                    )

st.divider()

# ── Country drill-down ────────────────────────────────────────────────────────

sel = st.session_state.selected_country
meta = COUNTRY_META.get(sel, {})
indicators = load_country_indicators(sel)
_filtered = map_data[map_data["country_code"] == sel] if not map_data.empty else pd.DataFrame()
model_row = _filtered.iloc[0] if not _filtered.empty else None

st.subheader(f":material/pin_drop: {meta.get('name', sel)} — Detail View")
st.caption("Click any country on the map above to switch the detail view.")

if model_row is not None:
    with st.container(border=True):
        kc1, kc2, kc3 = st.columns(3)
        risk = model_row["composite_risk"]
        icon = ":material/dangerous:" if risk > 0.65 else (":material/warning:" if risk > 0.35 else ":material/check_circle:")
        if sel == "USA":
            _comp_help  = "0.5 × P(recession next 6 months) + 0.5 × P(high inflation regime). Mixed horizon: recession is 6-month forward; inflation is current-state. Not independently calibrated."
            _rec_label  = ":material/trending_down: Recession Probability"
            _rec_help   = "P(NBER recession begins within the next 6 months). Calibrated logistic regression with sigmoid calibration (CalibratedClassifierCV, 5-fold TimeSeriesSplit), trained Jan 1985–Dec 2019. Post-2020 is out-of-sample. Point estimate — no confidence interval shown."
            _inf_help   = "Most probable latent state from the Gaussian HMM (3 states), labelled by learned CPI emission means. Current-state estimate only — not a forward forecast."
        else:
            _comp_help  = "Normalized stress index: 0.4 × GDP stress + 0.4 × CPI stress + 0.2 × unemployment stress. Each component z-scored against own history, min-max scaled to [0–1]. Not a probability — not calibrated against any ground truth."
            _rec_label  = ":material/trending_down: Recession Stress Score"
            _rec_help   = "Not a probability. Inverted GDP growth z-scored against this country's own history, min-max scaled to [0–1]. Higher = weaker growth relative to historical average. No model training or calibration applied."
            _inf_help   = "Rule-based threshold on World Bank annual CPI: < 2% = Low, 2–4% = Moderate, > 4% = High. Deterministic — no model or calibration."
        kc1.metric(f"{icon} Composite Risk", f"{risk:.0%}", help=_comp_help)
        kc2.metric(_rec_label, f"{model_row['recession_prob']:.0%}", help=_rec_help)
        kc3.metric(":material/trending_up: Inflation Regime", str(model_row["inflation_state"]).capitalize(), help=_inf_help)

col_d1, col_d2, col_d3 = st.columns(3)
chart_cfg = [
    ("GDP Growth",    col_d1, "#2DC653", "GDP Growth Rate (%)"),
    ("CPI Inflation", col_d2, "#E63946", "CPI Inflation (%)"),
    ("Unemployment",  col_d3, "#457B9D", "Unemployment (%)"),
]
for label, col, color, ytitle in chart_cfg:
    series = indicators.get(label)
    if series is None:
        continue
    with col:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=series.index, y=series.values,
                                       mode="lines+markers",
                                       line=dict(color=color, width=2),
                                       marker=dict(size=5)))
            fig_d.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                                yaxis_title=ytitle, showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)

st.divider()

# ── AI Risk Summary ───────────────────────────────────────────────────────────

def _build_risk_context(map_df: pd.DataFrame, rec_df: pd.DataFrame, inf_df: pd.DataFrame) -> str:
    lines = []
    for _, row in map_df.iterrows():
        lines.append(
            f"- {row['name']} ({row['country_code']}): composite risk {row['composite_risk']:.0%}, "
            f"recession prob {row['recession_prob']:.0%}, inflation regime {row['inflation_state']}"
        )
    if not rec_df.empty:
        latest_rec = rec_df.sort_values("date").iloc[-1]
        lines.append(f"\nUS recession model (logistic regression, 6-month horizon): "
                     f"P(recession) = {latest_rec['recession_prob']:.1%} as of {latest_rec['date']}.")
    if not inf_df.empty:
        latest_inf = inf_df.sort_values("date").iloc[-1]
        probs = json.loads(latest_inf["inflation_probs"])
        lines.append(f"US inflation regime (Gaussian HMM): current state = {latest_inf['inflation_state']}, "
                     f"P(low)={probs.get('low',0):.0%}, P(moderate)={probs.get('moderate',0):.0%}, "
                     f"P(high)={probs.get('high',0):.0%} as of {latest_inf['date']}.")
    return "\n".join(lines)


@st.cache_data(ttl=3600, show_spinner=False)
def generate_risk_summary(context_str: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": (
                        "You are a senior macroeconomic analyst. "
                        "Given model outputs from a Global Economic Risk Monitor, write a concise 2–3 paragraph "
                        "natural-language risk brief suitable for an executive audience. "
                        "Be direct, data-grounded, and avoid jargon. "
                        "Do not repeat raw numbers verbatim — synthesize them into insight. "
                        "Do not use bullet points."
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Here are the latest model outputs:\n\n{context_str}\n\nWrite the risk brief.",
                }
            ],
        )
        return response.content[0].text
    except anthropic.APIError as exc:
        return f"_Summary unavailable ({type(exc).__name__}): {exc}_"
    except Exception as exc:
        return f"_Summary unavailable: {exc}_"


_anthropic_key_set = bool(os.getenv("ANTHROPIC_API_KEY", ""))

if _anthropic_key_set:
    st.subheader(":material/summarize: AI Risk Summary")
    with st.container(border=True):
        with st.spinner("Generating macro risk brief…"):
            context_str = _build_risk_context(map_data, recession_df, inflation_df)
            summary = generate_risk_summary(context_str)
        if summary:
            st.markdown(summary)
            st.caption(
                "Generated by Claude (claude-haiku-4-5) · cached 1 hour · "
                "for illustrative purposes only, not investment advice."
            )
    st.divider()

# ── Deep-dive navigation ──────────────────────────────────────────────────────

st.subheader(":material/search: Explore Model Details")
nav1, nav2 = st.columns(2)

with nav1:
    with st.container(border=True):
        st.markdown("#### :material/trending_down: Recession Model")
        st.markdown(
            "Backtesting · Key indicators · Confusion matrix · "
            "Feature breakdown and model notes."
        )
        st.page_link(
            "pages/1_Recession_Model.py",
            label="Open Recession Deep-Dive",
            icon=":material/open_in_new:",
            use_container_width=True,
        )

with nav2:
    with st.container(border=True):
        st.markdown("#### :material/trending_up: Inflation Model")
        st.markdown(
            "Regime history · CPI & PCE charts · "
            "Monthly probability table and model notes."
        )
        st.page_link(
            "pages/2_Inflation_Model.py",
            label="Open Inflation Deep-Dive",
            icon=":material/open_in_new:",
            use_container_width=True,
        )
