"""
Global Economic Risk Monitor — Main Dashboard
"""

import os
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from data.store import data_is_stale, read_model_outputs, read_indicators, read_indicators_multi
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
    """Fetch all indicators for a country in one DB query."""
    if country_code == "USA":
        series_ids = ["T10Y2Y", "T10Y3M", "UNRATE", "INDPRO", "CPIAUCSL", "VIXCLS"]
        return read_indicators_multi("USA", series_ids)
    wb_ids = ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"]
    raw = read_indicators_multi(country_code, wb_ids)
    return {
        "GDP Growth":    raw.get("NY.GDP.MKTP.KD.ZG"),
        "CPI Inflation": raw.get("FP.CPI.TOTL.ZG"),
        "Unemployment":  raw.get("SL.UEM.TOTL.ZS"),
    }

@st.cache_data(ttl=3600, show_spinner=False)
def load_nber_recessions() -> list[dict]:
    df = read_indicators(country_code="USA", series_id="USREC")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    periods, start = [], None
    for _, row in df.iterrows():
        if row["value"] == 1 and start is None:
            start = row["date"]
        elif row["value"] != 1 and start is not None:
            periods.append({"start": start, "end": row["date"]})
            start = None
    return periods

# ── Actions ───────────────────────────────────────────────────────────────────

def refresh_all_data():
    with st.status("Refreshing data...", expanded=True) as status:
        st.write(":material/download: Fetching from FRED and World Bank...")
        run_pipeline()
        st.write(":material/psychology: Running recession model...")
        RecessionModel().run()
        st.write(":material/bar_chart: Running inflation model...")
        InflationModel().run()
        st.write(":material/public: Scoring multi-country composite risk...")
        run_composite()
        status.update(label="Refresh complete!", state="complete", expanded=False)
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
nber_periods  = load_nber_recessions()

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
                    st.markdown(f"### {meta.get('flag','')} {row['country_code']}")
                    st.caption(meta.get("name", ""))
                with c2:
                    st.metric("Composite Risk", f"{risk:.0%}")
                    st.caption(
                        f"Recession: **{row['recession_prob']:.0%}** · "
                        f"Inflation: **{str(row['inflation_state']).capitalize()}**"
                    )

st.divider()

# ── Country drill-down ────────────────────────────────────────────────────────

sel = st.session_state.selected_country
meta = COUNTRY_META.get(sel, {})
indicators = load_country_indicators(sel)
model_row = map_data[map_data["country_code"] == sel].iloc[0] if not map_data.empty and sel in map_data["country_code"].values else None

st.subheader(f":material/pin_drop: {meta.get('flag','')} {meta.get('name', sel)} — Detail View")
st.caption("Click any country on the map above to switch the detail view.")

if model_row is not None:
    with st.container(border=True):
        kc1, kc2, kc3 = st.columns(3)
        risk = model_row["composite_risk"]
        icon = ":material/dangerous:" if risk > 0.65 else (":material/warning:" if risk > 0.35 else ":material/check_circle:")
        kc1.metric(f"{icon} Composite Risk",    f"{risk:.0%}")
        kc2.metric(":material/trending_down: Recession Risk", f"{model_row['recession_prob']:.0%}")
        kc3.metric(":material/trending_up: Inflation Regime", str(model_row["inflation_state"]).capitalize())

if sel == "USA":
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        with st.container(border=True):
            st.markdown("**Yield Curve Spreads**")
            fig_d1 = go.Figure()
            for p in nber_periods:
                fig_d1.add_vrect(x0=p["start"], x1=p["end"], fillcolor="grey", opacity=0.12, line_width=0)
            fig_d1.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
            if indicators.get("T10Y2Y") is not None:
                s = indicators["T10Y2Y"].resample("MS").mean()
                fig_d1.add_trace(go.Scatter(x=s.index, y=s.values, name="10Y-2Y",
                                            line=dict(color="#457B9D", width=1.5)))
            if indicators.get("T10Y3M") is not None:
                s = indicators["T10Y3M"].resample("MS").mean()
                fig_d1.add_trace(go.Scatter(x=s.index, y=s.values, name="10Y-3M",
                                            line=dict(color="#F4A261", width=1.5, dash="dash")))
            fig_d1.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis_title="Spread (%)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_d1, use_container_width=True)

    with col_d2:
        with st.container(border=True):
            st.markdown("**Unemployment & Industrial Production**")
            fig_d2 = go.Figure()
            for p in nber_periods:
                fig_d2.add_vrect(x0=p["start"], x1=p["end"], fillcolor="grey", opacity=0.12, line_width=0)
            if indicators.get("UNRATE") is not None:
                s = indicators["UNRATE"].resample("MS").last()
                fig_d2.add_trace(go.Scatter(x=s.index, y=s.values, name="Unemployment %",
                                            line=dict(color="#E63946", width=1.5)))
            fig_d2.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis_title="Rate (%)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_d2, use_container_width=True)

else:
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

# ── Time Series Charts ────────────────────────────────────────────────────────

st.subheader(":material/timeline: Model Output — United States")

tab_rec, tab_inf = st.tabs([
    ":material/trending_down: Recession Probability",
    ":material/trending_up: Inflation Regime",
])

with tab_rec:
    if not recession_df.empty:
        recession_df["date"] = pd.to_datetime(recession_df["date"])
        recession_df = recession_df.sort_values("date")

        fig_rec = go.Figure()
        for i, period in enumerate(nber_periods):
            fig_rec.add_vrect(
                x0=period["start"], x1=period["end"],
                fillcolor="grey", opacity=0.15, line_width=0,
                annotation_text="NBER Recession" if i == 0 else "",
                annotation_position="top left",
            )
        fig_rec.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5,
                          annotation_text="50% threshold")
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
            height=340,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.05),
            modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_rec, use_container_width=True)

        latest_rec = recession_df.iloc[-1]
        st.caption(
            f"**{latest_rec['date'].strftime('%b %Y')}** — "
            f"P(recession next 6 months) = **{latest_rec['recession_prob']:.1%}**. "
            f"Model: Calibrated Logistic Regression trained on NBER recessions 1985–2019."
        )

with tab_inf:
    if not inflation_df.empty:
        inflation_df["date"] = pd.to_datetime(inflation_df["date"])
        inflation_df = inflation_df.sort_values("date")

        prob_cols = {"low": [], "moderate": [], "high": []}
        for probs_json in inflation_df["inflation_probs"]:
            probs = json.loads(probs_json)
            for k in prob_cols:
                prob_cols[k].append(probs.get(k, 0.0))

        fig_inf = go.Figure()
        for regime in ["low", "moderate", "high"]:
            fig_inf.add_trace(go.Scatter(
                x=inflation_df["date"],
                y=prob_cols[regime],
                mode="lines",
                name=regime.capitalize(),
                stackgroup="one",
                line=dict(width=0.5, color=REGIME_COLORS[regime]),
                fillcolor=REGIME_RGBA[regime],
            ))
        fig_inf.update_layout(
            yaxis=dict(title="Regime Probability", tickformat=".0%", range=[0, 1]),
            xaxis_title="Date",
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_inf, use_container_width=True)

        latest_inf = inflation_df.iloc[-1]
        probs_latest = json.loads(latest_inf["inflation_probs"])
        st.caption(
            f"**{latest_inf['date'].strftime('%b %Y')}** — "
            f"Regime = **{latest_inf['inflation_state'].capitalize()}** | "
            f"Low: {probs_latest.get('low', 0):.0%} · "
            f"Moderate: {probs_latest.get('moderate', 0):.0%} · "
            f"High: {probs_latest.get('high', 0):.0%}. "
            f"Model: Gaussian HMM (3 states) on CPI, Core CPI, PCE, expectations."
        )

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
