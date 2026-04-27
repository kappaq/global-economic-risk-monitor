"""
Inflation Regime Model Deep-Dive Page
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.store import read_indicators, read_model_outputs

st.set_page_config(page_title="Inflation Model", page_icon="📈", layout="wide")
st.title("📈 Inflation Regime Model — Deep Dive")
st.caption("Gaussian Hidden Markov Model · 3 states: Low / Moderate / High · Trained on CPI, Core CPI, PCE, Expectations")

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    inf_out = read_model_outputs(country_code="USA", model_name="inflation")
    inf_out["date"] = pd.to_datetime(inf_out["date"])

    def _get(series_id):
        df = read_indicators(country_code="USA", series_id=series_id)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"].sort_index()

    cpi      = _get("CPIAUCSL").resample("MS").last()
    core_cpi = _get("CPILFESL").resample("MS").last()
    pce      = _get("PCEPI").resample("MS").last()
    mich     = _get("MICH").resample("MS").last()

    cpi_yoy  = cpi.pct_change(12, fill_method=None) * 100
    core_yoy = core_cpi.pct_change(12, fill_method=None) * 100

    return inf_out.sort_values("date"), cpi_yoy, core_yoy, mich

inf_out, cpi_yoy, core_yoy, mich = load_data()

# Parse probabilities
prob_low  = []
prob_mod  = []
prob_high = []
for probs_json in inf_out["inflation_probs"]:
    p = json.loads(probs_json)
    prob_low.append(p.get("low", 0))
    prob_mod.append(p.get("moderate", 0))
    prob_high.append(p.get("high", 0))

inf_out["prob_low"]      = prob_low
inf_out["prob_moderate"] = prob_mod
inf_out["prob_high"]     = prob_high

# ── Metrics ───────────────────────────────────────────────────────────────────

latest = inf_out.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Regime", latest["inflation_state"].capitalize())
col2.metric("P(Low Inflation)",      f"{latest['prob_low']:.1%}")
col3.metric("P(Moderate Inflation)", f"{latest['prob_moderate']:.1%}")
col4.metric("P(High Inflation)",     f"{latest['prob_high']:.1%}")

st.divider()

# ── Stacked regime probability chart ─────────────────────────────────────────

st.subheader("Inflation Regime Probabilities Over Time")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=inf_out["date"], y=inf_out["prob_low"],
    name="Low (<2%)", stackgroup="one",
    line=dict(width=0.5, color="#2DC653"),
    fillcolor="rgba(45,198,83,0.6)",
))
fig.add_trace(go.Scatter(
    x=inf_out["date"], y=inf_out["prob_moderate"],
    name="Moderate (2-4%)", stackgroup="one",
    line=dict(width=0.5, color="#F4A261"),
    fillcolor="rgba(244,162,97,0.6)",
))
fig.add_trace(go.Scatter(
    x=inf_out["date"], y=inf_out["prob_high"],
    name="High (>4%)", stackgroup="one",
    line=dict(width=0.5, color="#E63946"),
    fillcolor="rgba(230,57,70,0.6)",
))
fig.update_layout(
    height=350,
    yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1]),
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig, use_container_width=True)

# ── Raw CPI chart ─────────────────────────────────────────────────────────────

st.subheader("Underlying Inflation Data")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**CPI and Core CPI — Year-over-Year Change**")
    common = cpi_yoy.index.intersection(core_yoy.index)
    fig2 = go.Figure()
    fig2.add_hline(y=2, line_dash="dot", line_color="grey", opacity=0.6,
                   annotation_text="2% target")
    fig2.add_trace(go.Scatter(x=cpi_yoy.loc[common].index, y=cpi_yoy.loc[common].values,
                              name="CPI YoY", line=dict(color="#E63946")))
    fig2.add_trace(go.Scatter(x=core_yoy.loc[common].index, y=core_yoy.loc[common].values,
                              name="Core CPI YoY", line=dict(color="#F4A261", dash="dash")))
    fig2.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                       yaxis_title="YoY Change (%)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("**1-Year Inflation Expectations (UMich)**")
    fig3 = go.Figure()
    fig3.add_hline(y=2, line_dash="dot", line_color="grey", opacity=0.6,
                   annotation_text="2% target")
    fig3.add_trace(go.Scatter(x=mich.index, y=mich.values,
                              name="Expectations", line=dict(color="#457B9D")))
    fig3.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                       yaxis_title="Expected Inflation (%)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

# ── Regime timeline ───────────────────────────────────────────────────────────

st.divider()
st.subheader("Current Regime Snapshot")

regime_color = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
recent = inf_out.tail(12)[["date", "inflation_state", "prob_low", "prob_moderate", "prob_high"]].copy()
recent["date"] = recent["date"].dt.strftime("%b %Y")
recent["inflation_state"] = recent["inflation_state"].apply(
    lambda s: f"{regime_color.get(s, '')} {s.capitalize()}"
)
recent.columns = ["Month", "Regime", "P(Low)", "P(Moderate)", "P(High)"]
for col in ["P(Low)", "P(Moderate)", "P(High)"]:
    recent[col] = recent[col].apply(lambda x: f"{x:.1%}")
st.dataframe(recent.set_index("Month"), use_container_width=True)

st.divider()
st.subheader("Model Notes")
st.markdown("""
| Parameter | Value |
|-----------|-------|
| Algorithm | Gaussian Hidden Markov Model (`hmmlearn.GaussianHMM`) |
| Hidden states | 3 (labeled Low / Moderate / High by mean CPI post-training) |
| Features | CPI YoY · Core CPI YoY · PCE YoY · UMich 1Y inflation expectations |
| Training | Full history from 1980 |
| Covariance type | Full (allows correlated feature distributions per state) |

**Why HMM?** Inflation evolves through regimes with persistence — once in a high-inflation state, the economy tends to stay there. HMMs capture this latent state transition structure naturally, unlike a simple threshold rule.

**Output interpretation:** If P(High) = 0.72, there is a 72% posterior probability that the current macroeconomic state belongs to the high-inflation regime, given all observed indicator history up to that date.
""")
