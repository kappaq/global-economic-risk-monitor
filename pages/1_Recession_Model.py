"""
Recession Model Deep-Dive Page
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.store import read_indicators, read_model_outputs

st.set_page_config(page_title="Recession Model", page_icon="📉", layout="wide")
st.title("📉 Recession Probability Model — Deep Dive")
st.caption("Calibrated Logistic Regression · 6-month horizon · Trained on NBER recessions 1985–2019")

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    rec_out = read_model_outputs(country_code="USA", model_name="recession")
    rec_out["date"] = pd.to_datetime(rec_out["date"])

    def _get(series_id):
        df = read_indicators(country_code="USA", series_id=series_id)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"].sort_index().resample("MS").last()

    nber    = _get("USREC")
    t10y2   = _get("T10Y2Y").resample("MS").mean()
    t10y3   = _get("T10Y3M").resample("MS").mean()
    unemp   = _get("UNRATE")
    indpro  = _get("INDPRO")
    sent    = _get("UMCSENT")

    return rec_out.sort_values("date"), nber, t10y2, t10y3, unemp, indpro, sent

rec_out, nber, t10y2, t10y3, unemp, indpro, sent = load_data()

# NBER recession periods
nber_periods = []
start = None
nber_df = nber.reset_index()
for _, row in nber_df.iterrows():
    if row["value"] == 1 and start is None:
        start = row["date"]
    elif row["value"] != 1 and start is not None:
        nber_periods.append({"start": start, "end": row["date"]})
        start = None

def add_recessions(fig, label=False):
    for i, p in enumerate(nber_periods):
        fig.add_vrect(
            x0=p["start"], x1=p["end"], fillcolor="grey",
            opacity=0.15, line_width=0,
            annotation_text="NBER Recession" if (label and i == 0) else "",
            annotation_position="top left",
        )
    return fig

# ── Metrics ───────────────────────────────────────────────────────────────────

latest = rec_out.iloc[-1]
prev   = rec_out.iloc[-4]  # 3 months ago

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Recession Probability", f"{latest['recession_prob']:.1%}",
            delta=f"{latest['recession_prob'] - prev['recession_prob']:+.1%} vs 3 months ago")
col2.metric("Latest Yield Spread (10Y-2Y)", f"{t10y2.iloc[-1]:.2f}%")
col3.metric("Unemployment Rate", f"{unemp.iloc[-1]:.1f}%")
col4.metric("Latest 10Y-3M Spread", f"{t10y3.iloc[-1]:.2f}%")

st.divider()

# ── Full recession probability chart ──────────────────────────────────────────

st.subheader("Recession Probability Over Time (1985–Present)")

fig = go.Figure()
add_recessions(fig, label=True)
fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5,
              annotation_text="50% threshold")
fig.add_trace(go.Scatter(
    x=rec_out["date"], y=rec_out["recession_prob"],
    mode="lines", name="P(Recession)",
    line=dict(color="#E63946", width=2),
    fill="tozeroy", fillcolor="rgba(230,57,70,0.1)",
))
fig.update_layout(
    height=350,
    yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1]),
    xaxis_title="", margin=dict(l=0, r=0, t=30, b=0),
    modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Feature charts ────────────────────────────────────────────────────────────

st.subheader("Key Recession Indicators")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Yield Curve Spreads** (10Y−2Y and 10Y−3M)")
    fig2 = go.Figure()
    add_recessions(fig2)
    fig2.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4)
    fig2.add_trace(go.Scatter(x=t10y2.index, y=t10y2.values,
                              name="10Y-2Y", line=dict(color="#457B9D")))
    fig2.add_trace(go.Scatter(x=t10y3.index, y=t10y3.values,
                              name="10Y-3M", line=dict(color="#F4A261", dash="dash")))
    fig2.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                       yaxis_title="Spread (%)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("**Unemployment Rate**")
    fig3 = go.Figure()
    add_recessions(fig3)
    fig3.add_trace(go.Scatter(x=unemp.index, y=unemp.values,
                              name="UNRATE", line=dict(color="#E63946")))
    fig3.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                       yaxis_title="Rate (%)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

# ── Backtesting note ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Model Notes")
st.markdown("""
| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression with isotonic calibration (`CalibratedClassifierCV`) |
| Training window | January 1985 – December 2019 |
| Evaluation window | January 2020 – present (includes COVID recession) |
| Target | NBER recession indicator shifted **−6 months** (forward-looking) |
| Features | 10Y-2Y spread · 10Y-3M spread · ΔUnemployment (3M) · ΔIndustrial Production (3M) · ΔPayrolls (3M) · Consumer Sentiment z-score |
| Calibration | Isotonic regression on 5-fold CV |
| Class weighting | Balanced (recessions are rare events) |

**Why Logistic Regression?** Interpretable, well-calibrated with isotonic method, and performs well with 40 years of monthly data where sample size is limited. A neural net would overfit on ~500 training samples.

**Why 6-month horizon?** Follows the NY Fed recession probability model convention and gives actionable lead time for portfolio adjustments.
""")
