"""
Inflation Regime Model Deep-Dive Page
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go

from data.store import read_indicators, read_model_outputs

st.set_page_config(
    page_title="Inflation Model",
    page_icon=":material/trending_up:",
    layout="wide",
)

st.title(":material/trending_up: Inflation Regime Model")
st.caption("Gaussian Hidden Markov Model · 3 states: Low / Moderate / High · CPI, Core CPI, PCE, Expectations")

REGIME_COLORS = {"low": "#2DC653", "moderate": "#F4A261", "high": "#E63946"}
REGIME_RGBA   = {"low": "rgba(45,198,83,0.6)", "moderate": "rgba(244,162,97,0.6)", "high": "rgba(230,57,70,0.6)"}
REGIME_ICONS  = {"low": ":material/check_circle:", "moderate": ":material/warning:", "high": ":material/dangerous:"}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
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
    pce_yoy  = pce.pct_change(12, fill_method=None) * 100

    return inf_out.sort_values("date"), cpi_yoy, core_yoy, pce_yoy, mich

inf_out, cpi_yoy, core_yoy, pce_yoy, mich = load_data()

if inf_out.empty:
    st.error(
        "No inflation model outputs found. Return to the dashboard and click **Refresh Data** to run the pipeline.",
        icon=":material/error:",
    )
    st.stop()

# Parse probabilities once — guard against NULL inflation_probs (non-US composite rows)
prob_cols = {"low": [], "moderate": [], "high": []}
for probs_json in inf_out["inflation_probs"]:
    p = json.loads(probs_json) if probs_json else {"low": 0.0, "moderate": 0.0, "high": 0.0}
    for k in prob_cols:
        prob_cols[k].append(p.get(k, 0.0))
inf_out["prob_low"]      = prob_cols["low"]
inf_out["prob_moderate"] = prob_cols["moderate"]
inf_out["prob_high"]     = prob_cols["high"]

latest = inf_out.iloc[-1]
probs_latest = json.loads(latest["inflation_probs"])

# ── KPI metrics ───────────────────────────────────────────────────────────────

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    icon = REGIME_ICONS.get(latest["inflation_state"], "")
    c1.metric(
        f"{icon} Current Regime",
        latest["inflation_state"].capitalize(),
        help=(
            "Most probable latent inflation state at this date, from the Gaussian HMM (3 hidden states). "
            "States are labelled Low / Moderate / High by sorting learned emission means on CPI YoY — "
            "not by a fixed threshold. Current-state estimate, not a forecast."
        ),
    )
    c2.metric(
        ":material/check_circle: P(Low)",
        f"{latest['prob_low']:.2%}",
        help=(
            "Smoothed posterior P(current state = Low inflation). "
            "Estimated via the forward-backward algorithm on the full observation sequence (CPI YoY, Core CPI YoY, PCE YoY, UMich expectations). "
            "Horizon: current state only — not a forward forecast. "
            "Not calibrated against external labels (unsupervised model)."
        ),
    )
    c3.metric(
        ":material/warning: P(Moderate)",
        f"{latest['prob_moderate']:.2%}",
        help=(
            "Smoothed posterior P(current state = Moderate inflation). "
            "Same forward-backward pass as P(Low). "
            "Horizon: current state only. Not calibrated."
        ),
    )
    c4.metric(
        ":material/dangerous: P(High)",
        f"{latest['prob_high']:.2%}",
        help=(
            "Smoothed posterior P(current state = High inflation). "
            "The HMM learned its transition dynamics from 1980–2019 data; "
            "the 2021–2023 inflation surge is fully out-of-sample. "
            "Horizon: current state only. Not calibrated."
        ),
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_regime, tab_data, tab_history, tab_notes = st.tabs([
    ":material/stacked_bar_chart: Regime Probabilities",
    ":material/analytics: Underlying Data",
    ":material/history: Recent History",
    ":material/info: Model Notes",
])

# ── Tab 1: Regime probabilities ───────────────────────────────────────────────

with tab_regime:
    fig = go.Figure()
    for regime in ["low", "moderate", "high"]:
        label = {"low": "Low (<2%)", "moderate": "Moderate (2–4%)", "high": "High (>4%)"}[regime]
        fig.add_trace(go.Scatter(
            x=inf_out["date"], y=inf_out[f"prob_{regime}"],
            name=label, stackgroup="one",
            line=dict(width=0.5, color=REGIME_COLORS[regime]),
            fillcolor=REGIME_RGBA[regime],
        ))
    fig.update_layout(
        height=380,
        yaxis=dict(title="Regime Probability", tickformat=".0%", range=[0, 1]),
        xaxis_title="",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.06),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**{latest['date'].strftime('%b %Y')}** — "
        f"Low: {probs_latest.get('low', 0):.2%} · "
        f"Moderate: {probs_latest.get('moderate', 0):.2%} · "
        f"High: {probs_latest.get('high', 0):.2%}. "
        f"Stacked area sums to 100% at every date. "
        f"Near-100% posteriors are expected HMM behaviour when observations fall squarely "
        f"within one state's emission distribution — they reflect model sharpness, not certainty."
    )

# ── Tab 2: Underlying data ────────────────────────────────────────────────────

with tab_data:
    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.markdown("**CPI and Core CPI — Year-over-Year**")
            common = cpi_yoy.index.intersection(core_yoy.index)
            fig2 = go.Figure()
            fig2.add_hline(y=2, line_dash="dot", line_color="grey", opacity=0.6,
                           annotation_text="2% Fed target", annotation_position="bottom right")
            fig2.add_trace(go.Scatter(x=cpi_yoy.loc[common].index, y=cpi_yoy.loc[common].values,
                                      name="CPI YoY", line=dict(color="#E63946", width=1.5)))
            fig2.add_trace(go.Scatter(x=core_yoy.loc[common].index, y=core_yoy.loc[common].values,
                                      name="Core CPI YoY", line=dict(color="#F4A261", width=1.5, dash="dash")))
            fig2.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title="YoY Change (%)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        with st.container(border=True):
            st.markdown("**PCE and 1-Year Inflation Expectations (UMich)**")
            common2 = pce_yoy.index.intersection(mich.index)
            fig3 = go.Figure()
            fig3.add_hline(y=2, line_dash="dot", line_color="grey", opacity=0.6,
                           annotation_text="2% Fed target", annotation_position="bottom right")
            fig3.add_trace(go.Scatter(x=pce_yoy.loc[common2].index, y=pce_yoy.loc[common2].values,
                                      name="PCE YoY", line=dict(color="#457B9D", width=1.5)))
            fig3.add_trace(go.Scatter(x=mich.loc[common2].index, y=mich.loc[common2].values,
                                      name="UMich Expectations", line=dict(color="#A8DADC", width=1.5, dash="dash")))
            fig3.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title="(%)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Recent History ─────────────────────────────────────────────────────

with tab_history:
    st.markdown("Last 12 months of model outputs")
    REGIME_EMOJI = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
    recent = inf_out.tail(12)[["date", "inflation_state", "prob_low", "prob_moderate", "prob_high"]].copy()
    recent["date"] = recent["date"].dt.strftime("%b %Y")
    recent["inflation_state"] = recent["inflation_state"].apply(
        lambda s: f"{REGIME_EMOJI.get(s, '')} {s.capitalize()}"
    )
    recent.columns = ["Month", "Regime", "P(Low)", "P(Moderate)", "P(High)"]
    st.dataframe(
        recent.set_index("Month"),
        column_config={
            "P(Low)":      st.column_config.ProgressColumn("P(Low)",      min_value=0, max_value=1, format="%.0%"),
            "P(Moderate)": st.column_config.ProgressColumn("P(Moderate)", min_value=0, max_value=1, format="%.0%"),
            "P(High)":     st.column_config.ProgressColumn("P(High)",     min_value=0, max_value=1, format="%.0%"),
        },
        use_container_width=True,
    )

# ── Tab 4: Model Notes ────────────────────────────────────────────────────────

with tab_notes:
    with st.container(border=True):
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Algorithm | Gaussian Hidden Markov Model (`hmmlearn.GaussianHMM`) |
| Hidden states | 3 — labeled Low / Moderate / High by mean CPI post-training |
| Features | CPI YoY · Core CPI YoY · PCE YoY · UMich 1Y inflation expectations |
| Training | Full history from 1980 (no hold-out — unsupervised latent model) |
| Covariance type | Full — allows correlated feature distributions per state |
| Iterations | 200 |
""")

    col_why1, col_why2 = st.columns(2)
    with col_why1:
        with st.container(border=True):
            st.markdown("**Why HMM?**")
            st.markdown(
                "Inflation evolves through regimes with persistence — once in a high-inflation "
                "state, the economy tends to stay there. HMMs capture latent state transition "
                "dynamics naturally, unlike a simple threshold rule. The model learns the "
                "transition matrix from data, not from prior assumptions."
            )
    with col_why2:
        with st.container(border=True):
            st.markdown("**Output interpretation**")
            st.markdown(
                "If P(High) = 0.72 at a given date, there is a 72% posterior probability "
                "that the current macroeconomic state belongs to the high-inflation regime, "
                "given all observed indicator history up to that point. This is a smoothed "
                "estimate, not a point forecast of next month's CPI."
            )
