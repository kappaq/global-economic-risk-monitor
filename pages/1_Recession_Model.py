"""
Recession Model Deep-Dive Page
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from data.store import read_indicators, read_indicators_multi, read_model_outputs

st.set_page_config(
    page_title="Recession Model",
    page_icon=":material/trending_down:",
    layout="wide",
)

st.title(":material/trending_down: Recession Probability Model")
st.caption("Calibrated Logistic Regression · 6-month horizon · Trained on NBER recessions 1985–2019")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    rec_out = read_model_outputs(country_code="USA", model_name="recession")
    rec_out["date"] = pd.to_datetime(rec_out["date"])

    series = read_indicators_multi("USA", ["USREC", "T10Y2Y", "T10Y3M", "UNRATE", "INDPRO"])
    nber   = series.get("USREC",  pd.Series(dtype=float)).resample("MS").last()
    t10y2  = series.get("T10Y2Y", pd.Series(dtype=float)).resample("MS").mean()
    t10y3  = series.get("T10Y3M", pd.Series(dtype=float)).resample("MS").mean()
    unemp  = series.get("UNRATE", pd.Series(dtype=float)).resample("MS").last()
    indpro = series.get("INDPRO", pd.Series(dtype=float)).resample("MS").last()

    return rec_out.sort_values("date"), nber, t10y2, t10y3, unemp, indpro

rec_out, nber, t10y2, t10y3, unemp, indpro = load_data()

if rec_out.empty:
    st.error(
        "No recession model outputs found. Return to the dashboard and click **Refresh Data** to run the pipeline.",
        icon=":material/error:",
    )
    st.stop()

# Build recession period list
nber_periods = []
start = None
for date, val in nber.items():
    if val == 1 and start is None:
        start = date
    elif val != 1 and start is not None:
        nber_periods.append({"start": start, "end": date})
        start = None

def add_recessions(fig, label=False):
    for i, p in enumerate(nber_periods):
        fig.add_vrect(
            x0=p["start"], x1=p["end"],
            fillcolor="grey", opacity=0.15, line_width=0,
            annotation_text="NBER Recession" if (label and i == 0) else "",
            annotation_position="top left",
        )
    return fig

# ── KPI metrics ───────────────────────────────────────────────────────────────

latest = rec_out.iloc[-1]
prev   = rec_out.iloc[-4] if len(rec_out) >= 4 else rec_out.iloc[0]

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        ":material/crisis_alert: Recession Probability",
        f"{latest['recession_prob']:.1%}",
        delta=f"{latest['recession_prob'] - prev['recession_prob']:+.1%} vs 3 months ago",
        delta_color="inverse",
    )
    c2.metric(":material/show_chart: 10Y–2Y Yield Spread", f"{t10y2.iloc[-1]:.2f}%")
    c3.metric(":material/show_chart: 10Y–3M Yield Spread", f"{t10y3.iloc[-1]:.2f}%")
    c4.metric(":material/people: Unemployment Rate",       f"{unemp.iloc[-1]:.1f}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_prob, tab_ind, tab_bt, tab_notes = st.tabs([
    ":material/timeline: Probability Over Time",
    ":material/analytics: Key Indicators",
    ":material/fact_check: Backtesting",
    ":material/info: Model Notes",
])

# ── Tab 1: Probability ────────────────────────────────────────────────────────

with tab_prob:
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
        height=380,
        yaxis=dict(title="P(Recession next 6 months)", tickformat=".0%", range=[0, 1]),
        xaxis_title="",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.05),
        modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**{latest['date'].strftime('%b %Y')}** — "
        f"P(recession next 6 months) = **{latest['recession_prob']:.1%}**. "
        f"Grey bands = NBER-dated US recessions."
    )

# ── Tab 2: Key Indicators ─────────────────────────────────────────────────────

with tab_ind:
    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.markdown("**Yield Curve Spreads** (10Y−2Y and 10Y−3M)")
            fig2 = go.Figure()
            add_recessions(fig2)
            fig2.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4,
                           annotation_text="Inversion threshold", annotation_position="bottom right")
            fig2.add_trace(go.Scatter(x=t10y2.index, y=t10y2.values,
                                      name="10Y-2Y", line=dict(color="#457B9D", width=1.5)))
            fig2.add_trace(go.Scatter(x=t10y3.index, y=t10y3.values,
                                      name="10Y-3M", line=dict(color="#F4A261", width=1.5, dash="dash")))
            fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title="Spread (%)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("An inverted yield curve (< 0) has preceded every US recession since 1955.")

    with col_b:
        with st.container(border=True):
            st.markdown("**Unemployment Rate**")
            fig3 = go.Figure()
            add_recessions(fig3)
            fig3.add_trace(go.Scatter(x=unemp.index, y=unemp.values,
                                      name="UNRATE", line=dict(color="#E63946", width=1.5)))
            fig3.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title="Rate (%)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Rising unemployment is a lagging signal — often peaks after recession ends.")

    with st.container(border=True):
        st.markdown("**Industrial Production Index**")
        fig4 = go.Figure()
        add_recessions(fig4)
        fig4.add_trace(go.Scatter(x=indpro.index, y=indpro.values,
                                  name="INDPRO", line=dict(color="#2DC653", width=1.5)))
        fig4.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Index (2017=100)", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig4, use_container_width=True)

# ── Tab 3: Backtesting ────────────────────────────────────────────────────────

with tab_bt:
    st.markdown("Model predictions vs. realized NBER recessions — evaluating the 6-month forward signal.")

    # Build evaluation dataset
    nber_monthly = nber.resample("MS").last().rename("actual")
    bt = rec_out.set_index("date")[["recession_prob"]].copy()
    bt["eval_date"] = bt.index + pd.DateOffset(months=6)
    bt = bt.reset_index().merge(
        nber_monthly.reset_index().rename(columns={"date": "eval_date"}),
        on="eval_date", how="inner"
    )
    bt = bt[bt["eval_date"] <= pd.Timestamp.now()].dropna()
    bt["predicted"] = (bt["recession_prob"] > 0.5).astype(int)
    bt["actual"]    = bt["actual"].astype(int)

    if len(bt) > 10:
        auc       = roc_auc_score(bt["actual"], bt["recession_prob"])
        precision = precision_score(bt["actual"], bt["predicted"], zero_division=0)
        recall    = recall_score(bt["actual"], bt["predicted"], zero_division=0)
        tp = int(((bt["predicted"] == 1) & (bt["actual"] == 1)).sum())
        fp = int(((bt["predicted"] == 1) & (bt["actual"] == 0)).sum())
        fn = int(((bt["predicted"] == 0) & (bt["actual"] == 1)).sum())
        tn = int(((bt["predicted"] == 0) & (bt["actual"] == 0)).sum())

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(":material/auto_graph: ROC-AUC",  f"{auc:.3f}",
                      help="Area under the ROC curve. 1.0 = perfect, 0.5 = random.")
            m2.metric(":material/precision_manufacturing: Precision", f"{precision:.1%}",
                      help="Of months flagged as pre-recession, what fraction actually were?")
            m3.metric(":material/radar: Recall",      f"{recall:.1%}",
                      help="Of actual pre-recession months, what fraction did the model catch?")
            m4.metric(":material/check_circle: Specificity",
                      f"{tn / (tn + fp):.1%}" if (tn + fp) > 0 else "N/A",
                      help="Of non-recession months, what fraction did the model correctly pass?")

        # Confusion matrix
        with st.expander(":material/table_chart: Confusion matrix (threshold = 50%)"):
            cm_df = pd.DataFrame(
                [[tp, fn], [fp, tn]],
                index=["Actual: Recession", "Actual: No Recession"],
                columns=["Predicted: Recession", "Predicted: No Recession"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(
                f"Evaluated on **{len(bt)} months** where the 6-month outcome is known. "
                f"Training ended Dec 2019; post-2020 months are fully out-of-sample."
            )

        # Backtesting chart
        st.markdown("**Model signal vs. actual recession periods**")
        fig_bt = go.Figure()
        add_recessions(fig_bt, label=True)
        fig_bt.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.4,
                         annotation_text="50% threshold")
        fig_bt.add_trace(go.Scatter(
            x=bt["date"], y=bt["recession_prob"],
            mode="lines", name="P(Recession)",
            line=dict(color="#E63946", width=1.8),
            fill="tozeroy", fillcolor="rgba(230,57,70,0.08)",
        ))
        # Highlight false positives
        fp_mask = (bt["predicted"] == 1) & (bt["actual"] == 0)
        if fp_mask.any():
            fig_bt.add_trace(go.Scatter(
                x=bt.loc[fp_mask, "date"],
                y=bt.loc[fp_mask, "recession_prob"],
                mode="markers", name="False positive",
                marker=dict(color="orange", size=6, symbol="circle"),
            ))
        fig_bt.update_layout(
            height=340,
            yaxis=dict(title="P(Recession)", tickformat=".0%", range=[0, 1]),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.05),
            modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        st.caption(
            "Orange dots = false positives (model > 50% but no recession followed within 6 months). "
            "Grey bands = actual NBER recessions."
        )
    else:
        st.info("Not enough evaluated months yet to compute backtesting metrics.",
                icon=":material/info:")

# ── Tab 4: Model Notes ────────────────────────────────────────────────────────

with tab_notes:
    with st.container(border=True):
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression + isotonic calibration (`CalibratedClassifierCV`) |
| Training window | January 1985 – December 2019 |
| Evaluation window | January 2020 – present (includes COVID recession) |
| Target | NBER recession indicator shifted **−6 months** (forward-looking) |
| Features | 10Y-2Y spread · 10Y-3M spread · ΔUnemployment (3M) · ΔIndustrial Production (3M) · ΔPayrolls (3M) · Sentiment z-score |
| Calibration | Isotonic regression on 5-fold cross-validation |
| Class weighting | Balanced — recessions are ~12% of months since 1985 |
""")

    col_why1, col_why2 = st.columns(2)
    with col_why1:
        with st.container(border=True):
            st.markdown("**Why Logistic Regression?**")
            st.markdown(
                "Interpretable, well-calibrated with isotonic method, and performs reliably "
                "on ~500 monthly training samples. A neural network would overfit on this "
                "dataset size. Calibration ensures P(0.73) means 73% historically correct."
            )
    with col_why2:
        with st.container(border=True):
            st.markdown("**Why 6-month horizon?**")
            st.markdown(
                "Follows the NY Fed recession probability model convention. Provides "
                "actionable lead time for portfolio or policy decisions, while avoiding "
                "the noise of very short-horizon prediction. Labelled explicitly in all outputs."
            )
