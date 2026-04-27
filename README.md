# Global Economic Risk Monitor

A prototype application for visually assessing **recession risk** and **inflation regime risk** across the United States, United Kingdom, Germany, and Japan — with a geographic map view and forward-looking probabilistic model outputs.

Built as a hands-on data engineering and modeling exercise using an AI-assisted workflow (Claude Code).

---

## What It Does

- **Ingests** macroeconomic indicators from [FRED](https://fred.stlouisfed.org/) (US) and the [World Bank API](https://datahelpdesk.worldbank.org/) (multi-country) into a local DuckDB store
- **Recession model**: Calibrated logistic regression predicting P(recession in next 6 months) using yield spreads, unemployment, industrial production, payrolls, and consumer sentiment
- **Inflation regime model**: Gaussian Hidden Markov Model classifying the economy into Low / Moderate / High inflation regimes with per-date probabilities
- **Dashboard**: Interactive Plotly choropleth map + time-series overlays + model deep-dive pages

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A free FRED API key → [register here](https://fred.stlouisfed.org/docs/api/api_key.html) (takes ~1 minute)

### 2. Clone and configure

```bash
git clone <repo-url>
cd global-economic-risk-monitor

cp .env.example .env
# Edit .env and set: FRED_API_KEY=your_key_here
```

### 3. Run

```bash
bash run.sh
```

This will:
1. Install dependencies
2. Fetch data from FRED and World Bank into DuckDB
3. Train and run both models
4. Launch the Streamlit dashboard at http://localhost:8501

---

## Manual Steps (if preferred)

```bash
pip install -r requirements.txt

# Ingest data
python -m data.ingest

# Run models
python -m models.recession
python -m models.inflation
python -m models.composite

# Launch app
streamlit run Dashboard.py
```

The pipeline is **idempotent** — safe to re-run at any time. The app also has a **Refresh Data** button that re-runs the full pipeline from the UI.

---

## Project Structure

```
├── data/
│   ├── ingest.py       # FRED + World Bank fetchers
│   ├── store.py        # DuckDB read/write layer
│   └── schema.sql      # Table definitions
├── models/
│   ├── base.py         # Abstract base model class
│   ├── recession.py    # Logistic regression recession model
│   ├── inflation.py    # Gaussian HMM inflation regime model
│   └── composite.py    # Multi-country composite risk scorer
├── pages/
│   ├── 1_Recession_Model.py   # Recession deep-dive page
│   └── 2_Inflation_Model.py   # Inflation deep-dive page
├── app.py              # Main Streamlit dashboard
├── DECISIONS.md        # Key design choices and trade-offs
└── requirements.txt
```

---

## Data Sources

| Source | Coverage | Series |
|--------|----------|--------|
| FRED API | United States, monthly | Yield spreads, unemployment, CPI, industrial production, VIX, and more |
| World Bank API | US, UK, Germany, Japan, annual | GDP growth, CPI inflation, unemployment |

---

## Models

### Recession Probability
- **Algorithm**: Logistic Regression + isotonic calibration
- **Target**: P(NBER recession in next 6 months)
- **Features**: 10Y-2Y spread, 10Y-3M spread, Δunemployment, Δindustrial production, Δpayrolls, consumer sentiment z-score
- **Training**: 1985–2019 | **Evaluation**: 2020–present

### Inflation Regime
- **Algorithm**: Gaussian Hidden Markov Model (3 hidden states)
- **States**: Low (<2%) / Moderate (2–4%) / High (>4%) inflation regimes
- **Features**: CPI YoY, Core CPI YoY, PCE YoY, 1Y inflation expectations
- **Output**: Probability of being in each regime per date

See [DECISIONS.md](DECISIONS.md) for detailed rationale behind every design choice.
