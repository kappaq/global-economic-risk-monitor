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

### Optional: LLM Risk Summary

Add your Anthropic API key to `.env` to enable the AI-generated macro risk brief:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The section is silently skipped if the key is absent — the app works fully without it.

---

## Docker

```bash
docker compose up --build
```

Opens at http://localhost:8501. API keys are read from your `.env` file automatically.

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
│   ├── ingest.py           # FRED + World Bank fetchers
│   ├── store.py            # DuckDB read/write layer
│   └── schema.sql          # Table definitions
├── models/
│   ├── base.py             # Abstract base model class
│   ├── recession.py        # Logistic regression recession model
│   ├── inflation.py        # Gaussian HMM inflation regime model
│   └── composite.py        # Multi-country composite risk scorer
├── pages/
│   ├── 1_Recession_Model.py    # Recession deep-dive page
│   └── 2_Inflation_Model.py    # Inflation deep-dive page
├── tests/
│   ├── conftest.py         # Shared pytest fixtures (temp DB, seeded data)
│   ├── test_store.py       # Store layer: upsert, read, staleness
│   ├── test_composite.py   # Composite model: zscore, scoring, labels
│   └── test_ingest.py      # World Bank response parsing
├── Dashboard.py            # Main Streamlit dashboard
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Single-command Docker startup
├── DECISIONS.md            # Key design choices and trade-offs
├── requirements.txt        # Runtime dependencies
└── requirements-dev.txt    # Test dependencies (pytest)
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

24 tests covering the store layer, composite risk scorer, and World Bank ingest parsing. All tests run without network calls or a live FRED key.

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
- **States**: Low / Moderate / High — labelled post-training by sorting learned CPI YoY emission means (not fixed thresholds)
- **Features**: CPI YoY, Core CPI YoY, PCE YoY, 1Y inflation expectations
- **Output**: Smoothed posterior probability of being in each regime at each date (forward-backward algorithm)
- **Robustness**: Training retries up to 30 random seeds and selects the highest-likelihood model whose emission means satisfy separation checks, preventing state collapse

### Probability transparency

Every metric in the app surfaces four dimensions via inline tooltip (`?`):

| Dimension | Example (US recession probability) |
|---|---|
| **Probability of what** | P(NBER recession starts within the next 6 months) |
| **Horizon** | 6-month forward |
| **Calibrated how** | Isotonic calibration, 5-fold TimeSeriesSplit, trained 1985–2019 |
| **Uncertainty** | Point estimate — no confidence interval computed |

Non-US "Recession Risk" is labelled **Recession Stress Score** to distinguish it from the US calibrated probability — it is a normalized GDP-growth index, not a model-derived probability.

See [DECISIONS.md](DECISIONS.md) for detailed rationale behind every design choice.
