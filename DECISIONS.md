# DECISIONS.md — Global Economic Risk Monitor

Key architectural and modeling choices, with rationale and trade-offs considered.

---

## 1. Architecture: Streamlit single-process over React + FastAPI

**Chose:** Streamlit with clean internal separation (data / models / ui layers).

**Why:** The exercise states *"a polished single-process app beats a half-wired API with a broken frontend."* Given a 6–8 hour budget, a separated stack introduces significant wiring overhead (CORS, API contracts, two dev servers) that risks leaving either the backend or frontend incomplete. Streamlit allows Python expertise to carry across all layers while still demonstrating clean separation of concerns through module boundaries.

**Trade-off:** A React + FastAPI stack would score higher on systems design in a production context, and would make the API independently testable. Accepted that trade-off in favour of a working, polished product.

---

## 2. Storage: DuckDB over SQLite or a cloud database

**Chose:** DuckDB (file-based, embedded analytics database).

**Why:** DuckDB is purpose-built for analytical queries — fast columnar scans, native time-series aggregations, and a full SQL interface. No server process needed, so the app runs from a single command without infrastructure setup. Compared to SQLite, DuckDB handles `pct_change`, window functions, and range queries over date-indexed data significantly faster.

**Trade-off:** DuckDB does not support concurrent writes from multiple processes. Acceptable here since ingestion and the app are not running simultaneously.

---

## 3. Primary US data source: FRED API

**Chose:** FRED (Federal Reserve Economic Data) via the `fredapi` Python library.

**Why:** FRED is the authoritative source for US macroeconomic indicators, maintained by the Federal Reserve Bank of St. Louis. It provides all required series (yield spreads, unemployment, industrial production, CPI, VIX) under a free API with a simple key registration. Data is clean, well-documented, and updated on the same schedule as official releases.

**Trade-off:** Requires a free API key. Documented in `.env.example`. Alternative sources (Yahoo Finance, Alpha Vantage) were considered but are less authoritative for macro indicators.

---

## 4. Multi-country data: World Bank API

**Chose:** World Bank Open Data API (no authentication required).

**Why:** Single consistent REST API covering GDP growth, CPI inflation, and unemployment for all target economies (US, UK, Germany, Japan) back to the 1970s. Free, no API key needed, well-maintained.

**Trade-off:** Annual frequency only — insufficient for the US monthly models. Used exclusively for the multi-country choropleth composite score, where annual granularity is appropriate for geographic risk comparison.

---

## 5. Recession model: Logistic Regression with isotonic calibration

**Chose:** `sklearn.linear_model.LogisticRegression` wrapped in `CalibratedClassifierCV(method='isotonic')`.

**Why:** Interpretable, well-calibrated, and appropriate for the dataset size (~500 monthly training samples). The multi-feature approach (6 indicators) demonstrates broader indicator understanding beyond just the yield curve — addressing the exercise's explicit warning that "a threshold on the 10Y-2Y spread hasn't shown us much." Isotonic calibration ensures the output probabilities are meaningful (a 0.73 output is actually correct 73% of the time historically).

**Trade-off:** A gradient-boosted model (XGBoost) or neural network might capture non-linear interactions better. Rejected due to overfitting risk on limited monthly data, and because interpretability matters more in a macro risk context where analysts need to understand *why* the model is signalling.

---

## 6. Recession forecast horizon: 6 months

**Chose:** Target variable = NBER recession indicator shifted −6 months forward.

**Why:** Follows the NY Fed recession probability model convention. Six months provides actionable lead time for portfolio or policy decisions, while avoiding the noise of very short-term prediction. The model is explicit about this horizon: "P(recession in the next 6 months)."

**Trade-off:** A 12-month horizon would give more lead time but degrades signal quality. One month would be near-trivial to predict. Six months is the established macro standard.

---

## 7. Inflation model: Gaussian HMM over threshold rules

**Chose:** `hmmlearn.hmm.GaussianHMM` with 3 hidden states.

**Why:** Inflation evolves through regimes with persistence — once in a high-inflation state, economies tend to stay there. HMMs capture this latent state transition structure naturally. The three states self-organize from data (labeled Low / Moderate / High post-training by mean CPI) rather than being manually defined thresholds, making the model adaptive to different historical regimes.

**Trade-off:** HMMs are harder to interpret than a simple rule (e.g., "CPI > 4% = high"). Mitigated by the deep-dive page showing state probabilities, transition dynamics, and the underlying CPI series in context. An alternative Markov Regime-Switching regression (statsmodels) was considered but added complexity without proportional benefit.

---

## 8. Choropleth scalar: composite risk score

**Chose:** Single composite risk per country: `0.5 × recession_prob + 0.5 × inflation_risk` for the US; z-score normalized proxy for UK/Germany/Japan.

**Why:** A choropleth requires one scalar per geography. The compression from two model outputs to one number is a genuine design decision — chose equal weighting as the most transparent and defensible default. The DECISIONS.md documents this explicitly rather than hiding it.

**Trade-off:** Equal weighting may not reflect a specific analyst's priorities (e.g., recession risk may matter more to an equity portfolio manager). A future enhancement would expose weighting sliders in the UI.

---

## 9. Class imbalance in recession model: balanced weighting

**Chose:** `class_weight='balanced'` in LogisticRegression.

**Why:** NBER recession periods cover roughly 12% of months since 1985. Without balancing, the model learns to predict "no recession" almost always and achieves high accuracy but zero recall on the minority class. Balanced weighting upsamples recession months to give the model signal on rare events.

**Trade-off:** Slightly inflated false-positive rate. Acceptable given the use case — a false alarm is less costly than missing a recession signal in a risk monitoring tool.
