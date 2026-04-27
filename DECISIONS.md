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

---

## 10. LLM risk summary: Claude Haiku with prompt caching, opt-in

**Chose:** `claude-haiku-4-5` via the Anthropic SDK, with `cache_control: ephemeral` on the system prompt. Feature is opt-in — only renders if `ANTHROPIC_API_KEY` is present in the environment.

**Why:** The LLM summary synthesizes all model outputs into a 2–3 paragraph executive brief that no chart can replicate — it explains *what the combination of signals means*, not just what each number is. Haiku was chosen over larger models for two reasons: (1) latency — the dashboard must feel responsive, and Haiku responds in under 2 seconds; (2) cost — the task is structured summarization, not complex reasoning, so paying for Opus would add expense without quality benefit. Prompt caching on the stable system prompt reduces per-request cost by ~90% on repeated loads, and Streamlit's `@st.cache_data(ttl=3600)` avoids calling the API on every page rerun.

**Trade-off:** The feature requires an Anthropic API key and billing setup, which adds friction for a new user. The opt-in design (silent skip when no key is set) ensures the app is fully functional without it — the LLM summary is an enhancement, not a dependency.

---

## 11. Code review process: BASPCT framework

The codebase was reviewed after initial implementation using the **BASPCT** mnemonic — a structured Python code review checklist applied in order:

| Pass | Focus | Key findings & fixes |
|------|-------|----------------------|
| **B — Bug Hunter** | Logic errors, type safety, bounds | `data_is_stale()` was comparing historical macro data dates (always months old) against now — always returned `True`. Fixed to use DB file mtime. Added `iloc` bounds guards in the recession page and Dashboard country filter. |
| **A — Architect** | Package boundaries, separation of concerns, query efficiency | Recession detail page made 5 separate `read_indicators()` calls; replaced with one `read_indicators_multi()` batch query. Import boundaries between data / model / UI layers confirmed clean — no circular imports. |
| **S — Security** | Injection, secrets, exception handling | SQL queries use parameterised placeholders throughout — no f-string injection risk. API keys loaded exclusively from environment variables. Narrowed overly broad `except Exception` in the Anthropic API call to catch `anthropic.APIError` first. |
| **P — Performance** | Redundant I/O, unnecessary computation | Schema SQL was re-read from disk on every DuckDB connection — cached at module level. Yield spread resampling in the recession page was calling `.resample("MS").mean()` on already-monthly data; fixed to apply mean aggregation on raw daily data. |
| **C — Clean Code** | Dead code, misleading comments, PEP 8 | Removed dead `iso2` variable in `ingest.py`. Corrected misleading comment in `inflation.py` that attributed `predict_proba()` to "Viterbi" — it uses the forward-backward algorithm; Viterbi gives a hard state sequence, not probabilities. |
| **T — Tests** | Unit and integration coverage | Added 24 pytest tests across three modules (store, composite model, ingest parsing). All tests run without network calls or a real FRED key — World Bank HTTP responses are monkeypatched; DB is redirected to a `tmp_path` fixture. |

**Running the test suite:**

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

**Why BASPCT over an ad-hoc review?** The ordered pass structure forces a deliberate separation between correctness (B), architecture (A), security (S), performance (P), style (C), and validation (T). Running them in order prevents premature optimisation and ensures bugs are caught before tests are written against broken behaviour.

---

## 12. Multi-agent code review and targeted hardening (post-BASPCT)

A second review pass was run using four specialist AI agents (code-reviewer, data-scientist, data-engineer, Streamlit-UX) in parallel. Their consolidated findings drove a targeted set of fixes applied in priority order:

**Critical fixes applied:**

| Fix | Rationale |
|-----|-----------|
| `get_connection()` converted from plain function to `@contextmanager` | DuckDB's raw connection did not guarantee `close()` on exception; explicit `try/finally` removes that ambiguity. |
| `fetch_fred_series()` raises `RuntimeError` when `rows` is empty | `pd.concat([])` raises `ValueError` — guarded before concatenation. |
| `RecessionModel.predict()` / `InflationModel.predict()` guard against `None` pipeline | Calling predict before train previously raised `AttributeError` with no useful message. |
| HMM trained only on pre-2020 data (`TRAIN_END = "2019-12-31"`) | Training on the full sequence meant post-2019 regime labels were in-sample. Model now trains on 1980–2019; `predict()` runs on full history for out-of-sample evaluation post-2020. |
| Schema file wrapped in `try/except FileNotFoundError` at import | Module-level `read_text()` with no guard raised an opaque `FileNotFoundError` on cold start with a missing file. |

**High-priority model methodology fixes:**

| Fix | Rationale |
|-----|-----------|
| `CalibratedClassifierCV` now uses `TimeSeriesSplit(n_splits=5)` | Default random k-fold cross-validation allows future months to appear in training folds. `TimeSeriesSplit` enforces temporal ordering — training folds always precede validation folds. |
| Recession training trimmed by 6-month forecast horizon | Target is `USREC.shift(-6)`. The last 6 months of training used COVID-era targets (post-cutoff); training cutoff now adjusted to `TRAIN_END - 6 months` to prevent that leakage. |
| HMM state labeling uses `model.means_[:, 0]` (emission means) | Re-predicting on training data to obtain mean CPI per state was circular — state indices can shift between runs. Using learned Gaussian means is deterministic, interpretable, and independent of any prediction call. |
| Inflation risk normalization guards against zero range | `(val - min) / (max - min)` raises `ZeroDivisionError` when all values are identical (e.g., constant series during testing). Falls back to 0.5. |

**UI hardening:**

| Fix | Rationale |
|-----|-----------|
| `refresh_all_data()` wraps pipeline in `try/except`; `st.status` transitions to `state="error"` | Pipeline failures previously left the spinner hanging indefinitely. |
| Deep-dive pages (`1_Recession_Model`, `2_Inflation_Model`) call `st.stop()` on empty data | `iloc[0]` / `.iloc[-1]` on empty DataFrames raised `IndexError` with no user-facing message. |
| `json.loads()` on `inflation_probs` guarded against `NULL` values | Non-US composite rows store `NULL` in that column; `json.loads(None)` raises `TypeError`. |

**Medium and low priority hardening (second pass):**

| Fix | File(s) | Rationale |
|-----|---------|-----------|
| `str \| None` type hints on nullable params | `data/store.py` | `read_indicators()` and `read_model_outputs()` accepted `None` but declared `str` — misleading for any type-checker or reader. |
| Unified `_to_timestamp()` date normalizer | `data/ingest.py` | FRED returned tz-aware `DatetimeIndex` with time components; World Bank dates were plain year strings. Both now pass through `_to_timestamp()` which strips timezone and normalizes to midnight before DB insertion — one canonical path instead of two ad-hoc patterns. |
| Failed FRED series surfaced in the UI | `data/ingest.py`, `Dashboard.py` | `fetch_fred_series()` now returns `(DataFrame, failed_series_list)`. `run_pipeline()` propagates the list. `refresh_all_data()` shows a `st.warning` naming any missing series, so a partial fetch is visible to the operator rather than silently producing an incomplete model. |
| DDL skipped on repeat connections | `data/store.py` | `CREATE TABLE IF NOT EXISTS` is idempotent but added measurable overhead per connection. A module-level `_schema_applied_for` string tracks which DB path has been initialised; DDL runs only once per process. |
| World Bank retry on HTTP 429 | `data/ingest.py` | Public API rate-limits with 429. Added exponential backoff (1 s, 2 s) for up to 3 attempts before propagating the error — eliminates transient failures during demo. |
| `print()` replaced with `logging` | `data/ingest.py`, `models/*.py`, `Dashboard.py` | `print()` bypasses Python's logging infrastructure — no timestamps, no severity levels, no ability to suppress in tests. All pipeline output now uses `logging.getLogger(__name__)`. `Dashboard.py` calls `basicConfig` at startup so logs appear in the Streamlit terminal. |
| Test mock updated with `status_code` | `tests/test_ingest.py` | The World Bank retry logic reads `resp.status_code`; the existing `_MockResponse` fixture lacked the attribute, causing 5 test failures after the retry fix. Added `status_code = 200` default. |
