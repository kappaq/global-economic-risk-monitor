"""
Inflation Regime Model
-----------------------
Algorithm : Gaussian Hidden Markov Model (3 hidden states)
States    : Low (<2% CPI), Moderate (2-4%), High (>4%) inflation regimes
Features  : CPI YoY, Core CPI YoY, PCE YoY, inflation expectations
Output    : Per-date probability of being in each regime + most likely state
"""

import json
import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from data.store import read_indicators, upsert_model_outputs
from models.base import BaseRiskModel

logger = logging.getLogger(__name__)

MODEL_NAME = "inflation"
N_STATES = 3
RANDOM_STATE = 42
TRAIN_END = "2019-12-31"


class InflationModel(BaseRiskModel):
    def __init__(self):
        self.model: GaussianHMM | None = None
        self.state_labels: dict[int, str] = {}

    def build_features(self) -> pd.DataFrame:
        def _get(series_id: str) -> pd.Series:
            df = read_indicators(country_code="USA", series_id=series_id)
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["value"].sort_index()

        cpi      = _get("CPIAUCSL").resample("MS").last()
        core_cpi = _get("CPILFESL").resample("MS").last()
        pce      = _get("PCEPI").resample("MS").last()
        mich     = _get("MICH").resample("MS").last()

        df = pd.DataFrame({
            "cpi_yoy":               cpi.pct_change(12, fill_method=None) * 100,
            "core_cpi_yoy":          core_cpi.pct_change(12, fill_method=None) * 100,
            "pce_yoy":               pce.pct_change(12, fill_method=None) * 100,
            "inflation_expectations": mich,
        }).dropna()

        return df

    def train(self, features: pd.DataFrame) -> None:
        # Train only on pre-2020 data so post-2019 predictions are out-of-sample
        train_features = features[features.index <= TRAIN_END]
        X = train_features.values

        # HMM Baum-Welch can land in local optima where two states collapse into one.
        # Try up to MAX_SEEDS initializations and keep the highest-likelihood model
        # whose emission means satisfy the three separation checks.
        MAX_SEEDS = 30
        best_model, best_score = None, -np.inf
        for seed in range(MAX_SEEDS):
            candidate = GaussianHMM(
                n_components=N_STATES,
                covariance_type="full",
                n_iter=200,
                random_state=seed,
            )
            candidate.fit(X)
            score = candidate.score(X)
            if score <= best_score:
                continue
            low, mid, high = self._sorted_cpi_means(candidate)
            if not self._means_are_valid(low, mid, high):
                logger.debug("Seed %d rejected (score=%.1f): means low=%.2f mid=%.2f high=%.2f",
                             seed, score, low, mid, high)
                continue
            best_score = score
            best_model = candidate
            logger.debug("Seed %d accepted (score=%.1f): means low=%.2f mid=%.2f high=%.2f",
                         seed, score, low, mid, high)

        if best_model is None:
            raise RuntimeError(
                f"HMM could not find well-separated states across {MAX_SEEDS} random seeds. "
                "Check that training data covers sufficient historical CPI variation."
            )

        logger.info("Best HMM seed selected with log-likelihood %.2f.", best_score)
        self.model = best_model
        self._label_states()

    @staticmethod
    def _sorted_cpi_means(model: GaussianHMM) -> tuple[float, float, float]:
        """Return (low, mid, high) CPI YoY emission means sorted ascending."""
        means = sorted(float(model.means_[s, 0]) for s in range(N_STATES))
        return means[0], means[1], means[2]

    @staticmethod
    def _means_are_valid(low: float, mid: float, high: float) -> bool:
        return high >= 3.5 and low <= 3.0 and (mid - low) >= 1.0 and (high - mid) >= 1.0

    def _label_states(self) -> None:
        """Map HMM state indices to Low/Moderate/High using emission means (model.means_).

        Uses the learned Gaussian means directly instead of re-predicting on training data,
        which avoids circular state assignment and is more numerically stable.
        Feature order: [cpi_yoy, core_cpi_yoy, pce_yoy, inflation_expectations].
        """
        # means_ shape: (n_components, n_features); index 0 = cpi_yoy
        cpi_means = {s: float(self.model.means_[s, 0]) for s in range(N_STATES)}
        sorted_states = sorted(cpi_means, key=cpi_means.get)
        labels = ["low", "moderate", "high"]
        self.state_labels = {state: label for state, label in zip(sorted_states, labels)}

        low_mean  = cpi_means[sorted_states[0]]
        mid_mean  = cpi_means[sorted_states[1]]
        high_mean = cpi_means[sorted_states[2]]

        logger.info(
            "HMM state emission means — low: %.2f%% CPI, moderate: %.2f%% CPI, high: %.2f%% CPI",
            low_mean, mid_mean, high_mean,
        )

        # The high state should clearly exceed the Fed target — training data includes the
        # 1980s when CPI peaked at ~14%, so anything below 3.5% suggests state collapse.
        if high_mean < 3.5:
            raise RuntimeError(
                f"HMM 'high' state has a CPI YoY mean of {high_mean:.2f}% — below the 3.5% sanity "
                f"threshold. States may have collapsed or training data is insufficient. "
                f"All means: low={low_mean:.2f}%, moderate={mid_mean:.2f}%, high={high_mean:.2f}%."
            )

        # The low state should sit at or below the Fed target range.
        if low_mean > 3.0:
            raise RuntimeError(
                f"HMM 'low' state has a CPI YoY mean of {low_mean:.2f}% — above 3.0%. "
                f"The model has not captured a genuine low-inflation regime (Great Moderation). "
                f"All means: low={low_mean:.2f}%, moderate={mid_mean:.2f}%, high={high_mean:.2f}%."
            )

        # Adjacent states must be meaningfully separated; < 1% gap signals near-degenerate states.
        if (mid_mean - low_mean) < 1.0 or (high_mean - mid_mean) < 1.0:
            raise RuntimeError(
                f"HMM states are not sufficiently separated (need ≥ 1% CPI gap between adjacent "
                f"states). States may have collapsed. "
                f"All means: low={low_mean:.2f}%, moderate={mid_mean:.2f}%, high={high_mean:.2f}%. "
                f"Try increasing n_iter or re-running with a different random_state."
            )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("InflationModel has not been trained. Call train() or run() first.")
        X = features.values
        # Smoothed posterior state probabilities via forward-backward algorithm
        state_probs = self.model.predict_proba(X)

        rows = []
        for date, prob_row in zip(features.index, state_probs):
            probs_by_label = {
                self.state_labels[s]: float(prob_row[s]) for s in range(N_STATES)
            }
            most_likely = max(probs_by_label, key=probs_by_label.get)
            rows.append({
                "date":             date,
                "inflation_state":  most_likely,
                "inflation_probs":  json.dumps(probs_by_label),
                "prob_low":         probs_by_label["low"],
                "prob_moderate":    probs_by_label["moderate"],
                "prob_high":        probs_by_label["high"],
            })
        return pd.DataFrame(rows).set_index("date")

    def run(self) -> pd.DataFrame:
        logger.info("Building inflation features...")
        features = self.build_features()
        logger.info("Training HMM on %d months of data...", len(features))
        self.train(features)
        logger.info("Generating regime probabilities...")
        preds = self.predict(features)

        # Composite inflation risk: weighted toward high-inflation regime
        inflation_risk = preds["prob_moderate"] * 0.4 + preds["prob_high"] * 1.0
        _range = inflation_risk.max() - inflation_risk.min()
        if _range > 0:
            inflation_risk = (inflation_risk - inflation_risk.min()) / _range
        else:
            inflation_risk = pd.Series(0.5, index=inflation_risk.index)

        output = pd.DataFrame({
            "country_code":   "USA",
            "model_name":     MODEL_NAME,
            "date":           preds.index,
            "recession_prob":  None,
            "inflation_state": preds["inflation_state"].values,
            "inflation_probs": preds["inflation_probs"].values,
            "composite_risk":  inflation_risk.values,
        })
        upsert_model_outputs(output)
        logger.info("Stored %d inflation regime estimates.", len(output))
        return preds

    @staticmethod
    def _feature_cols() -> list[str]:
        return ["cpi_yoy", "core_cpi_yoy", "pce_yoy", "inflation_expectations"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model = InflationModel()
    preds = model.run()
    print("\nLatest inflation regime probabilities:")
    print(preds[["inflation_state", "prob_low", "prob_moderate", "prob_high"]].tail(3).to_string())
