"""
Inflation Regime Model
-----------------------
Algorithm : Gaussian Hidden Markov Model (3 hidden states)
States    : Low (<2% CPI), Moderate (2-4%), High (>4%) inflation regimes
Features  : CPI YoY, Core CPI YoY, PCE YoY, inflation expectations
Output    : Per-date probability of being in each regime + most likely state
"""

import json
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from data.store import read_indicators, upsert_model_outputs
from models.base import BaseRiskModel

MODEL_NAME = "inflation"
N_STATES = 3
RANDOM_STATE = 42


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
        X = features.values
        self.model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=200,
            random_state=RANDOM_STATE,
        )
        self.model.fit(X)
        self._label_states(features)

    def _label_states(self, features: pd.DataFrame) -> None:
        """Map HMM state indices to Low/Moderate/High by mean CPI YoY."""
        states = self.model.predict(features.values)
        mean_cpi = {}
        for s in range(N_STATES):
            mask = states == s
            mean_cpi[s] = features.loc[mask, "cpi_yoy"].mean() if mask.any() else 0.0

        sorted_states = sorted(mean_cpi, key=mean_cpi.get)
        labels = ["low", "moderate", "high"]
        self.state_labels = {state: label for state, label in zip(sorted_states, labels)}

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features.values
        # Posterior state probabilities via Viterbi forward-backward
        log_probs = self.model.predict_proba(X)

        rows = []
        for i, (date, log_row) in enumerate(zip(features.index, log_probs)):
            probs_by_label = {
                self.state_labels[s]: float(log_row[s]) for s in range(N_STATES)
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
        print("  Building inflation features...")
        features = self.build_features()
        print(f"  Training HMM on {len(features)} months of data...")
        self.train(features)
        print("  Generating regime probabilities...")
        preds = self.predict(features)

        # Composite inflation risk: weighted toward high-inflation regime
        inflation_risk = preds["prob_moderate"] * 0.4 + preds["prob_high"] * 1.0
        inflation_risk = (inflation_risk - inflation_risk.min()) / (inflation_risk.max() - inflation_risk.min())

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
        print(f"  Stored {len(output)} inflation regime estimates.")
        return preds

    @staticmethod
    def _feature_cols() -> list[str]:
        return ["cpi_yoy", "core_cpi_yoy", "pce_yoy", "inflation_expectations"]


if __name__ == "__main__":
    model = InflationModel()
    preds = model.run()
    print("\nLatest inflation regime probabilities:")
    print(preds[["inflation_state", "prob_low", "prob_moderate", "prob_high"]].tail(3).to_string())
