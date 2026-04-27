"""
Recession Probability Model
----------------------------
Algorithm : Logistic Regression with isotonic calibration
Target    : P(NBER recession in the next 6 months)
Features  : yield spreads, unemployment delta, industrial production,
            payrolls growth, consumer sentiment z-score
Training  : 1985-2019  |  Evaluation window: 2020-present
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data.store import read_indicators, upsert_model_outputs
from models.base import BaseRiskModel

TRAIN_END = "2019-12-31"
FORECAST_HORIZON_MONTHS = 6
MODEL_NAME = "recession"


class RecessionModel(BaseRiskModel):
    def __init__(self):
        self.pipeline: Pipeline | None = None

    def build_features(self) -> pd.DataFrame:
        def _get(series_id: str) -> pd.Series:
            df = read_indicators(country_code="USA", series_id=series_id)
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["value"].sort_index()

        rec   = _get("USREC").resample("MS").last()
        t10y2 = _get("T10Y2Y").resample("MS").mean()
        t10y3 = _get("T10Y3M").resample("MS").mean()
        unemp = _get("UNRATE").resample("MS").last()
        indp  = _get("INDPRO").resample("MS").last()
        pay   = _get("PAYEMS").resample("MS").last()
        sent  = _get("UMCSENT").resample("MS").last()

        df = pd.DataFrame({
            "recession":     rec,
            "spread_10y2y":  t10y2,
            "spread_10y3m":  t10y3,
            "unemp_3m_delta": unemp.diff(3),
            "indpro_3m_pct":  indp.pct_change(3) * 100,
            "payems_3m_pct":  pay.pct_change(3) * 100,
        })

        # Sentiment z-score vs 5Y (60-month) rolling mean
        roll = sent.rolling(60, min_periods=24)
        df["sentiment_zscore"] = (sent - roll.mean()) / roll.std()

        # Shift target: predict recession 6 months ahead
        df["target"] = df["recession"].shift(-FORECAST_HORIZON_MONTHS)

        return df.dropna()

    def train(self, features: pd.DataFrame) -> None:
        train = features[features.index <= TRAIN_END]
        X = train[self._feature_cols()]
        y = train["target"].astype(int)

        base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", calibrated),
        ])
        self.pipeline.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[self._feature_cols()]
        probs = self.pipeline.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "date": features.index,
            "recession_prob": probs,
        }).set_index("date")

    def run(self) -> pd.DataFrame:
        print("  Building recession features...")
        features = self.build_features()
        print(f"  Training on {(features.index <= TRAIN_END).sum()} months (1985–2019)...")
        self.train(features)
        print("  Generating predictions...")
        preds = self.predict(features)

        output = pd.DataFrame({
            "country_code":   "USA",
            "model_name":     MODEL_NAME,
            "date":           preds.index,
            "recession_prob": preds["recession_prob"].values,
            "inflation_state": None,
            "inflation_probs": None,
            "composite_risk":  preds["recession_prob"].values,
        })
        upsert_model_outputs(output)
        print(f"  Stored {len(output)} recession probability estimates.")
        return preds

    @staticmethod
    def _feature_cols() -> list[str]:
        return [
            "spread_10y2y", "spread_10y3m", "unemp_3m_delta",
            "indpro_3m_pct", "payems_3m_pct", "sentiment_zscore",
        ]


if __name__ == "__main__":
    model = RecessionModel()
    preds = model.run()
    latest = preds.tail(3)
    print("\nLatest recession probabilities:")
    print(latest.to_string())
