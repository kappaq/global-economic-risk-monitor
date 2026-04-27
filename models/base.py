"""Abstract base class for all risk models."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseRiskModel(ABC):
    @abstractmethod
    def build_features(self) -> pd.DataFrame:
        """Load indicators from store and compute model features."""

    @abstractmethod
    def train(self, features: pd.DataFrame) -> None:
        """Fit the model on historical data."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with probabilistic outputs."""

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """End-to-end: build features, train, predict, persist outputs."""
