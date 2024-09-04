"""Majority classifier."""

import pandas as pd


class MajorityClassifier:
    """Majority classifier."""

    def __init__(self: "MajorityClassifier") -> None:
        """Initialize."""
        self.probability_of_fraud = 0.5

    def fit(self: "MajorityClassifier", y: pd.Series) -> "MajorityClassifier":
        """Fit.

        Args:
            y (pd.Series): Target data

        Returns:
            MajorityClassifier: self
        """
        self.probability_of_fraud = y.mean()
        return self

    def predict(self: "MajorityClassifier", x: pd.DataFrame) -> pd.Series:
        """Predict.

        Args:
            x (pd.DataFrame): Input data

        Returns:
            pd.Series: Predicted data
        """
        probability_50_percent = 0.5
        return pd.Series((self.predict_proba(x) > probability_50_percent).astype(int))

    def predict_proba(self: "MajorityClassifier", x: pd.DataFrame) -> pd.Series:
        """Predict.

        Args:
            x (pd.DataFrame): Input data

        Returns:
            pd.Series: Predicted data
        """
        return pd.Series([self.probability_of_fraud] * len(x))
