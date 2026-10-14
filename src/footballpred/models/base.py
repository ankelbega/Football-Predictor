"""Define the interchangeable probability-model contract and vector checks.

This model-layer base is imported by baselines and fitted classifiers; evaluation expects every
implementation to fit from matches and emit probability columns in one fixed outcome order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ProbabilityModel(ABC):
    """Abstract interface for models predicting home win, draw, and away win."""

    name: str
    classes_: np.ndarray

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> ProbabilityModel:
        """Fit parameters on training matches.

        Args:
            matches: Chronological processed rows, typically with date, teams, goals and result.

        Returns:
            This fitted model instance.
        """

    @abstractmethod
    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Return one probability vector per fixture in home/draw/away order.

        Args:
            matches: Fixture or historical match rows for prediction.

        Returns:
            An ``(n_matches, 3)`` matrix whose rows sum to one.
        """


def validate_probabilities(probabilities: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Reject malformed probability rows and return a clipped float array.

    Args:
        probabilities: Candidate matrix with three class columns.
        tolerance: Allowed floating-point slack near zero, one and row sum one.

    Returns:
        The same probabilities represented as floats and clipped for tiny boundary errors.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected an (n_matches, 3) probability matrix")
    if (
        not np.isfinite(values).all()
        or (values < -tolerance).any()
        or (values > 1 + tolerance).any()
    ):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise ValueError("Each probability vector must sum to 1")
    return np.clip(values, 0.0, 1.0)
