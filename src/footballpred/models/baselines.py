"""Simple class-frequency and bookmaker probability reference models.

This model-stage module is used by evaluation and comparison reports. It turns results or
decimal odds into normalized probabilities so complex models have honest reference forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footballpred.models.base import ProbabilityModel, validate_probabilities

OUTCOMES = ("H", "D", "A")


def outcome_probabilities(matches: pd.DataFrame) -> np.ndarray:
    """Encode observed outcomes as one-hot rows ordered H, D, A.

    Args:
        matches: Dataframe with a ``result`` column containing H, D or A.

    Returns:
        Float array with one row and three columns per match.
    """
    mapping = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    if not matches["result"].isin(mapping).all():
        raise ValueError("Matches contain unexpected result values")
    return np.eye(3, dtype=float)[matches["result"].map(mapping).to_numpy()]


class ClassPriorModel(ProbabilityModel):
    """Use training outcome frequencies as the same forecast for every fixture."""

    name = "class_prior"
    classes_ = np.asarray(OUTCOMES)

    def __init__(self) -> None:
        """Create a uniform prior until fit receives training outcomes."""
        self.priors_ = np.full(3, 1 / 3)

    def fit(self, matches: pd.DataFrame) -> ClassPriorModel:
        """Estimate empirical H/D/A rates from training matches.

        Args:
            matches: Training dataframe with a valid H/D/A ``result`` column.

        Returns:
            This fitted baseline. Empty input is rejected rather than inventing a prior.
        """
        if matches.empty:
            raise ValueError("Cannot estimate class priors from an empty training set")
        self.priors_ = outcome_probabilities(matches).mean(axis=0)
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Repeat fitted outcome frequencies once per requested match.

        Args:
            matches: Any fixture dataframe; only its row count is used.

        Returns:
            Valid ``(n_matches, 3)`` probability matrix.
        """
        return validate_probabilities(np.tile(self.priors_, (len(matches), 1)))


def bookmaker_probabilities(matches: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Use closing odds when complete, otherwise best available complete market prices.

    Available odds triplets are normalized per row to remove the overround proportionally.
    The per-row fallback order is Pinnacle closing, Bet365, then any generic Avg triplet.

    Args:
        matches: Fixture rows containing at least one complete three-way odds triplet.

    Returns:
        H/D/A probability matrix and combined source label. Missing valid odds raise rather
        than inventing a replacement probability.
    """
    # NORMALIZATION: 1/decimal odds are implied probabilities; dividing by their sum removes
    # the bookmaker margin proportionally. Fallback is per fixture, not per whole file.
    candidates = (
        ("psch", "pscd", "psca", "pinnacle_closing"),
        ("b365h", "b365d", "b365a", "bet365"),
        ("avgh", "avgd", "avga", "average_odds"),
    )
    result = np.full((len(matches), 3), np.nan, dtype=float)
    source_by_row = np.full(len(matches), "", dtype=object)
    used: set[str] = set()
    for home_col, draw_col, away_col, label in candidates:
        if not all(column in matches.columns for column in (home_col, draw_col, away_col)):
            continue
        odds = matches[[home_col, draw_col, away_col]].apply(pd.to_numeric, errors="coerce")
        values = odds.to_numpy(dtype=float)
        # NORMALIZATION: decimal odds at or below 1 cannot represent a valid bookmaker price.
        valid = np.isfinite(values).all(axis=1) & (values > 1.0).all(axis=1)
        open_rows = ~np.isfinite(result).all(axis=1)
        use = valid & open_rows
        if use.any():
            # NORMALIZATION: divide inverse odds by their per-row sum to remove overround.
            implied = 1 / values[use]
            result[use] = implied / implied.sum(axis=1, keepdims=True)
            source_by_row[use] = label
            used.add(label)
    valid_rows = np.isfinite(result).all(axis=1)
    if not valid_rows.all():
        missing = int((~valid_rows).sum())
        raise ValueError(f"No valid complete odds triplet for {missing} match(es)")
    # `+` marks a mixed per-match source fallback across the sample.
    source = str(source_by_row[0]) if len(used) == 1 else "+".join(sorted(used))
    return validate_probabilities(result), source


class BookmakerModel(ProbabilityModel):
    """Market-implied probabilities with proportional overround removal."""

    name = "bookmaker"
    classes_ = np.asarray(OUTCOMES)

    def fit(self, matches: pd.DataFrame) -> BookmakerModel:
        """Return self because market probabilities are read from each fixture's odds.

        Args:
            matches: Unused; retained to match the interchangeable model interface.

        Returns:
            This stateless bookmaker model.
        """
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Convert fixture odds into no-vig H/D/A probabilities.

        Args:
            matches: Fixture rows with at least one complete valid odds triplet.

        Returns:
            Rows of probabilities in H/D/A order. Invalid or incomplete odds raise ValueError.
        """
        return bookmaker_probabilities(matches)[0]
