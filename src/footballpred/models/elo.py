"""Sequential Elo updates and a draw-aware probability mapping.

This model-stage module is used by feature engineering and evaluation. Ratings update from
chronological results; a small mapping fit on training outcomes converts rating differences
to home/draw/away probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from footballpred.models.base import ProbabilityModel, validate_probabilities
from footballpred.models.baselines import OUTCOMES, outcome_probabilities


@dataclass(frozen=True)
class EloConfig:
    """Elo starting point, venue advantage, and update sensitivity.

    Args:
        initial_rating: Neutral rating assigned to unseen teams (conventionally 1500).
        home_advantage: Rating points added to the home side only when computing expectation.
        k_factor: Rating change scale; larger values make ratings react more quickly.
        goal_difference_multiplier: Whether larger score margins increase the update size.
    """

    initial_rating: float = 1500.0
    home_advantage: float = 65.0
    k_factor: float = 20.0
    goal_difference_multiplier: bool = True


def elo_expected_score(rating_difference: float) -> float:
    """Map rating-point difference to expected result score using Elo's logistic curve.

    Args:
        rating_difference: Home rating plus home advantage minus away rating.

    Returns:
        Expected home score between zero and one. A draw counts as half a result point.
    """
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def goal_margin_multiplier(goal_difference: int) -> float:
    """Scale updates for goal margin while dampening extreme scorelines.

    Args:
        goal_difference: Home goals minus away goals.

    Returns:
        One for a one-goal margin; a logarithmic increase for larger margins.
    """
    # ASSUMPTION: the conventional logarithmic multiplier tempers blowout scores, which are
    # less informative about underlying strength than repeated close results.
    margin = abs(int(goal_difference))
    if margin <= 1:
        return 1.0
    return float(np.log1p(margin) * 2.2 / (0.001 * 1500 + 2.2))


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_goals: int,
    away_goals: int,
    config: EloConfig,
) -> tuple[float, float]:
    """Update both teams by equal and opposite rating changes.

    Args:
        home_rating: Home team's rating before the match.
        away_rating: Away team's rating before the match.
        home_goals: Observed home score.
        away_goals: Observed away score.
        config: Elo settings.

    Returns:
        New ``(home_rating, away_rating)`` values; their sum is unchanged.

    Formula:
        Change is K times the margin multiplier times (actual home result score minus
        expected home score). Home advantage changes expectation, not rating conservation.
    """
    expected_home = elo_expected_score(home_rating + config.home_advantage - away_rating)
    actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    multiplier = (
        goal_margin_multiplier(home_goals - away_goals)
        if config.goal_difference_multiplier
        else 1.0
    )
    # Zero-sum update: the away team gets exactly the negative of the home team's change.
    delta = config.k_factor * multiplier * (actual_home - expected_home)
    return home_rating + delta, away_rating - delta


def add_elo_pre_match_features(matches: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """Attach home and away Elo ratings immediately before each match.

    Args:
        matches: Dataframe with date, team names, and final goals; results update later rows.
        config: Elo initialization and update settings.

    Returns:
        Copy with ``elo_home``, ``elo_away``, and ``elo_difference`` columns. Original row
        order is retained.

    Assumption:
        Kickoff times are absent, so all fixtures on the same calendar date share pre-date
        ratings and are applied only after all features for that date have been recorded.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Elo input missing columns: {sorted(missing)}")
    # Required match-shaped columns: date, home_team, away_team, home_goals, away_goals.
    ordered = matches.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered["_original_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["date", "_original_order"], kind="stable")
    ratings: dict[str, float] = {}
    outputs: dict[int, tuple[float, float]] = {}
    for _, day in ordered.groupby("date", sort=True):
        # LEAKAGE GUARD: every fixture on a date sees ratings from strictly earlier dates.
        for index, match in day.iterrows():  # noqa: B007 -- preserve original iterator binding
            home = str(match["home_team"])
            away = str(match["away_team"])
            outputs[int(match["_original_order"])] = (
                ratings.get(home, config.initial_rating),
                ratings.get(away, config.initial_rating),
            )
        for _, match in day.iterrows():
            home = str(match["home_team"])
            away = str(match["away_team"])
            home_rating = ratings.get(home, config.initial_rating)
            away_rating = ratings.get(away, config.initial_rating)
            ratings[home], ratings[away] = update_ratings(
                home_rating,
                away_rating,
                int(match["home_goals"]),
                int(match["away_goals"]),
                config,
            )
    feature_values = [outputs[index] for index in range(len(matches))]
    result = matches.copy()
    result["elo_home"] = [item[0] for item in feature_values]
    result["elo_away"] = [item[1] for item in feature_values]
    result["elo_difference"] = result["elo_home"] + config.home_advantage - result["elo_away"]
    return result


class EloModel(ProbabilityModel):
    """Elo features plus a training-fitted draw-aware mapping to 1X2 probabilities.

    Home, draw and away labels use the shared H/D/A column order. Hyperparameters may be
    selected on a later validation season; this class itself only fits the mapping on input.
    The mapping uses opposing logistic tails for home/away and assigns the remaining mass to draw.
    """

    # Fit the draw-aware probability mapping on training rows after chronological rating replay.
    name = "elo"
    classes_ = np.asarray(OUTCOMES)

    def __init__(self, config: EloConfig | None = None) -> None:
        """Create an unfitted Elo probability model.

        Args:
            config: Elo rating settings, or the documented defaults.
        """
        self.config = config or EloConfig()
        self.mapping_: np.ndarray | None = None
        self._fitted = False

    def fit(self, matches: pd.DataFrame) -> EloModel:
        """Fit the draw-aware rating-to-probability mapping on training matches.

        Args:
            matches: Chronological match rows with date, teams, goals, and H/D/A result.

        Returns:
            This fitted model instance.
        """
        features = add_elo_pre_match_features(matches, self.config)
        x = features["elo_difference"].to_numpy(dtype=float) / 400.0
        y = outcome_probabilities(features)

        # Home and away cumulative curves leave the middle probability for a draw.
        def probabilities(parameters: np.ndarray) -> np.ndarray:
            """Use opposing logistic tails as home/away chances and their remainder as draw."""
            slope = np.exp(parameters[0])
            home_threshold = parameters[1]
            away_threshold = parameters[2]
            cumulative_home = expit(slope * x - home_threshold)
            cumulative_away = expit(-slope * x - away_threshold)
            draw = np.clip(1.0 - cumulative_home - cumulative_away, 1e-8, 1.0)
            p = np.column_stack([cumulative_home, draw, cumulative_away])
            return p / p.sum(axis=1, keepdims=True)

        def objective(parameters: np.ndarray) -> float:
            """Fit mapping parameters by minimizing training-label negative log likelihood."""
            return float(-np.sum(y * np.log(np.clip(probabilities(parameters), 1e-12, 1.0))))

        initial = np.array([0.0, 1.0, 1.0])
        bounds = [(-3.0, 3.0), (-5.0, 5.0), (-5.0, 5.0)]
        fitted = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        if not fitted.success and not np.isfinite(fitted.fun):
            raise RuntimeError(f"Failed to fit Elo probability mapping: {fitted.message}")
        self.mapping_ = fitted.x
        self._fitted = True
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Return probabilities ordered home win, draw, away win.

        Args:
            matches: Fixture rows with date, teams, and goals (goals are used only by the
                sequential rating builder for fixtures earlier than each predicted date).

        Returns:
            A valid ``(n_matches, 3)`` probability matrix.
        """
        if not self._fitted or self.mapping_ is None:
            raise RuntimeError("EloModel.fit must be called before predict_proba")
        features = add_elo_pre_match_features(matches, self.config)
        x = features["elo_difference"].to_numpy(dtype=float) / 400.0
        slope = np.exp(self.mapping_[0])
        p_home = expit(slope * x - self.mapping_[1])
        p_away = expit(-slope * x - self.mapping_[2])
        p_draw = np.clip(1 - p_home - p_away, 1e-8, 1.0)
        probabilities = np.column_stack([p_home, p_draw, p_away])
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return validate_probabilities(probabilities)
