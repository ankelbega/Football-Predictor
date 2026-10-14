"""Build leakage-safe match features from the processed match table.

This feature-stage module is called by training, evaluation, CLI prediction, and API serving.
It joins prior results, venue form, rest and Elo ratings without using the current result.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from footballpred.models.elo import EloConfig, add_elo_pre_match_features

FORM_WINDOWS = (5, 10)
METRICS = ("points", "goals_for", "goals_against", "shots", "shots_on_target")


def build_features(
    matches: pd.DataFrame,
    elo_config: EloConfig | None = None,
    form_windows: tuple[int, ...] = FORM_WINDOWS,
) -> pd.DataFrame:
    """Build one causal numeric feature row per match, preserving input row order.

    Args:
        matches: Rows with ``date``, ``home_team``, ``away_team``, ``home_goals`` and
            ``away_goals``. Optional ``season`` and shots columns are used when available.
        elo_config: Elo initialization and home-advantage settings; defaults to ``EloConfig``.
        form_windows: Recent-match counts used for rolling team form.

    Returns:
        Numeric dataframe aligned row-for-row with ``matches``. ``elo_home`` and
        ``elo_away`` are pre-match ratings; unavailable early form/shots are NaN.

    Assumptions:
        Season progress divides prior distinct match dates by 38, the planned matchdays
        for a standard 20-team league; it is capped at one.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Feature input missing columns: {sorted(missing)}")
    data = matches.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if "season" not in data:
        data["season"] = data["date"].dt.year.astype(str)

    # NORMALIZATION: source shot headers vary by season, so use the first recognized alias.
    numeric_inputs = {
        "home_shots": _first_column(data, ("hs", "HS", "home_shots")),
        "away_shots": _first_column(data, ("as", "AS", "away_shots")),
        "home_sot": _first_column(data, ("hst", "HST", "home_shots_on_target")),
        "away_sot": _first_column(data, ("ast", "AST", "away_shots_on_target")),
    }
    for name, column in numeric_inputs.items():
        if column is None:
            data[name] = np.nan
        else:
            data[name] = pd.to_numeric(data[column], errors="coerce")

    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_played: dict[str, pd.Timestamp] = {}
    season_games: dict[tuple[str, str], int] = defaultdict(int)
    feature_rows: dict[int, dict[str, float]] = {}
    grouped = data.assign(row_number=np.arange(len(data))).sort_values(
        ["date", "row_number"], kind="stable"
    )
    for match_date, day in grouped.groupby("date", sort=True):
        # LEAKAGE GUARD: snapshot every fixture first; same-day results cannot cross-match.
        for _, row in day.iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            season = str(row["season"])
            row_number = int(row["row_number"])
            features: dict[str, float] = {}
            features["home_days_rest"] = _days_rest(match_date, last_played.get(home))
            features["away_days_rest"] = _days_rest(match_date, last_played.get(away))
            features["home_matches_season"] = float(season_games[(season, home)])
            features["away_matches_season"] = float(season_games[(season, away)])
            for side, team, venue in (("home", home, "home"), ("away", away, "away")):
                history = histories[team]
                venue_history = [item for item in history if item["venue"] == venue]
                for window in form_windows:
                    _add_form_features(features, side, history[-window:], "overall", window)
                    _add_form_features(features, side, venue_history[-window:], venue, window)
            feature_rows[row_number] = features

        # LEAKAGE GUARD: append this date's scores and shots only after its snapshot is complete.
        for _, row in day.iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            season = str(row["season"])
            hgoals, agoals = int(row["home_goals"]), int(row["away_goals"])
            hpoints = 3 if hgoals > agoals else 1 if hgoals == agoals else 0
            apoints = 3 if agoals > hgoals else 1 if hgoals == agoals else 0
            histories[home].append(
                {
                    "points": float(hpoints), "goals_for": float(hgoals),
                    "goals_against": float(agoals), "shots": _number_or_nan(row["home_shots"]),
                    "shots_on_target": _number_or_nan(row["home_sot"]), "venue": "home",
                }
            )
            histories[away].append(
                {
                    "points": float(apoints), "goals_for": float(agoals),
                    "goals_against": float(hgoals), "shots": _number_or_nan(row["away_shots"]),
                    "shots_on_target": _number_or_nan(row["away_sot"]), "venue": "away",
                }
            )
            last_played[home] = match_date
            last_played[away] = match_date
            season_games[(season, home)] += 1
            season_games[(season, away)] += 1

    feature_frame = pd.DataFrame.from_records(
        [feature_rows[index] for index in range(len(data))], index=data.index
    )
    feature_frame.insert(0, "match_id", np.arange(len(data), dtype=int))
    elo = add_elo_pre_match_features(data, elo_config or EloConfig())
    feature_frame["elo_home"] = elo["elo_home"].to_numpy()
    feature_frame["elo_away"] = elo["elo_away"].to_numpy()
    feature_frame["elo_difference"] = elo["elo_difference"].to_numpy()
    # ASSUMPTION: use 38 matchdays (380 fixtures in a full 20-team league) rather than
    # final season size, which would leak how many matches were eventually played.
    prior_days = data.groupby("season")["date"].rank(method="dense").to_numpy() - 1.0
    feature_frame["season_progress"] = np.minimum(prior_days / 38.0, 1.0)
    return feature_frame.replace([np.inf, -np.inf], np.nan)


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first available column alias, or None when all are absent.

    Args:
        frame: Source dataframe whose headers are inspected.
        candidates: Alias names in preferred order.

    Returns:
        The matched column name, or None.
    """
    return next((name for name in candidates if name in frame.columns), None)


def _days_rest(current: pd.Timestamp, last: pd.Timestamp | None) -> float:
    """Return full rest days before a fixture, or NaN when the team is unseen.

    Args:
        current: Current fixture date.
        last: Previous date on which the team played.

    Returns:
        Difference in whole calendar days, or NaN for a cold start.
    """
    return float((current - last).days) if last is not None else np.nan


def _number_or_nan(value: object) -> float:
    """Convert an optional source statistic to a finite float or NaN.

    Args:
        value: Raw cell value from an optional shots column.

    Returns:
        Finite numeric value when parseable; otherwise NaN for later imputation.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _add_form_features(
    output: dict[str, float],
    side: str,
    history: list[dict[str, Any]],
    venue: str,
    window: int,
) -> None:
    """Add rolling per-match means for the selected team form window.

    Args:
        output: Feature mapping populated in place.
        side: Fixture perspective, either home or away.
        history: Only completed matches before the fixture date.
        venue: ``overall``, ``home``, or ``away`` history label.
        window: Maximum number of matches represented by this history slice.

    Returns:
        None. Missing optional shots remain NaN for the model imputer.
    """
    for metric in METRICS:
        values = np.asarray([row[metric] for row in history], dtype=float)
        if len(values):
            valid = values[np.isfinite(values)]
            output[f"{side}_{venue}_last{window}_{metric}_per_match"] = (
                float(valid.mean()) if len(valid) else np.nan
            )
        else:
            output[f"{side}_{venue}_last{window}_{metric}_per_match"] = np.nan
