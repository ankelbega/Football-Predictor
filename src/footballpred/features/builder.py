"""Build historical team form features using only matches on earlier calendar dates."""

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
    """Return one feature row per match, aligned to input ordering.

    Fixtures on the same date are batched: features are read first, then every result from
    that date is added to team histories. Current-match shots and scores only enter histories
    after all fixtures dated that day have had their features materialized.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Feature input missing columns: {sorted(missing)}")
    data = matches.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if "season" not in data:
        data["season"] = data["date"].dt.year.astype(str)

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
        # Feature snapshot for all games on a date is taken before updating histories.
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

        # Update after emitting all pre-match feature rows for this date.
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
    # Progress is strictly causal: fraction of planned season fixtures elapsed before this day,
    # using the conventional 380 match count for a 20-club league and no final season size.
    prior_days = data.groupby("season")["date"].rank(method="dense").to_numpy() - 1.0
    feature_frame["season_progress"] = np.minimum(prior_days / 38.0, 1.0)
    return feature_frame.replace([np.inf, -np.inf], np.nan)


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _days_rest(current: pd.Timestamp, last: pd.Timestamp | None) -> float:
    return float((current - last).days) if last is not None else np.nan


def _number_or_nan(value: object) -> float:
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
    """Add per-game means; missing optional shooting stats remain NaN for LightGBM."""
    for metric in METRICS:
        values = np.asarray([row[metric] for row in history], dtype=float)
        if len(values):
            valid = values[np.isfinite(values)]
            output[f"{side}_{venue}_last{window}_{metric}_per_match"] = (
                float(valid.mean()) if len(valid) else np.nan
            )
        else:
            output[f"{side}_{venue}_last{window}_{metric}_per_match"] = np.nan
