"""Chronological season and validation/test split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Train/validation/test frames with chronological boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_season: str
    test_seasons: tuple[str, ...]


def season_split(
    matches: pd.DataFrame,
    validation_season: str,
    test_seasons: tuple[str, ...] | list[str],
) -> TemporalSplit:
    """Split by configured season labels; training seasons must precede validation and test."""
    if "season" not in matches.columns:
        raise ValueError("Input matches require a season column")
    seasons = matches["season"].astype(str)
    test_tuple = tuple(str(season) for season in test_seasons)
    if not test_tuple:
        raise ValueError("At least one test season is required")
    if validation_season in test_tuple:
        raise ValueError("Validation season cannot also be a test season")
    all_seasons = sorted(seasons.unique(), key=_season_start)
    if validation_season not in all_seasons or any(
        season not in all_seasons for season in test_tuple
    ):
        missing = [s for s in (validation_season, *test_tuple) if s not in all_seasons]
        raise ValueError(f"Configured season(s) absent from data: {missing}")
    validation_pos = all_seasons.index(validation_season)
    test_positions = [all_seasons.index(season) for season in test_tuple]
    if any(position <= validation_pos for position in test_positions):
        raise ValueError("Every test season must be later than the validation season")
    train_seasons = set(all_seasons[:validation_pos])
    test_season_set = set(test_tuple)
    train = matches.loc[seasons.isin(train_seasons)].copy()
    validation = matches.loc[seasons.eq(validation_season)].copy()
    test = matches.loc[seasons.isin(test_season_set)].copy()
    _assert_chronological(train, validation, test)
    return TemporalSplit(train, validation, test, validation_season, test_tuple)


def _season_start(season: str) -> int:
    """Sort football-data season labels (e.g. 1415, 2021, 2526) chronologically."""
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"Cannot order season label {season!r}; expected four digits")
    first = int(season[:2])
    return 2000 + first if first <= 79 else 1900 + first


def _assert_chronological(*frames: pd.DataFrame) -> None:
    dates = [pd.to_datetime(frame["date"]) for frame in frames if not frame.empty]
    for earlier, later in zip(dates, dates[1:], strict=False):
        if earlier.max() >= later.min():
            raise ValueError("Temporal partitions overlap or are not strictly chronological")
