"""Create ordered train, validation and test season partitions.

This evaluate-stage module is called before tuning and walk-forward scoring. It uses season
labels plus actual date checks so later forecasts cannot train on future results.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Container for match frames and the season labels used to create the split.

    Args:
        train: All rows from seasons before the validation season.
        validation: Rows from the selected tuning season.
        test: Rows from configured later seasons.
        validation_season: Four-digit football-data season label.
        test_seasons: Ordered labels of held-out seasons.
    """

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
    """Split matches by season and verify strict chronological partition boundaries.

    Args:
        matches: Match dataframe requiring ``season`` and ``date`` columns.
        validation_season: Season reserved for tuning and calibration decisions.
        test_seasons: One or more later seasons for final reporting.

    Returns:
        Train, validation and test copies with non-overlapping chronological dates.

    Raises:
        ValueError: If a season is absent, labels cannot be ordered, or date ranges overlap.
    """
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
    # SPLIT GUARD: training only includes seasons preceding validation; test seasons must
    # follow validation and are independently checked using match dates below.
    train_seasons = set(all_seasons[:validation_pos])
    test_season_set = set(test_tuple)
    train = matches.loc[seasons.isin(train_seasons)].copy()
    validation = matches.loc[seasons.eq(validation_season)].copy()
    test = matches.loc[seasons.isin(test_season_set)].copy()
    _assert_chronological(train, validation, test)
    return TemporalSplit(train, validation, test, validation_season, test_tuple)


def _season_start(season: str) -> int:
    """Convert the first two season digits into a sortable start year.

    Args:
        season: Four-digit football-data season, such as 1415 or 2021.

    Returns:
        Four-digit calendar year at the start of the season.
    """
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"Cannot order season label {season!r}; expected four digits")
    first = int(season[:2])
    return 2000 + first if first <= 79 else 1900 + first


def _assert_chronological(*frames: pd.DataFrame) -> None:
    """Raise if adjacent non-empty partitions touch or overlap in calendar dates.

    Args:
        frames: Train, validation and test dataframes with date columns in chronological order.

    Returns:
        None. This is the final date-level split safeguard.
    """
    # SPLIT GUARD: season labels alone are insufficient if a malformed season file has bad dates.
    dates = [pd.to_datetime(frame["date"]) for frame in frames if not frame.empty]
    for earlier, later in zip(dates, dates[1:], strict=False):
        if earlier.max() >= later.min():
            raise ValueError("Temporal partitions overlap or are not strictly chronological")
