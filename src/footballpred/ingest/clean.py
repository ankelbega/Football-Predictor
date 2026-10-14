"""Read and normalize football-data.co.uk source rows into the project schema.

This ingest-stage module is called by the downloader before validation and feature building.
Source dates, encodings, optional odds headers, and club spellings vary between seasons.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")
ODDS_COLUMNS = ("B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA")


def normalize_team_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """Collapse whitespace and map a source club name to its configured canonical name.

    Args:
        name: Raw home or away team cell.
        aliases: Exact source-name to canonical-name mapping from YAML.

    Returns:
        Cleaned canonical team name. Unlisted names are retained after whitespace cleanup.
    """
    # NORMALIZATION: explicit aliases unify short forms without fuzzy matching unrelated clubs.
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    return (aliases or {}).get(cleaned, cleaned)


def parse_match_date(value: object) -> pd.Timestamp:
    """Parse day-first source dates in two- or four-digit year form.

    Args:
        value: Raw date cell from a football-data CSV.

    Returns:
        Timestamp or NaT when empty or unparseable. Day-first avoids US month/day ambiguity.
    """
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    text = str(value).strip()
    # NORMALIZATION: day-first avoids interpreting 04/08/24 as April 8 instead of 4 August.
    return pd.to_datetime(text, dayfirst=True, errors="coerce", format="mixed")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a source CSV while tolerating common encodings and malformed lines.

    Args:
        path: Cached raw CSV path.

    Returns:
        Parsed dataframe; trailing blank rows are removed by the cleaner.
    """
    errors: list[Exception] = []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError as exc:
            errors.append(exc)
    raise errors[-1]


def clean_source_file(
    path: str | Path,
    league: str,
    season: str,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Clean one league-season CSV into the canonical match dataframe.

    Args:
        path: Downloaded source CSV path.
        league: League identifier such as E0.
        season: Four-digit season identifier such as 2324.
        aliases: Optional canonical team-name map.

    Returns:
        Sorted dataframe with date, league, season, teams, goals, result and available odds.
        Required result fields are checked before any rows are returned.
    """
    frame = _read_csv(Path(path))
    frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

    frame = frame.dropna(how="all").copy()
    # NORMALIZATION: create stable project field names while leaving raw odds as numeric values.
    frame["date"] = frame["Date"].map(parse_match_date)
    frame["home_team"] = frame["HomeTeam"].map(lambda value: normalize_team_name(value, aliases))
    frame["away_team"] = frame["AwayTeam"].map(lambda value: normalize_team_name(value, aliases))
    for original, clean in (("FTHG", "home_goals"), ("FTAG", "away_goals")):
        frame[clean] = pd.to_numeric(frame[original], errors="coerce")
    frame["result"] = frame["FTR"].astype("string").str.strip().str.upper()
    frame["league"] = league
    frame["season"] = season

    # NORMALIZATION: keep numeric odds under stable lowercase names; source files omit some seasons.
    for column in (
        *ODDS_COLUMNS,
        "B365>2.5",
        "B365<2.5",
        "Avg>2.5",
        "Avg<2.5",
    ):
        if column in frame.columns:
            frame[column.lower().replace(">", "_over_").replace("<", "_under_")] = pd.to_numeric(
                frame[column], errors="coerce"
            )
    for column in ("B365btts_yes", "B365btts_no", "Avgbtts_yes", "Avgbtts_no"):
        if column in frame.columns:
            frame[column.lower()] = pd.to_numeric(frame[column], errors="coerce")
    keep = [
        "date", "league", "season", "home_team", "away_team", "home_goals", "away_goals", "result"
    ]
    keep.extend(
        column
        for column in frame.columns
        if column.startswith(("b365", "psc", "avg"))
    )
    # Unknown teams are retained rather than dropped; an alias can be added without data loss.
    cleaned = frame[keep].dropna(
        subset=["date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    cleaned["home_goals"] = cleaned["home_goals"].astype("int64")
    cleaned["away_goals"] = cleaned["away_goals"].astype("int64")
    return cleaned.sort_values(
        ["date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
