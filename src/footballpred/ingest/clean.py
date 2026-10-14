"""Parsing, normalization, and cleaning for football-data.co.uk CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")
ODDS_COLUMNS = ("B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA")


def normalize_team_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """Return a canonical team label using explicit aliases and whitespace cleanup."""
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    return (aliases or {}).get(cleaned, cleaned)


def parse_match_date(value: object) -> pd.Timestamp:
    """Parse common football-data date forms (dd/mm/yy and dd/mm/yyyy)."""
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    text = str(value).strip()
    # Explicit day-first avoids interpreting strings like 04/08/24 as month-first.
    return pd.to_datetime(text, dayfirst=True, errors="coerce", format="mixed")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a source CSV tolerantly; source files vary in encoding and blank trailing rows."""
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
    """Normalize one downloaded source file into a stable match table."""
    frame = _read_csv(Path(path))
    frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

    frame = frame.dropna(how="all").copy()
    frame["date"] = frame["Date"].map(parse_match_date)
    frame["home_team"] = frame["HomeTeam"].map(lambda value: normalize_team_name(value, aliases))
    frame["away_team"] = frame["AwayTeam"].map(lambda value: normalize_team_name(value, aliases))
    for original, clean in (("FTHG", "home_goals"), ("FTAG", "away_goals")):
        frame[clean] = pd.to_numeric(frame[original], errors="coerce")
    frame["result"] = frame["FTR"].astype("string").str.strip().str.upper()
    frame["league"] = league
    frame["season"] = season

    for column in (*ODDS_COLUMNS, "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5"):
        if column in frame.columns:
            frame[column.lower().replace(">", "_over_").replace("<", "_under_")] = pd.to_numeric(
                frame[column], errors="coerce"
            )
    keep = [
        "date", "league", "season", "home_team", "away_team", "home_goals", "away_goals", "result"
    ]
    keep.extend(column for column in frame.columns if column.startswith(("b365", "psc", "avg")))
    cleaned = frame[keep].dropna(
        subset=["date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    cleaned["home_goals"] = cleaned["home_goals"].astype("int64")
    cleaned["away_goals"] = cleaned["away_goals"].astype("int64")
    return cleaned.sort_values(
        ["date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
