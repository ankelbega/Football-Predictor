"""Guard ingestion behavior against source-date and team-label variations.

These tests exercise ingest helpers with small fixtures so parsing regressions are caught without
downloading real seasons or changing committed evaluation outputs.
"""

from pathlib import Path

import pandas as pd

from footballpred.ingest.clean import clean_source_file, normalize_team_name, parse_match_date


def test_parse_dates_day_first_two_and_four_digit_years() -> None:
    """Protect against ambiguous dates being parsed month-first or invalid dates crashing."""
    assert parse_match_date("16/08/14") == pd.Timestamp("2014-08-16")
    assert parse_match_date("16/08/2024") == pd.Timestamp("2024-08-16")
    assert pd.isna(parse_match_date(""))
    assert pd.isna(parse_match_date("not a date"))


def test_team_aliases_and_whitespace() -> None:
    """Protect canonical names from whitespace changes while preserving unknown names."""
    alias = {"Man Utd": "Manchester United"}
    assert normalize_team_name("  Man   Utd ", alias) == "Manchester United"
    assert normalize_team_name("Some FC") == "Some FC"


def test_clean_source_drops_blank_rows_and_normalizes(tmp_path: Path) -> None:
    """Protect the cleaned schema against blank trailing rows and alias variations."""
    source = tmp_path / "E0_2324.csv"
    source.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "19/08/23,Man Utd,Wolves,1,0,H,1.8,3.5,4.2\n"
        ",,,,,,,,\n",
        encoding="utf-8",
    )
    result = clean_source_file(
        source, "E0", "2324", {"Man Utd": "Manchester United", "Wolves": "Wolverhampton Wanderers"}
    )
    assert len(result) == 1
    assert result.loc[0, "date"] == pd.Timestamp("2023-08-19")
    assert result.loc[0, "home_team"] == "Manchester United"
    assert result.loc[0, "away_team"] == "Wolverhampton Wanderers"
    assert result.loc[0, "b365h"] == 1.8


def test_malformed_dates_and_missing_scores_are_removed(tmp_path: Path) -> None:
    """Protect the processed dataset from unusable dates and incomplete final scores."""
    source = tmp_path / "bad_rows.csv"
    source.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "bad-date,Arsenal,Chelsea,1,0,H\n"
        "01/09/24,Liverpool,Everton,,1,A\n"
        "02/09/24,Arsenal,Everton,2,0,H\n",
        encoding="utf-8",
    )
    cleaned = clean_source_file(source, "E0", "2425")
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["home_team"] == "Arsenal"
