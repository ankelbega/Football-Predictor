"""Exercise ingest validation counts, warnings and summary reporting."""


import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from footballpred.ingest.validate import (
    validate_matches,
    write_validation_report,
    summarize_matches,
    write_summary,
)


def sample_valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "league": ["E0", "E0"],
            "season": ["2324", "2324"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_goals": [2, 1],
            "away_goals": [0, 1],
            "result": ["H", "D"],
            "b365h": [2.1, 2.3],
            "b365d": [3.2, 3.3],
            "b365a": [3.1, 2.9],
        }
    )


def test_validate_matches_counts_problems_and_flags_suspicious_odds() -> None:
    """Risk: data-quality regression must be visible instead of silently entering model inputs."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "league": ["E0"] * 3,
            "season": ["2324"] * 3,
            "home_team": ["Arsenal", "Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Chelsea", "Arsenal"],
            "home_goals": [2, -1, 1],
            "away_goals": [0, 0, 1],
            "result": ["H", "X", "D"],
            "b365h": [2.1, 0.9, np.nan],
            "b365d": [3.2, 3.0, np.nan],
            "b365a": [3.1, 2.8, np.nan],
        }
    )
    report = validate_matches(frame)
    assert report["rows"] == 3
    assert report["duplicate_matches"] == 2
    assert report["impossible_scores"] == 1
    assert report["invalid_result_codes"] == 1
    assert report["rows_with_suspicious_odds"] == 1
    assert "duplicate match keys found" in report["warnings"]


def test_validate_matches_requires_columns() -> None:
    """Risk: validation should fail fast when a processed file has lost required fields."""
    with pytest.raises(ValueError, match="missing required columns"):
        validate_matches(pd.DataFrame({"date": [1]}))


def test_validate_empty_frame_returns_none_dates() -> None:
    """Risk: empty processed datasets should not crash reporting and should report None bounds."""
    report = validate_matches(sample_valid_frame().iloc[0:0])
    assert report["rows"] == 0
    assert report["date_min"] is None
    assert report["date_max"] is None
    assert report["warnings"] == []


def test_write_validation_report_round_trips_json(tmp_path: Path) -> None:
    """Risk: validation report generation must persist the same mapping it returns."""
    report = write_validation_report(sample_valid_frame(), tmp_path / "validation.json")
    assert Path(tmp_path / "validation.json").exists()
    assert json.loads((tmp_path / "validation.json").read_text()) == report


def test_summarize_matches_reports_rates_and_goals(tmp_path: Path) -> None:
    """Risk: summary statistics must reflect input outcomes and goal totals."""
    summary = summarize_matches(sample_valid_frame())
    assert summary["matches"] == 2
    assert summary["draw_rate"] == 0.5
    assert summary["home_win_rate"] == 0.5
    assert summary["mean_total_goals"] == 2.0
    assert summary["teams"] == 2
    persisted = write_summary(sample_valid_frame(), tmp_path / "summary.json")
    assert persisted == summary
    assert (tmp_path / "summary.json").exists()


def test_summarize_empty_frame_does_not_crash() -> None:
    """Risk: empty processed datasets should still summarize without division errors."""
    frame = sample_valid_frame().iloc[0:0]
    summary = summarize_matches(frame)
    assert summary["matches"] == 0
    assert summary["home_win_rate"] is None
    assert summary["total_goals_quantiles"] == {}
