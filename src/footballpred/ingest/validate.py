"""Validate required match rows and write compact dataset summaries.

This ingest-stage module runs after cleaning and before feature/model work. It reports source
problems rather than silently repairing them, keeping downstream results reviewable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_matches(frame: pd.DataFrame) -> dict[str, Any]:
    """Count missing fields, duplicate fixtures, impossible scores and suspicious odds.

    Args:
        frame: Processed matches with date, league, season, home/away teams, goals and result.

    Returns:
        JSON-compatible counts and warnings. Odds <= 1 are flagged because decimal prices
        must exceed one; missing optional odds do not count as suspicious.
    """
    required = [
        "date",
        "league",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Processed dataset missing required columns: {missing_columns}")
    # Duplicate match keys are counted on both copies so the report shows affected rows.
    duplicates = frame.duplicated(["date", "league", "home_team", "away_team"], keep=False)
    impossible = (frame["home_goals"] < 0) | (frame["away_goals"] < 0)
    invalid_result = ~frame["result"].isin(["H", "D", "A"])
    odds_cols = [
        column
        for column in ("b365h", "b365d", "b365a", "psch", "pscd", "psca")
        if column in frame
    ]
    if odds_cols:
        odds = frame[odds_cols].apply(pd.to_numeric, errors="coerce")
        odds_invalid = ((odds <= 1.0) | ~np.isfinite(odds)).any(axis=1) & odds.notna().any(axis=1)
    else:
        odds_invalid = pd.Series(False, index=frame.index)
    missing = {column: int(frame[column].isna().sum()) for column in required}
    return {
        "rows": int(len(frame)),
        "date_min": (
            str(pd.to_datetime(frame["date"], errors="coerce").min().date()) if len(frame) else None
        ),
        "date_max": (
            str(pd.to_datetime(frame["date"], errors="coerce").max().date()) if len(frame) else None
        ),
        "missing_values": missing,
        "duplicate_matches": int(duplicates.sum()),
        "impossible_scores": int(impossible.sum()),
        "invalid_result_codes": int(invalid_result.sum()),
        "rows_with_suspicious_odds": int(odds_invalid.sum()),
        "odds_columns_checked": odds_cols,
        "warnings": [
            message
            for count, message in (
                (int(duplicates.sum()), "duplicate match keys found"),
                (int(impossible.sum()), "negative goal totals found"),
                (int(invalid_result.sum()), "unexpected full-time result codes found"),
                (int(odds_invalid.sum()), "odds at or below 1.0 or non-finite found"),
            )
            if count
        ],
    }


def write_validation_report(frame: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Save validation counts to JSON and log the same report.

    Args:
        frame: Canonical processed match dataframe.
        path: Destination JSON path.

    Returns:
        The report mapping written to disk.
    """
    report = validate_matches(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Validation: %s", json.dumps(report, sort_keys=True))
    return report


def summarize_matches(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate match counts, outcome rates and goal distribution summaries.

    Args:
        frame: Processed matches with result and home/away goals.

    Returns:
        JSON-compatible descriptive statistics; no predictive metrics are calculated here.
    """
    goals = frame["home_goals"] + frame["away_goals"]
    return {
        "matches": int(len(frame)),
        "seasons": sorted(frame["season"].dropna().astype(str).unique().tolist()),
        "leagues": sorted(frame["league"].dropna().astype(str).unique().tolist()),
        "teams": int(pd.concat([frame["home_team"], frame["away_team"]]).nunique()),
        "home_win_rate": float((frame["result"] == "H").mean()) if len(frame) else None,
        "draw_rate": float((frame["result"] == "D").mean()) if len(frame) else None,
        "away_win_rate": float((frame["result"] == "A").mean()) if len(frame) else None,
        "mean_home_goals": float(frame["home_goals"].mean()) if len(frame) else None,
        "mean_away_goals": float(frame["away_goals"].mean()) if len(frame) else None,
        "mean_total_goals": float(goals.mean()) if len(frame) else None,
        "total_goals_quantiles": {
            str(q): float(goals.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)
        }
        if len(frame)
        else {},
    }


def write_summary(frame: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Write the descriptive summary to JSON and return it.

    Args:
        frame: Processed match dataframe.
        path: Destination JSON path.

    Returns:
        Summary mapping.
    """
    summary = summarize_matches(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Data summary: %s", json.dumps(summary, sort_keys=True))
    return summary
