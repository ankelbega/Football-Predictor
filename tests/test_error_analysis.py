"""Smoke-test the error analysis writer on tiny prediction fixtures."""


import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_error_analysis_writes_expected_files(tmp_path: Path) -> None:
    """Risk: the error-analysis generator must produce its files."""
    n = 20
    predictions = pd.DataFrame(
        {
            "season": ["2425"] * n,
            "date": [f"2025-01-{i + 1:02d}" for i in range(n)],
            "home_team": ["Arsenal"] * n,
            "away_team": ["Chelsea"] * n,
            "actual_result": ["H"] * 10 + ["D"] * 5 + ["A"] * 5,
            "home_win_prob": [0.6] * n,
            "draw_prob": [0.25] * n,
            "away_win_prob": [0.15] * n,
            "best_model_logloss": [0.5] * n,
            "bookmaker_logloss": [0.6] * n,
            "bookmaker_home": [0.55] * n,
            "bookmaker_draw": [0.25] * n,
            "bookmaker_away": [0.20] * n,
            "team_status": ["established"] * n,
            "season_stage": (
                ["early"] * (n // 3) + ["mid"] * (n // 3) + ["late"] * (n - 2 * (n // 3))
            ),
            "over25_actual": [1] * 10 + [0] * 10,
            "over25_prob": [0.7] * n,
            "over25_bookmaker_prob": [0.65] * n,
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    output_dir = tmp_path / "docs"
    figures_dir = output_dir / "figures"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "error_analysis.py"),
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--figures-dir",
            str(figures_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote report" in result.stdout
    assert (output_dir / "ERROR_ANALYSIS.md").exists()
    assert (figures_dir / "error_reliability_1x2.png").exists()
    for table_name in (
        "error_slice_season_stage.csv",
        "error_slice_team_status.csv",
        "error_slice_match_type.csv",
        "error_reliability_1x2.csv",
        "error_reliability_over25.csv",
        "error_top_logloss.csv",
    ):
        table = pd.read_csv(output_dir / table_name)
        assert len(table.columns) > 0
