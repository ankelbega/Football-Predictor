"""Smoke-test evaluation report generation on a tiny deterministic dataset."""


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footballpred.evaluate.report import evaluate_phase2, evaluate_phase3, evaluate_phase4


def make_dataset() -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    seasons = [
        ("2122", "2021-08-01"),
        ("2223", "2022-08-01"),
        ("2324", "2023-08-01"),
    ]
    for season, start in seasons:
        for index in range(20):
            home, away = teams[index % 4], teams[(index + 1) % 4]
            home_goals, away_goals = (2, 0) if index % 3 == 0 else (1, 1) if index % 3 == 1 else (0, 1)
            result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
            match_date = (pd.Timestamp(start) + pd.to_timedelta(index, unit="D")).date().isoformat()
            rows.append(
                {
                    "date": match_date,
                    "season": season,
                    "league": "E0",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "b365h": 2.0,
                    "b365d": 3.5,
                    "b365a": 3.5,
                    "psch": 2.1,
                    "pscd": 3.4,
                    "psca": 3.4,
                    "b365_over_2.5": 1.9,
                    "b365_under_2.5": 1.9,
                }
            )
    return pd.DataFrame(rows)


SETTINGS = {
    "validation_season": "2223",
    "test_seasons": ["2324"],
    "elo": {
        "validation_k_factors": [20.0],
        "validation_home_advantages": [50.0],
        "initial_rating": 1500.0,
    },
    "lightgbm": {
        "validation_estimators": [5],
        "validation_num_leaves": [7],
        "learning_rate": 0.05,
        "min_child_samples": 2,
        "reg_lambda": 1.0,
        "random_state": 17,
    },
    "dixon_coles": {
        "validation_xi": [0.0, 0.0018],
        "max_goals": 5,
        "score_optimization_maxiter": 30,
    },
}


def test_evaluate_reports_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: report generation should complete and persist outputs without changing committed artifacts."""
    monkeypatch.chdir(tmp_path)
    matches = make_dataset()
    output_path = tmp_path / "results" / "model-comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result2 = evaluate_phase2(matches, "2223", ["2324"], SETTINGS["elo"], output_path)
    assert not result2.empty
    assert output_path.exists()

    result3 = evaluate_phase3(matches, "2223", ["2324"], SETTINGS, output_path)
    assert not result3.empty
    assert (tmp_path / "results" / "phase3-market-comparison.csv").exists()

    result4 = evaluate_phase4(matches, "2223", ["2324"], SETTINGS, output_path)
    assert not result4.empty
    assert (tmp_path / "results" / "phase4-model-comparison.csv").exists()
