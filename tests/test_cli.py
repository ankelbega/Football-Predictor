"""Exercise CLI commands through Typer's in-process runner."""


from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from footballpred.cli import app


def make_raw(path: Path, count: int = 80) -> None:
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    lines = ["Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"]
    for index in range(count):
        home, away = teams[index % 4], teams[(index + 1) % 4]
        home_goals, away_goals = (2, 0) if index % 3 == 0 else (1, 1) if index % 3 == 1 else (0, 1)
        result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
        day = (pd.Timestamp("2021-01-01") + pd.to_timedelta(index, unit="D")).strftime("%d/%m/%y")
        lines.append(f"{day},{home},{away},{home_goals},{away_goals},{result}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config(tmp_path: Path, raw_path: Path, processed_path: Path) -> Path:
    payload = {
        "data": {
            "source_url": "https://example.test/{league}_{season}.csv",
            "leagues": ["E0"],
            "seasons": ["2122"],
            "cache_dir": str(tmp_path / "raw"),
            "processed_path": str(processed_path),
            "validation_report_path": str(tmp_path / "validation.json"),
            "stats_path": str(tmp_path / "summary.json"),
            "results_path": str(tmp_path / "model-comparison.csv"),
            "model_path": str(tmp_path / "model.joblib"),
        },
        "evaluation": {
            "validation_season": "2122",
            "test_seasons": [],
            "elo": {
                "validation_k_factors": [20.0],
                "validation_home_advantages": [50.0],
                "initial_rating": 1500.0,
            },
            "lightgbm": {
                "validation_estimators": [8],
                "validation_num_leaves": [7],
                "learning_rate": 0.05,
                "min_child_samples": 2,
                "reg_lambda": 1.0,
                "random_state": 17,
            },
            "dixon_coles": {
                "validation_xi": [0.0, 0.0018],
                "max_goals": 6,
                "score_optimization_maxiter": 50,
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_download_with_mocked_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: CLI download must turn a successful ingest into a user-facing saved-row summary."""
    raw_path = tmp_path / "E0_2122.csv"
    make_raw(raw_path, 20)
    monkeypatch.setattr("footballpred.ingest.download.download_source", lambda cfg, refresh=False: [raw_path])
    processed_path = tmp_path / "processed.csv"
    config_path = write_config(tmp_path, raw_path, processed_path)
    runner = CliRunner()
    result = runner.invoke(app, ["download", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output
    assert processed_path.exists()


def test_cli_validate_and_build_features(tmp_path: Path) -> None:
    """Risk: CLI read-transform commands must consume the processed dataset and write feature output."""
    raw_path = tmp_path / "raw.csv"
    make_raw(raw_path, 20)
    from footballpred.ingest.clean import clean_source_file

    cleaned = clean_source_file(raw_path, "E0", "2122")
    processed_path = tmp_path / "processed.csv"
    cleaned.to_csv(processed_path, index=False, date_format="%Y-%m-%d")
    config_path = write_config(tmp_path, raw_path, processed_path)
    runner = CliRunner()
    validate_result = runner.invoke(app, ["validate", "--config", str(config_path)])
    assert validate_result.exit_code == 0, validate_result.output
    features_path = tmp_path / "features.csv"
    features_result = runner.invoke(
        app,
        ["build-features", "--config", str(config_path), "--output", str(features_path)],
    )
    assert features_result.exit_code == 0, features_result.output
    assert features_path.exists()


def test_cli_train_teams_and_predict(tmp_path: Path) -> None:
    """Risk: CLI training artifact creation and later prediction must remain end-to-end usable."""
    raw_path = tmp_path / "raw.csv"
    make_raw(raw_path, 80)
    from footballpred.ingest.clean import clean_source_file

    cleaned = clean_source_file(raw_path, "E0", "2122")
    processed_path = tmp_path / "processed.csv"
    cleaned.to_csv(processed_path, index=False, date_format="%Y-%m-%d")
    config_path = write_config(tmp_path, raw_path, processed_path)
    runner = CliRunner()
    train_result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert train_result.exit_code == 0, train_result.output
    teams_result = runner.invoke(app, ["teams", "--config", str(config_path)])
    assert teams_result.exit_code == 0, teams_result.output
    assert "Arsenal" in teams_result.output
    predict_result = runner.invoke(
        app,
        ["predict", "--home", "Arsenal", "--away", "Chelsea", "--date", "2022-01-01", "--config", str(config_path)],
    )
    assert predict_result.exit_code == 0, predict_result.output
    assert "probabilities_1x2" in predict_result.output
