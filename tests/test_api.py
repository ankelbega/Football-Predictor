from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from footballpred.api.app import create_app
from footballpred.pipeline import fit_deployable_models, save_bundle


def sample_data() -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    dates = pd.date_range("2021-08-01", periods=80, freq="7D")
    for index, match_date in enumerate(dates):
        home, away = teams[index % 4], teams[(index + 1) % 4]
        home_goals, away_goals = (2, 0) if index % 3 == 0 else (1, 1) if index % 3 == 1 else (0, 1)
        result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
        rows.append(
            {
                "date": match_date,
                "season": "2122",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
            }
        )
    return pd.DataFrame(rows)


def test_api_smoke_health_teams_predict_and_unknown_team(tmp_path: Path) -> None:
    data = sample_data()
    settings = {
        "lightgbm": {
            "validation_estimators": [10],
            "validation_num_leaves": [7],
            "learning_rate": 0.05,
            "min_child_samples": 2,
            "reg_lambda": 1.0,
            "random_state": 17,
        },
        "dixon_coles": {
            "validation_xi": [0.0, 0.0018],
            "max_goals": 10,
            "score_optimization_maxiter": 100,
        },
    }
    bundle = fit_deployable_models(data, settings)
    model_path = save_bundle(bundle, tmp_path / "model.joblib")
    client = TestClient(create_app(model_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    teams = client.get("/teams")
    assert teams.status_code == 200
    assert "Arsenal" in teams.json()
    prediction = client.get(
        "/predict",
        params={"home": "Arsenal", "away": "Chelsea", "date": "2025-04-01"},
    )
    assert prediction.status_code == 200
    result = prediction.json()
    probabilities = result["probabilities_1x2"]
    assert abs(sum(probabilities.values()) - 1) < 1e-6
    assert len(result["most_likely_scorelines"]) == 5
    assert len(result["explanatory_features"]) == 5
    unknown = client.get(
        "/predict",
        params={"home": "Unknown FC", "away": "Chelsea", "date": "2025-04-01"},
    )
    assert unknown.status_code == 404


def test_api_dixon_coles_variant(tmp_path: Path) -> None:
    data = sample_data()
    settings = {
        "lightgbm": {
            "validation_estimators": [10],
            "validation_num_leaves": [7],
            "learning_rate": 0.05,
            "min_child_samples": 2,
            "reg_lambda": 1.0,
            "random_state": 17,
        },
        "dixon_coles": {
            "validation_xi": [0.0, 0.0018],
            "max_goals": 10,
            "score_optimization_maxiter": 100,
        },
    }
    bundle = fit_deployable_models(data, settings)
    model_path = save_bundle(bundle, tmp_path / "dc-model.joblib")
    client = TestClient(create_app(model_path))
    response = client.get(
        "/predict",
        params={
            "home": "Arsenal",
            "away": "Chelsea",
            "date": "2025-04-01",
            "model": "dixon_coles",
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "dixon_coles"
