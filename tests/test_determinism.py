"""Check deterministic model output and the canonical cleaned-ingest golden file."""


import numpy as np
import pandas as pd
import pytest

from footballpred.ingest.clean import clean_source_file
from footballpred.pipeline import fit_deployable_models, save_bundle, load_bundle


def tiny_matches() -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    dates = pd.date_range("2021-01-01", periods=70, freq="7D")
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


SETTINGS = {
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
}


def test_fit_is_deterministic_across_runs(tmp_path) -> None:
    """Risk: repeated training on identical data must not drift due to random state or ordering."""
    data = tiny_matches()
    bundle_one = fit_deployable_models(data, SETTINGS)
    bundle_two = fit_deployable_models(data, SETTINGS)
    assert bundle_one["training_cutoff"] == bundle_two["training_cutoff"]
    fixture = data.iloc[[0]].copy()
    path_one = save_bundle(bundle_one, tmp_path / "one.joblib")
    path_two = save_bundle(bundle_two, tmp_path / "two.joblib")
    loaded_one = load_bundle(path_one)
    loaded_two = load_bundle(path_two)
    features = None
    from footballpred.features.builder import build_features

    features = build_features(data.iloc[:5])
    prediction_one = loaded_one["models"]["lightgbm"].predict_proba(features)
    prediction_two = loaded_two["models"]["lightgbm"].predict_proba(features)
    assert np.array_equal(prediction_one, prediction_two)


def test_cleaned_ingest_matches_golden_file() -> None:
    """Risk: any change to clean_source_file must be intentional and reviewed against this golden output."""
    from pathlib import Path
    import tempfile

    source = Path(tempfile.gettempdir()) / "golden_source_check.csv"
    source.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "19/08/23,Man Utd,Wolves,1,0,H,1.8,3.5,4.2\n"
        ",,,,,,,,\n",
        encoding="utf-8",
    )
    cleaned = clean_source_file(
        source, "E0", "2324", {"Man Utd": "Manchester United", "Wolves": "Wolverhampton Wanderers"}
    )
    golden_path = Path(__file__).resolve().parents[1] / "tests" / "golden" / "cleaned_expected.csv"
    golden = pd.read_csv(golden_path, parse_dates=["date"], dtype={"season": str})
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), golden, check_dtype=False)
