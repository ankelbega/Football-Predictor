"""Exercise bundle fitting and walk-forward cutoff safeguards."""


import pandas as pd
import pytest

from footballpred.pipeline import fit_deployable_models, load_bundle, save_bundle


def tiny_training_frame() -> pd.DataFrame:
    """Repeat a small round-robin history across many dates for fast model fitting."""
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    dates = pd.date_range("2021-01-01", periods=80, freq="7D")
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


LIGHTGBM_SETTINGS = {
    "validation_estimators": [8],
    "validation_num_leaves": [7],
    "learning_rate": 0.05,
    "min_child_samples": 2,
    "reg_lambda": 1.0,
    "random_state": 17,
}
DC_SETTINGS = {
    "validation_xi": [0.0, 0.0018],
    "max_goals": 6,
    "score_optimization_maxiter": 50,
}


def test_fit_deployable_models_respects_training_cutoff() -> None:
    """Risk: walk-forward evaluation must not silently include matches after its cutoff."""
    data = tiny_training_frame()
    settings = {"lightgbm": LIGHTGBM_SETTINGS, "dixon_coles": DC_SETTINGS}
    cutoff = pd.Timestamp("2021-06-01")
    bundle = fit_deployable_models(data, settings, training_cutoff=cutoff)
    assert pd.Timestamp(bundle["training_cutoff"]) < cutoff
    assert (pd.to_datetime(bundle["history"]["date"]) < cutoff).all()
    expected_rows = int((pd.to_datetime(data["date"]) < cutoff).sum())
    assert len(bundle["history"]) == expected_rows
    with pytest.raises(ValueError, match="No training"):
        fit_deployable_models(data, settings, training_cutoff=pd.Timestamp("2021-01-01"))


def test_bundle_round_trips_with_required_keys(tmp_path) -> None:
    """Risk: serving and CLI must fail early if a saved artifact is missing required metadata."""
    data = tiny_training_frame()
    settings = {"lightgbm": LIGHTGBM_SETTINGS, "dixon_coles": DC_SETTINGS}
    bundle = fit_deployable_models(data, settings)
    path = save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_bundle(path)
    assert {"models", "teams", "training_cutoff", "history"} <= set(loaded)
    assert "Arsenal" in loaded["teams"]


def test_end_to_end_tiny_fixture_pipeline(tmp_path, monkeypatch) -> None:
    """Risk: download-clean-fit-save-load must stay wired together for a fresh repository."""
    import pandas as pd

    from footballpred.config import DataConfig
    from footballpred.ingest.download import ingest_dataset
    from footballpred.features.builder import build_features

    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    lines = ["Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"]
    for index in range(80):
        home, away = teams[index % 4], teams[(index + 1) % 4]
        home_goals, away_goals = (2, 0) if index % 3 == 0 else (1, 1) if index % 3 == 1 else (0, 1)
        result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
        day = (pd.Timestamp("2021-01-01") + pd.to_timedelta(int(index), unit="D")).strftime("%d/%m/%y")
        lines.append(f"{day},{home},{away},{home_goals},{away_goals},{result}")
    raw = tmp_path / "E0_2122.csv"
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = DataConfig(
        source_url="https://example.test/{league}_{season}.csv",
        leagues=["E0"],
        seasons=["2122"],
        cache_dir=tmp_path / "raw",
        processed_path=tmp_path / "processed.csv",
        validation_report_path=tmp_path / "validation.json",
        stats_path=tmp_path / "summary.json",
    )
    monkeypatch.setattr(
        "footballpred.ingest.download.download_source",
        lambda cfg, refresh=False: [raw],
    )
    dataset = ingest_dataset(config, refresh=False)
    assert len(dataset) == 80
    assert (tmp_path / "processed.csv").exists()

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    features = build_features(frame)
    assert features.shape[0] == len(frame)
    settings = {"lightgbm": LIGHTGBM_SETTINGS, "dixon_coles": DC_SETTINGS}
    bundle = fit_deployable_models(frame, settings)
    model_path = save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_bundle(model_path)
    assert loaded["teams"] == sorted(set(frame["home_team"]) | set(frame["away_team"]))
    assert len(features.columns) > 5