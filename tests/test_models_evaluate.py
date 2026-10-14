import numpy as np
import pandas as pd
import pytest

from footballpred.evaluate.metrics import (
    accuracy,
    expected_calibration_error,
    log_loss,
    multiclass_brier_score,
    ranked_probability_score,
)
from footballpred.evaluate.splits import season_split
from footballpred.models.baselines import (
    BookmakerModel,
    ClassPriorModel,
    bookmaker_probabilities,
)
from footballpred.models.elo import (
    EloConfig,
    EloModel,
    add_elo_pre_match_features,
    elo_expected_score,
    update_ratings,
)


def small_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-08-01", "2022-08-02", "2022-08-03", "2023-08-01", "2024-08-01"]
            ),
            "season": ["2223", "2223", "2223", "2324", "2425"],
            "home_team": ["A", "B", "C", "A", "B"],
            "away_team": ["B", "C", "A", "B", "C"],
            "home_goals": [2, 0, 1, 1, 0],
            "away_goals": [0, 0, 2, 1, 1],
            "result": ["H", "D", "A", "D", "A"],
        }
    )


def test_metric_scores_match_hand_computed_values() -> None:
    labels = np.array(["H", "D"])
    probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    target = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    expected_log_loss = (-np.log(0.7) - np.log(0.5)) / 2
    assert log_loss(labels, probabilities) == pytest.approx(expected_log_loss)
    assert multiclass_brier_score(target, probabilities) == pytest.approx((0.14 + 0.38) / 2)
    # Home: cumulative errors 0.3, 0.1; Draw: errors 0.2, 0.5.
    assert ranked_probability_score(target, probabilities) == pytest.approx((0.05 + 0.065) / 2)
    assert accuracy(target, probabilities) == pytest.approx(1.0)
    assert expected_calibration_error(target, probabilities, bins=2) >= 0


def test_split_is_chronological_and_test_not_before_training() -> None:
    split = season_split(small_matches(), "2324", ["2425"])
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
    with pytest.raises(ValueError, match="later"):
        season_split(small_matches(), "2324", ["2223"])


def test_elo_update_is_zero_sum_and_expected_score_is_symmetric() -> None:
    config = EloConfig(k_factor=20, home_advantage=0)
    before = 1500 + 1475
    home_after, away_after = update_ratings(1500, 1475, 2, 0, config)
    assert home_after + away_after == pytest.approx(before)
    assert home_after > 1500
    assert elo_expected_score(0) == pytest.approx(0.5)


def test_elo_pre_match_features_same_day_does_not_update_between_fixtures() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "D", "C"],
            "home_goals": [4, 0, 1],
            "away_goals": [0, 3, 1],
        }
    )
    features = add_elo_pre_match_features(matches, EloConfig())
    assert features.loc[0, "elo_home"] == 1500
    assert features.loc[1, "elo_home"] == 1500
    assert features.loc[2, "elo_home"] != 1500


def test_baselines_emit_valid_probabilities_and_odds_remove_overround() -> None:
    training = small_matches().iloc[:3]
    test = pd.DataFrame(
        {
            "b365h": [2.0],
            "b365d": [4.0],
            "b365a": [4.0],
            "psch": [2.2],
            "pscd": [3.8],
            "psca": [3.5],
        }
    )
    prior = ClassPriorModel().fit(training).predict_proba(test)
    bookmaker, source = bookmaker_probabilities(test)
    assert np.allclose(prior.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(bookmaker.sum(axis=1), 1.0, atol=1e-6)
    assert source == "pinnacle_closing"
    assert bookmaker[0, 0] == pytest.approx((1 / 2.2) / (1 / 2.2 + 1 / 3.8 + 1 / 3.5))
    assert np.allclose(BookmakerModel().predict_proba(test), bookmaker)


def test_elo_model_outputs_valid_probabilities() -> None:
    matches = small_matches()
    model = EloModel(EloConfig()).fit(matches.iloc[:3])
    probabilities = model.predict_proba(matches.iloc[3:])
    assert probabilities.shape == (2, 3)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
