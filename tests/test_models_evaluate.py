"""Check probability math, temporal safeguards, model outputs and explanations.

Small deterministic fixtures keep these model/evaluation tests fast while protecting core
analytics behavior used by the full season reports.
"""

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
from footballpred.explain.text import explain_match
from footballpred.features.builder import build_features
from footballpred.models.baselines import (
    BookmakerModel,
    ClassPriorModel,
    bookmaker_probabilities,
)
from footballpred.models.calibration import select_multiclass_calibrator
from footballpred.models.dixon_coles import (
    DixonColesConfig,
    DixonColesModel,
    dixon_coles_tau,
    markets_from_score_matrix,
    poisson_score_matrix,
)
from footballpred.models.elo import (
    EloConfig,
    EloModel,
    add_elo_pre_match_features,
    elo_expected_score,
    update_ratings,
)
from footballpred.models.ensemble import combine_probabilities, tune_ensemble_weight
from footballpred.models.lgbm import LightGBM1X2, LightGBMBinary, LightGBMConfig, train_targets


def small_matches() -> pd.DataFrame:
    """Return five hand-readable fixtures spanning training, validation and test seasons."""
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
    """Protect proper scoring formulas against class-order or cumulative-sum mistakes."""
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
    """Protect the partition boundary from accidental random or backward season splits."""
    split = season_split(small_matches(), "2324", ["2425"])
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
    with pytest.raises(ValueError, match="later"):
        season_split(small_matches(), "2324", ["2223"])


def test_elo_update_is_zero_sum_and_expected_score_is_symmetric() -> None:
    """Protect rating conservation and the neutral Elo expected-score midpoint."""
    config = EloConfig(k_factor=20, home_advantage=0)
    before = 1500 + 1475
    home_after, away_after = update_ratings(1500, 1475, 2, 0, config)
    assert home_after + away_after == pytest.approx(before)
    assert home_after > 1500
    assert elo_expected_score(0) == pytest.approx(0.5)


def test_elo_pre_match_features_same_day_does_not_update_between_fixtures() -> None:
    """Protect same-day fixtures from learning one another's result through Elo state."""
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
    """Protect prior and market vectors from invalid sums and bookmaker margin leakage."""
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
    """Protect the draw-aware Elo mapping's output range and row normalization."""
    matches = small_matches()
    model = EloModel(EloConfig()).fit(matches.iloc[:3])
    probabilities = model.predict_proba(matches.iloc[3:])
    assert probabilities.shape == (2, 3)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_dixon_coles_matrix_normalization_and_derived_markets_consistent() -> None:
    """Protect score-matrix probability mass and market sums from separate calculations."""
    matrix = poisson_score_matrix(1.6, 1.1, rho=-0.08, max_goals=10)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)
    assert (matrix >= 0).all()
    markets = markets_from_score_matrix(matrix)
    assert np.asarray(markets["1x2"]).sum() == pytest.approx(1.0, abs=1e-12)
    assert markets["over_2_5"] + markets["under_2_5"] == pytest.approx(1.0)
    assert markets["btts_yes"] + markets["btts_no"] == pytest.approx(1.0)
    assert markets["1x2"][0] == pytest.approx(np.tril(matrix, k=-1).sum())
    goals = np.add.outer(np.arange(11), np.arange(11))
    assert markets["over_2_5"] == pytest.approx(matrix[goals >= 3].sum())
    assert markets["btts_yes"] == pytest.approx(matrix[1:, 1:].sum())
    assert len(markets["scorelines"]) == 5


def test_dixon_coles_correction_only_changes_low_scores() -> None:
    """Protect rho from affecting score pairs outside Dixon–Coles's four low-score cells."""
    assert dixon_coles_tau(0, 0, 1.4, 1.2, -0.1) != 1.0
    assert dixon_coles_tau(2, 1, 1.4, 1.2, -0.1) == 1.0


def test_dixon_coles_fit_and_predict_markets() -> None:
    """Protect fitted score models from malformed market probabilities or cutoff metadata."""
    matches = small_matches()
    config = DixonColesConfig(xi=0.0, max_goals=8, maxiter=100)
    model = DixonColesModel(config).fit(matches.iloc[:3])
    probabilities = model.predict_proba(matches.iloc[3:])
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    markets = model.predict_markets(matches.iloc[3:])
    assert len(markets) == 2
    assert all(0 <= entry["over_2_5"] <= 1 for entry in markets)
    assert model.training_cutoff_ == pd.Timestamp("2022-08-03")


def test_feature_pipeline_is_invariant_to_future_results() -> None:
    """Protect all earlier features from changes to a later match's outcomes and shots."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "season": ["2324"] * 4,
            "home_team": ["A", "B", "C", "A"],
            "away_team": ["B", "A", "D", "C"],
            "home_goals": [1, 2, 0, 1],
            "away_goals": [0, 1, 1, 0],
            "HS": [10, 12, 8, 15],
            "AS": [7, 8, 9, 6],
            "HST": [4, 6, 2, 7],
            "AST": [3, 4, 4, 2],
        }
    )
    original_features = build_features(frame)
    changed_future = frame.copy()
    changed_future.loc[3, ["home_goals", "away_goals", "HS", "AS", "HST", "AST"]] = [
        9, 0, 30, 1, 15, 0
    ]
    updated_features = build_features(changed_future)
    pd.testing.assert_frame_equal(
        original_features.iloc[:3].reset_index(drop=True),
        updated_features.iloc[:3].reset_index(drop=True),
        check_dtype=False,
    )


def test_same_day_match_results_do_not_affect_each_others_features() -> None:
    """Protect each fixture on a shared date from observing another fixture's final score."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
            "season": ["2324"] * 4,
            "home_team": ["A", "B", "C", "A"],
            "away_team": ["B", "A", "D", "C"],
            "home_goals": [1, 2, 0, 1],
            "away_goals": [0, 1, 1, 0],
        }
    )
    changed = frame.copy()
    changed.loc[1, ["home_goals", "away_goals"]] = [0, 7]
    before, after = build_features(frame), build_features(changed)
    pd.testing.assert_series_equal(before.loc[2].drop("match_id"), after.loc[2].drop("match_id"))


def test_lightgbm_classifiers_and_validation_calibration_probabilities() -> None:
    """Protect multiclass and binary LightGBM outputs and calibration candidate contracts."""
    matches = pd.concat([small_matches()] * 24, ignore_index=True)
    matches["date"] = pd.date_range("2021-01-01", periods=len(matches), freq="D")
    features = build_features(matches)
    config = LightGBMConfig(n_estimators=15, num_leaves=7, min_child_samples=3)
    multiclass = LightGBM1X2(config).fit(features.iloc[:60], matches.iloc[:60])
    probabilities = multiclass.predict_proba(features.iloc[60:])
    assert probabilities.shape == (len(matches) - 60, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    binary_targets = train_targets(matches)
    binary = LightGBMBinary(config).fit(features.iloc[:60], binary_targets["over_2_5"][:60])
    binary_probabilities = binary.predict_proba(features.iloc[60:])
    assert ((binary_probabilities >= 0) & (binary_probabilities <= 1)).all()
    name, _, scores = select_multiclass_calibrator(
        probabilities, matches.iloc[60:]["result"].to_numpy()
    )
    assert name in {"raw", "temperature", "platt", "isotonic"}
    assert scores["raw"] >= 0


def test_ensemble_weight_tuning_returns_valid_probability_vectors() -> None:
    """Protect validation-weighted blends from invalid weights or non-normalized rows."""
    labels = np.array(["H", "D", "A"])
    dc = np.array([[0.6, 0.25, 0.15], [0.3, 0.4, 0.3], [0.1, 0.25, 0.65]])
    lgb = np.array([[0.55, 0.3, 0.15], [0.25, 0.5, 0.25], [0.15, 0.25, 0.6]])
    weight, scores = tune_ensemble_weight(labels, dc, lgb)
    combined = combine_probabilities(dc, lgb, weight)
    assert weight in scores
    assert np.allclose(combined.sum(axis=1), 1.0, atol=1e-6)


def test_shap_local_explanation_is_labeled_as_raw_score_not_pp() -> None:
    """Protect local explanations from falsely describing raw SHAP values as probability points."""
    matches = pd.concat([small_matches()] * 24, ignore_index=True)
    matches["date"] = pd.date_range("2021-01-01", periods=len(matches), freq="D")
    features = build_features(matches)
    model = LightGBM1X2(
        LightGBMConfig(n_estimators=10, num_leaves=7, min_child_samples=3)
    ).fit(features.iloc[:60], matches.iloc[:60])
    explanation = explain_match(model, features.iloc[[60]], "A", "B", top_n=3)
    assert len(explanation["top_features"]) == 3
    assert "not percentage points" in explanation["explanation_scale"]
    assert set(explanation["probabilities"]) == {"home_win", "draw", "away_win"}
