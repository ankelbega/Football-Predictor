"""Chronological evaluation with reproducible season-wise market results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from footballpred.evaluate.metrics import (
    accuracy,
    binary_brier_score,
    binary_log_loss,
    expected_calibration_error,
    log_loss,
    multiclass_brier_score,
    ranked_probability_score,
)
from footballpred.evaluate.splits import season_split
from footballpred.explain.shap_utils import compute_global_importance, save_global_importance
from footballpred.explain.text import explain_match
from footballpred.features.builder import build_features
from footballpred.models.baselines import (
    BookmakerModel,
    ClassPriorModel,
    bookmaker_probabilities,
    outcome_probabilities,
)
from footballpred.models.calibration import (
    apply_multiclass_calibrator,
    select_binary_isotonic,
    select_multiclass_calibrator,
)
from footballpred.models.dixon_coles import (
    DixonColesConfig,
    DixonColesModel,
)
from footballpred.models.elo import EloConfig, EloModel
from footballpred.models.ensemble import combine_probabilities, tune_ensemble_weight
from footballpred.models.lgbm import LightGBM1X2, LightGBMBinary, LightGBMConfig, train_targets

LOGGER = logging.getLogger(__name__)


def _metric_row(
    model_name: str,
    partition: str,
    season: str,
    observed: pd.DataFrame,
    predicted: np.ndarray,
    source: str | None = None,
    market: str = "1x2",
    bootstrap_seed: int = 2026,
    bookmaker_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    truth = outcome_probabilities(observed)
    bookmaker_difference = np.nan
    confidence_low = np.nan
    confidence_high = np.nan
    if model_name != "bookmaker" and bookmaker_reference is not None:
        try:
            model_losses = _per_row_log_loss(observed["result"].to_numpy(), predicted)
            market_losses = _per_row_log_loss(
                observed["result"].to_numpy(), bookmaker_reference
            )
            differences = model_losses - market_losses
            bookmaker_difference = float(differences.mean())
            if len(differences) > 1:
                interval = bootstrap(
                    (differences,),
                    np.mean,
                    confidence_level=0.95,
                    n_resamples=1000,
                    method="percentile",
                    rng=np.random.default_rng(bootstrap_seed),
                ).confidence_interval
                confidence_low = float(interval.low)
                confidence_high = float(interval.high)
        except (ValueError, TypeError):
            pass
    return {
        "model": model_name,
        "partition": partition,
        "season": season,
        "market": market,
        "matches": len(observed),
        "odds_source": source,
        "log_loss": log_loss(observed["result"].to_numpy(), predicted),
        "brier_score": multiclass_brier_score(truth, predicted),
        "rps": ranked_probability_score(truth, predicted),
        "accuracy": accuracy(truth, predicted),
        "ece_top_label": expected_calibration_error(truth, predicted),
        "log_loss_difference_vs_bookmaker": bookmaker_difference,
        "bootstrap_95_low": confidence_low,
        "bootstrap_95_high": confidence_high,
    }


def _per_row_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Per-match multiclass log losses in H/D/A class order."""
    target = outcome_probabilities(pd.DataFrame({"result": labels}))
    return -np.sum(target * np.log(np.clip(probabilities, 1e-15, 1.0)), axis=1)


def tune_elo(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    k_factors: list[float],
    home_advantages: list[float],
    initial_rating: float = 1500.0,
) -> tuple[EloConfig, pd.DataFrame]:
    """Choose Elo update parameters by validation log loss only."""
    results: list[dict[str, float]] = []
    best_config: EloConfig | None = None
    best_loss = float("inf")
    for k_factor in k_factors:
        for home_advantage in home_advantages:
            config = EloConfig(initial_rating, home_advantage, k_factor, True)
            model = EloModel(config).fit(training)
            probabilities = model.predict_proba(validation)
            score = log_loss(validation["result"].to_numpy(), probabilities)
            results.append(
                {
                    "k_factor": k_factor,
                    "home_advantage": home_advantage,
                    "validation_log_loss": score,
                }
            )
            if score < best_loss:
                best_loss = score
                best_config = config
    if best_config is None:
        raise ValueError("No Elo hyperparameter candidates supplied")
    sorted_results = pd.DataFrame(results).sort_values("validation_log_loss").reset_index(drop=True)
    return best_config, sorted_results


def evaluate_phase2(
    matches: pd.DataFrame,
    validation_season: str,
    test_seasons: list[str],
    elo_settings: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """Tune Elo on validation; report validation and per-season test metrics."""
    split = season_split(matches, validation_season, test_seasons)
    k_values = [float(value) for value in elo_settings["validation_k_factors"]]
    advantages = [float(value) for value in elo_settings["validation_home_advantages"]]
    elo_config, tuning = tune_elo(
        split.train,
        split.validation,
        k_values,
        advantages,
        float(elo_settings["initial_rating"]),
    )
    # Final parameters are chosen on the validation season. Refit the outcome mapping on all
    # information available before each prediction partition; no test results inform the fit.
    rows: list[dict[str, Any]] = []
    prior = ClassPriorModel().fit(split.train)
    bookmaker = BookmakerModel()
    validation_models = [
        ("class_prior", prior.predict_proba(split.validation), None),
    ]
    market_probabilities, odds_source = bookmaker_probabilities(split.validation)
    validation_models.append(("bookmaker", market_probabilities, odds_source))
    for model_name, probabilities, odds_source in validation_models:
        rows.append(
            _metric_row(
                model_name,
                "validation",
                validation_season,
                split.validation,
                probabilities,
                odds_source,
                bookmaker_reference=(
                    market_probabilities if model_name != "bookmaker" else None
                ),
            )
        )
    validation_elo = EloModel(elo_config).fit(split.train)
    rows.append(
        _metric_row(
            "elo",
            "validation",
            validation_season,
            split.validation,
            validation_elo.predict_proba(split.validation),
            bookmaker_reference=market_probabilities,
        )
    )

    for season in split.test_seasons:
        test_season_frame = split.test.loc[split.test["season"].astype(str).eq(season)].copy()
        # Each test season gets a refit on all earlier observed seasons, including validation
        # and any preceding test year, mirroring expanding-window walk-forward use.
        first_test_date = pd.to_datetime(test_season_frame["date"]).min()
        earlier = matches.loc[pd.to_datetime(matches["date"]) < first_test_date]
        prior_model = ClassPriorModel().fit(earlier)
        for model_name, model in (("class_prior", prior_model), ("bookmaker", bookmaker)):
            try:
                odds_source = None
                if model_name == "bookmaker":
                    predicted, odds_source = bookmaker_probabilities(test_season_frame)
                else:
                    predicted = model.predict_proba(test_season_frame)
            except ValueError as exc:
                LOGGER.warning("Skipping %s for season %s: %s", model_name, season, exc)
                continue
            rows.append(
                _metric_row(
                    model_name,
                    "test",
                    season,
                    test_season_frame,
                    predicted,
                    odds_source,
                    bookmaker_reference=(
                        bookmaker_probabilities(test_season_frame)[0]
                        if model_name != "bookmaker"
                        else None
                    ),
                )
            )
        elo_model = EloModel(elo_config).fit(earlier)
        elo_probabilities = elo_model.predict_proba(test_season_frame)
        rows.append(
            _metric_row(
                "elo",
                "test",
                season,
                test_season_frame,
                elo_probabilities,
                bookmaker_reference=bookmaker_probabilities(test_season_frame)[0],
            )
        )

    results = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False, float_format="%.8f")
    tuning_path = destination.with_name("elo-validation-tuning.csv")
    tuning.to_csv(tuning_path, index=False, float_format="%.8f")
    markdown_path = destination.with_suffix(".md")
    markdown_table = _to_markdown(results)
    markdown_path.write_text(
        "# Phase 2 model comparison\n\n"
        "Generated by `footballpred evaluate`. Test rows are season-wise expanding-window "
        "folds. Log-loss difference is model minus bookmaker; paired 95% bootstrap intervals "
        "resample match-level differences. Positive differences favor the bookmaker.\n\n"
        + markdown_table
        + "\n\nElo tuning candidates were ranked on the validation season only; see "
        f"`{tuning_path.name}`.\n",
        encoding="utf-8",
    )
    test_fold_results = results.loc[results["partition"].eq("test")].copy()
    test_fold_results.to_csv(
        destination.with_name("walk-forward-test-results.csv"),
        index=False,
        float_format="%.8f",
    )
    save_walk_forward_figure(results, "docs/walk-forward-log-loss.png")
    return results


def _binary_metrics(
    model_name: str,
    partition: str,
    season: str,
    truth: np.ndarray,
    probabilities: np.ndarray,
    market: str,
    odds_source: str | None = None,
    bookmaker_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    difference = np.nan
    interval_low = np.nan
    interval_high = np.nan
    if model_name != "bookmaker" and bookmaker_reference is not None:
        raw = np.asarray(probabilities, dtype=float)
        market_probabilities = np.asarray(bookmaker_reference, dtype=float)
        differences = (
            -(truth * np.log(np.clip(raw, 1e-15, 1.0))
              + (1 - truth) * np.log(np.clip(1 - raw, 1e-15, 1.0)))
            + truth * np.log(np.clip(market_probabilities, 1e-15, 1.0))
            + (1 - truth) * np.log(np.clip(1 - market_probabilities, 1e-15, 1.0))
        )
        difference = float(differences.mean())
        if len(differences) > 1:
            ci = bootstrap(
                (differences,),
                np.mean,
                confidence_level=0.95,
                n_resamples=1000,
                method="percentile",
                rng=np.random.default_rng(2026),
            ).confidence_interval
            interval_low, interval_high = float(ci.low), float(ci.high)
    return {
        "model": model_name,
        "partition": partition,
        "season": season,
        "market": market,
        "matches": len(truth),
        "odds_source": odds_source,
        "log_loss": binary_log_loss(truth, probabilities),
        "brier_score": binary_brier_score(truth, probabilities),
        "rps": np.nan,
        "accuracy": float(np.mean((probabilities >= 0.5) == truth.astype(bool))),
        "ece_top_label": np.nan,
        "log_loss_difference_vs_bookmaker": difference,
        "bootstrap_95_low": interval_low,
        "bootstrap_95_high": interval_high,
    }


def evaluate_phase3(
    matches: pd.DataFrame,
    validation_season: str,
    test_seasons: list[str],
    settings: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """Tune Dixon-Coles decay on validation, then evaluate all requested markets walk-forward."""
    split = season_split(matches, validation_season, test_seasons)
    config_settings = settings["dixon_coles"]
    xi_candidates = [float(value) for value in config_settings["validation_xi"]]
    validation_scores: list[dict[str, float]] = []
    best_xi: float | None = None
    best_loss = float("inf")
    for xi in xi_candidates:
        model = DixonColesModel(
            DixonColesConfig(
                xi=xi,
                max_goals=int(config_settings["max_goals"]),
                maxiter=int(config_settings["score_optimization_maxiter"]),
            )
        ).fit(split.train)
        predicted = model.predict_proba(split.validation)
        score = log_loss(split.validation["result"].to_numpy(), predicted)
        validation_scores.append({"xi": xi, "validation_1x2_log_loss": score})
        if score < best_loss:
            best_xi, best_loss = xi, score
    if best_xi is None:
        raise ValueError("No Dixon-Coles time-decay candidates supplied")

    fitted_settings = DixonColesConfig(
        xi=best_xi,
        max_goals=int(config_settings["max_goals"]),
        maxiter=int(config_settings["score_optimization_maxiter"]),
    )
    rows: list[dict[str, Any]] = []
    validation_models: list[tuple[str, pd.DataFrame, np.ndarray, str | None]] = []
    prior = ClassPriorModel().fit(split.train)
    validation_models.append(
        ("class_prior", split.validation, prior.predict_proba(split.validation), None)
    )
    validation_dc = DixonColesModel(fitted_settings).fit(split.train)
    validation_models.append(
        ("dixon_coles", split.validation, validation_dc.predict_proba(split.validation), None)
    )

    for partition, season, observed, training in [
        ("validation", validation_season, split.validation, split.train),
        *[
            (
                "test",
                season,
                split.test.loc[split.test["season"].astype(str).eq(season)].copy(),
                matches.loc[
                    pd.to_datetime(matches["date"])
                    < pd.to_datetime(
                        split.test.loc[split.test["season"].astype(str).eq(season), "date"].min()
                    )
                ],
            )
            for season in split.test_seasons
        ],
    ]:
        if partition == "validation":
            dc = validation_dc
            class_prior = prior
        else:
            dc = DixonColesModel(fitted_settings).fit(training)
            class_prior = ClassPriorModel().fit(training)

        try:
            market_probs, source = bookmaker_probabilities(observed)
        except ValueError as exc:
            LOGGER.warning("No complete 1X2 odds for %s %s: %s", partition, season, exc)
            market_probs, source = None, None

        dc_markets = dc.predict_markets(observed)
        dc_1x2 = np.stack([entry["1x2"] for entry in dc_markets])
        prior_1x2 = class_prior.predict_proba(observed)
        dc_over = np.array([entry["over_2_5"] for entry in dc_markets])
        dc_btts = np.array([entry["btts_yes"] for entry in dc_markets])
        prior_over = float(((training["home_goals"] + training["away_goals"]) >= 3).mean())
        prior_btts = float(((training["home_goals"] > 0) & (training["away_goals"] > 0)).mean())
        if market_probs is not None:
            rows.append(
                _metric_row(
                    "bookmaker",
                    partition,
                    season,
                    observed,
                    market_probs,
                    source,
                    "1x2",
                )
            )
        bookmaker_reference = market_probs
        for model_name, probabilities in (("class_prior", prior_1x2), ("dixon_coles", dc_1x2)):
            rows.append(
                _metric_row(
                    model_name,
                    partition,
                    season,
                    observed,
                    probabilities,
                    market="1x2",
                    bookmaker_reference=bookmaker_reference,
                )
            )

        market_truths = {
            "over_2_5": (observed["home_goals"] + observed["away_goals"] >= 3).to_numpy(dtype=int),
            "btts": (
                ((observed["home_goals"] > 0) & (observed["away_goals"] > 0))
                .to_numpy(dtype=int)
            ),
        }
        odds_events = _bookmaker_binary_probabilities(observed)
        binary_predictions = {
            "class_prior": {
                "over_2_5": np.full(len(observed), prior_over),
                "btts": np.full(len(observed), prior_btts),
            },
            "dixon_coles": {"over_2_5": dc_over, "btts": dc_btts},
        }
        for market, truth in market_truths.items():
            for model_name in ("class_prior", "dixon_coles"):
                rows.append(
                    _binary_metrics(
                        model_name,
                        partition,
                        season,
                        truth,
                        binary_predictions[model_name][market],
                        market,
                    )
                )
            if odds_events is not None and market in odds_events:
                odds_probabilities, odds_label = odds_events[market]
                rows.append(
                    _binary_metrics(
                        "bookmaker",
                        partition,
                        season,
                        truth,
                        odds_probabilities,
                        market,
                        odds_label,
                    )
                )

    results = pd.DataFrame(rows)
    destination = Path(output_path).with_name("phase3-market-comparison.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False, float_format="%.8f")
    tuning = pd.DataFrame(validation_scores).sort_values("validation_1x2_log_loss")
    tuning_path = destination.with_name("dixon-coles-validation-tuning.csv")
    tuning.to_csv(tuning_path, index=False, float_format="%.8f")
    destination.with_suffix(".md").write_text(
        "# Phase 3 market comparison\n\n"
        "Generated by `footballpred evaluate`. Dixon-Coles time decay is selected on validation "
        "1X2 log loss only. Tests are season-wise expanding-window evaluations. Bookmaker odds "
        "are proportionally de-vigged. football-data.co.uk does not provide the independent "
        "BTTS odds required here; BTTS market rows are omitted rather than inferred from O/U "
        "odds.\n\n"
        + _to_markdown(results)
        + f"\n\nValidation xi search: `{tuning_path.name}`.\n",
        encoding="utf-8",
    )
    return results


def evaluate_phase4(
    matches: pd.DataFrame,
    validation_season: str,
    test_seasons: list[str],
    settings: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """Tune LightGBM on the validation season; fit calibrators on validation predictions only."""
    split = season_split(matches, validation_season, test_seasons)
    feature_frame = build_features(matches)
    # For each model partition, prior history is assembled causally from the earliest match
    # through the rows being evaluated. Shared ex-ante features are independent of test labels.
    all_features = feature_frame
    validation_indices = split.validation.index.to_numpy()
    train_indices = split.train.index.to_numpy()
    validation_features = all_features.loc[validation_indices].reset_index(drop=True)
    train_features = all_features.loc[train_indices].reset_index(drop=True)
    tuning_options = settings["lightgbm"]
    candidates: list[dict[str, float]] = []
    best_config: LightGBMConfig | None = None
    best_loss = float("inf")
    best_model: LightGBM1X2 | None = None
    for estimators in tuning_options["validation_estimators"]:
        for leaves in tuning_options["validation_num_leaves"]:
            candidate_config = LightGBMConfig(
                n_estimators=int(estimators),
                learning_rate=float(tuning_options["learning_rate"]),
                num_leaves=int(leaves),
                min_child_samples=int(tuning_options["min_child_samples"]),
                reg_lambda=float(tuning_options["reg_lambda"]),
                random_state=int(tuning_options["random_state"]),
            )
            model = LightGBM1X2(candidate_config).fit(train_features, split.train)
            probabilities = model.predict_proba(validation_features)
            score = log_loss(split.validation["result"].to_numpy(), probabilities)
            candidates.append(
                {
                    "n_estimators": int(estimators),
                    "num_leaves": int(leaves),
                    "validation_log_loss": score,
                }
            )
            if score < best_loss:
                best_config, best_loss, best_model = candidate_config, score, model
    if best_config is None or best_model is None:
        raise ValueError("No LightGBM tuning candidates supplied")

    # Fit calibrators on predictions made over the validation season only; all candidate
    # models are trained strictly on seasons before validation, so these are out-of-sample.
    calibration_raw = best_model.predict_proba(validation_features)
    calibration_name, multiclass_calibrator, calibration_scores = select_multiclass_calibrator(
        calibration_raw,
        split.validation["result"].to_numpy(),
    )
    validation_calibrated = apply_multiclass_calibrator(
        calibration_name, multiclass_calibrator, calibration_raw
    )
    validation_truth = outcome_probabilities(split.validation)
    rows: list[dict[str, Any]] = []
    validation_market_probabilities = bookmaker_probabilities(split.validation)[0]
    for model_name, probabilities in (
        ("lightgbm", calibration_raw),
        (f"lightgbm_calibrated_{calibration_name}", validation_calibrated),
    ):
        rows.append(
            _metric_row(
                model_name,
                "validation",
                validation_season,
                split.validation,
                probabilities,
                bookmaker_reference=validation_market_probabilities,
            )
        )
    validation_elo_model = EloModel(EloConfig()).fit(split.train)
    validation_dc = DixonColesModel(
        DixonColesConfig(
            xi=float(settings["dixon_coles"]["validation_xi"][1]),
            max_goals=int(settings["dixon_coles"]["max_goals"]),
            maxiter=int(settings["dixon_coles"]["score_optimization_maxiter"]),
        )
    ).fit(split.train)
    validation_dc_probability = validation_dc.predict_proba(split.validation)
    validation_dc_markets = validation_dc.predict_markets(split.validation)
    for model_name, probabilities in (
        ("class_prior", ClassPriorModel().fit(split.train).predict_proba(split.validation)),
        ("bookmaker", validation_market_probabilities),
        ("elo", validation_elo_model.predict_proba(split.validation)),
        ("dixon_coles", validation_dc_probability),
    ):
        rows.append(
            _metric_row(
                model_name,
                "validation",
                validation_season,
                split.validation,
                probabilities,
                bookmaker_reference=(
                    validation_market_probabilities if model_name != "bookmaker" else None
                ),
            )
        )
    for market, target in train_targets(split.validation).items():
        validation_binary_model = LightGBMBinary(best_config).fit(
            train_features, train_targets(split.train)[market]
        )
        validation_binary_probabilities = validation_binary_model.predict_proba(
            validation_features
        )
        binary_bookmaker_reference = (
            _market_bookmaker_probabilities(split.validation, market)
            if market == "over_2_5"
            else None
        )
        rows.append(
            _binary_metrics(
                "class_prior",
                "validation",
                validation_season,
                target,
                np.full(len(target), float(train_targets(split.train)[market].mean())),
                market,
                bookmaker_reference=binary_bookmaker_reference,
            )
        )
        if market == "over_2_5":
            rows.append(
                _binary_metrics(
                    "bookmaker",
                    "validation",
                    validation_season,
                    target,
                    _market_bookmaker_probabilities(split.validation, market),
                    market,
                    "bet365_over_under",
                    bookmaker_reference=binary_bookmaker_reference,
                )
            )
        rows.append(
            _binary_metrics(
                "lightgbm",
                "validation",
                validation_season,
                target,
                validation_binary_probabilities,
                market,
                bookmaker_reference=binary_bookmaker_reference,
            )
        )
        dc_binary_probabilities = np.asarray(
            [
                item["over_2_5"] if market == "over_2_5" else item["btts_yes"]
                for item in validation_dc_markets
            ]
        )
        rows.append(
            _binary_metrics(
                "dixon_coles",
                "validation",
                validation_season,
                target,
                dc_binary_probabilities,
                market,
                bookmaker_reference=binary_bookmaker_reference,
            )
        )

    validation_dc_probabilities = validation_dc_probability
    ensemble_weight, ensemble_scores = tune_ensemble_weight(
        split.validation["result"].to_numpy(),
        validation_dc_probabilities,
        calibration_raw,
    )
    ensemble_validation = combine_probabilities(
        validation_dc_probabilities, calibration_raw, ensemble_weight
    )
    rows.append(
        _metric_row(
            "ensemble",
            "validation",
            validation_season,
            split.validation,
            ensemble_validation,
            bookmaker_reference=validation_market_probabilities,
        )
    )
    pd.DataFrame(
        [
            {"lightgbm_weight": weight, "validation_log_loss": score}
            for weight, score in ensemble_scores.items()
        ]
    ).to_csv("results/ensemble-validation-weights.csv", index=False, float_format="%.8f")

    target_validation = train_targets(split.validation)
    binary_calibrators: dict[str, tuple[str, object | None, dict[str, float]]] = {}
    binary_calibration_rows: list[dict[str, Any]] = []
    binary_config = best_config
    for market, target in target_validation.items():
        binary = LightGBMBinary(binary_config).fit(
            train_features, train_targets(split.train)[market]
        )
        predicted = binary.predict_proba(validation_features)
        binary_name, binary_calibrator, binary_scores = select_binary_isotonic(
            predicted,
            target,
        )
        calibration_rows = [
            {"calibrator": "raw", "market": market, "validation_log_loss": binary_scores["raw"]},
            {
                "calibrator": "isotonic_diagnostic",
                "market": market,
                "validation_log_loss": binary_scores["isotonic"],
            },
        ]
        binary_calibration_rows.extend(calibration_rows)
        binary_calibrators[market] = (binary_name, binary_calibrator, binary_scores)
        calibration_market_name, calibrator, _ = binary_calibrators[market]
        validation_market_holdout = predicted
        adjusted = (
            calibrator.predict_proba(predicted)
            if calibrator is not None and calibration_market_name == "isotonic"
            else predicted
        )
        rows.append(
            _binary_metrics(
                "lightgbm",
                "validation",
                validation_season,
                target,
                validation_market_holdout,
                market,
            )
        )
        rows.append(
            _binary_metrics(
                f"lightgbm_calibrated_{calibration_market_name}",
                "validation",
                validation_season,
                target,
                adjusted,
                market,
            )
        )

    # walk-forward test records by season; fit feature generation on available history through
    # each fold using chronological transforms where future labels cannot affect earlier features.
    for season in split.test_seasons:
        test = split.test.loc[split.test["season"].astype(str).eq(season)].copy()
        first_day = pd.to_datetime(test["date"]).min()
        training = matches.loc[pd.to_datetime(matches["date"]) < first_day].copy()
        test_features = all_features.loc[test.index].reset_index(drop=True)
        training_features = all_features.loc[training.index].reset_index(drop=True)
        fitted = LightGBM1X2(best_config).fit(training_features, training)
        raw_probabilities = fitted.predict_proba(test_features)
        calibrated_probabilities = apply_multiclass_calibrator(
            calibration_name, multiclass_calibrator, raw_probabilities
        )
        test_market_probabilities = bookmaker_probabilities(test)[0]
        rows.append(
            _metric_row(
                "lightgbm",
                "test",
                season,
                test,
                raw_probabilities,
                bookmaker_reference=test_market_probabilities,
            )
        )
        rows.append(
            _metric_row(
                f"lightgbm_calibrated_{calibration_name}",
                "test",
                season,
                test,
                calibrated_probabilities,
                bookmaker_reference=test_market_probabilities,
            )
        )
        test_dc = DixonColesModel(
            DixonColesConfig(
                xi=float(settings["dixon_coles"]["validation_xi"][1]),
                max_goals=int(settings["dixon_coles"]["max_goals"]),
                maxiter=int(settings["dixon_coles"]["score_optimization_maxiter"]),
            )
        ).fit(training)
        test_dc_probabilities = test_dc.predict_proba(test)
        test_dc_markets = test_dc.predict_markets(test)
        ensemble_test = combine_probabilities(
            test_dc_probabilities, raw_probabilities, ensemble_weight
        )
        rows.append(
            _metric_row(
                "ensemble",
                "test",
                season,
                test,
                ensemble_test,
                bookmaker_reference=test_market_probabilities,
            )
        )
        for baseline_name, baseline_probabilities in (
            ("class_prior", ClassPriorModel().fit(training).predict_proba(test)),
            ("bookmaker", bookmaker_probabilities(test)[0]),
            ("dixon_coles", test_dc_probabilities),
        ):
            rows.append(
                _metric_row(
                    baseline_name,
                    "test",
                    season,
                    test,
                    baseline_probabilities,
                    bookmaker_reference=test_market_probabilities,
                )
            )

        test_elo = EloModel(EloConfig()).fit(training)
        rows.append(
            _metric_row(
                "elo",
                "test",
                season,
                test,
                test_elo.predict_proba(test),
                bookmaker_reference=test_market_probabilities,
            )
        )
        train_targets_by_market = train_targets(training)
        test_targets = train_targets(test)
        for market in train_targets_by_market:
            dc_binary_probability = np.asarray(
                [
                    item["over_2_5"] if market == "over_2_5" else item["btts_yes"]
                    for item in test_dc_markets
                ]
            )
            rows.append(
                _binary_metrics(
                    "dixon_coles",
                    "test",
                    season,
                    test_targets[market],
                    dc_binary_probability,
                    market,
                    bookmaker_reference=(
                        _market_bookmaker_probabilities(test, market)
                        if market == "over_2_5"
                        else None
                    ),
                )
            )
        for market, train_target in train_targets_by_market.items():
            binary = LightGBMBinary(best_config).fit(training_features, train_target)
            test_probability = binary.predict_proba(test_features)
            rows.append(
                _binary_metrics(
                    "class_prior",
                    "test",
                    season,
                    test_targets[market],
                    np.full(
                        len(test),
                        float(train_target.mean()),
                    ),
                    market,
                    bookmaker_reference=(
                        _market_bookmaker_probabilities(test, market)
                        if market == "over_2_5"
                        else None
                    ),
                )
            )
            if market == "over_2_5":
                rows.append(
                    _binary_metrics(
                        "bookmaker",
                        "test",
                        season,
                        test_targets[market],
                        _market_bookmaker_probabilities(test, market),
                        market,
                        "bet365_over_under",
                        bookmaker_reference=_market_bookmaker_probabilities(test, market),
                    )
                )
            rows.append(
                _binary_metrics(
                    "lightgbm",
                    "test",
                    season,
                    test_targets[market],
                    test_probability,
                    market,
                    bookmaker_reference=(
                        _market_bookmaker_probabilities(test, market)
                        if market == "over_2_5"
                        else None
                    ),
                )
            )
            calibration_market_name, calibrator, _ = binary_calibrators[market]
            if calibration_market_name == "isotonic" and calibrator is not None:
                test_probability_calibrated = calibrator.predict_proba(test_probability)
            else:
                test_probability_calibrated = test_probability
            rows.append(
                _binary_metrics(
                    f"lightgbm_calibrated_{calibration_market_name}",
                    "test",
                    season,
                    test_targets[market],
                    test_probability_calibrated,
                    market,
                )
            )

    results = pd.DataFrame(rows)
    destination = Path(output_path).with_name("phase4-model-comparison.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False, float_format="%.8f")
    tuning = pd.DataFrame(candidates).sort_values("validation_log_loss")
    tuning.to_csv(
        destination.with_name("lightgbm-validation-tuning.csv"),
        index=False,
        float_format="%.8f",
    )
    calibration_table = [
        {"market": "1x2", "calibrator": name, "validation_log_loss": score}
        for name, score in calibration_scores.items()
    ]
    calibration_table.extend(binary_calibration_rows)
    pd.DataFrame(calibration_table).to_csv(
        destination.with_name("lightgbm-calibration-validation.csv"),
        index=False,
        float_format="%.8f",
    )
    destination.with_suffix(".md").write_text(
        "# Phase 4 model comparison\n\n"
        "Generated by `footballpred evaluate`. LightGBM hyperparameters are chosen by validation "
        "1X2 log loss. Temperature scaling, one-vs-rest Platt, and class-wise isotonic are "
        "compared using validation predictions from models trained on earlier seasons. Isotonic "
        "is diagnostic only due to overfit risk on the same calibration sample. "
        "A parametric calibrator is retained only if validation log loss improves. "
        "Test seasons are expanding-window folds. For 1X2 and Over 2.5, log-loss difference "
        "is model minus bookmaker; paired 95% bootstrap intervals resample match-level "
        "differences. Positive values favor the bookmaker. BTTS odds are absent from this "
        "source.\n\n"
        + _to_markdown(results)
        + "\n\nSee `lightgbm-validation-tuning.csv` and `lightgbm-calibration-validation.csv` for "
        "validation-only choices.\n",
        encoding="utf-8",
    )
    _plot_reliability(
        validation_truth,
        validation_calibrated,
        Path("docs/reliability-validation.png"),
        title="1X2 validation reliability — LightGBM selected calibration",
    )
    _write_explainability_artifacts(
        matches=matches,
        train=split.train,
        features=all_features,
        model=best_model,
        validation_features=validation_features,
        config=best_config,
        validation_season=validation_season,
        dc_model=validation_dc,
    )
    return results


def _plot_reliability(
    truth: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Save class-wise reliability curves for H/D/A probabilities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="ideal")
    labels = ("Home", "Draw", "Away")
    edges = np.linspace(0.0, 1.0, 11)
    for index, label in enumerate(labels):
        predicted_bin: list[float] = []
        observed_bin: list[float] = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            selected = (probabilities[:, index] >= left) & (probabilities[:, index] < right)
            if right == 1.0:
                selected |= probabilities[:, index] == 1.0
            if selected.any():
                predicted_bin.append(float(probabilities[selected, index].mean()))
                observed_bin.append(float(truth[selected, index].mean()))
        axis.plot(predicted_bin, observed_bin, marker="o", label=label)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
    )
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_walk_forward_figure(results: pd.DataFrame, path: str | Path) -> None:
    """Plot per-season 1X2 log loss with the bookmaker as the explicit reference."""
    frame = results.loc[
        results["partition"].eq("test") & results["market"].eq("1x2")
    ].copy()
    selected = frame.loc[
        frame["model"].isin(
            ["bookmaker", "elo", "dixon_coles", "lightgbm", "lightgbm_calibrated_platt", "ensemble"]
        )
    ]
    pivot = selected.pivot(index="season", columns="model", values="log_loss")
    season_labels = [f"{season[:2]}/{season[2:]}" for season in pivot.index.astype(str)]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for model_name in pivot.columns:
        axis.plot(
            season_labels,
            pivot[model_name],
            marker="o",
            linewidth=2,
            label=model_name.replace("lightgbm_calibrated_platt", "LightGBM + Platt"),
        )
    axis.set_title("Held-out 1X2 log loss by season (lower is better)")
    axis.set_xlabel("Test season")
    axis.set_ylabel("Multiclass log loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _write_explainability_artifacts(
    matches: pd.DataFrame,
    train: pd.DataFrame,
    features: pd.DataFrame,
    model: LightGBM1X2,
    validation_features: pd.DataFrame,
    config: LightGBMConfig,
    validation_season: str,
    dc_model: DixonColesModel,
) -> None:
    """Save LightGBM global SHAP summary, sample explanations, and DC strength ratings."""
    importance = compute_global_importance(model, validation_features, max_samples=300)
    save_global_importance(
        importance,
        "results/shap-global-importance.csv",
        "docs/shap-global-importance.png",
    )
    sample = matches.loc[matches["season"].astype(str).eq(validation_season)].head(3)
    explanations = []
    model_features = build_features(matches)
    explanation_model = LightGBM1X2(config).fit(
        features.loc[train.index].reset_index(drop=True), train
    )
    for index, row in sample.iterrows():
        feature_row = model_features.loc[[index]].reset_index(drop=True)
        explanations.append(
            explain_match(
                explanation_model,
                feature_row,
                str(row["home_team"]),
                str(row["away_team"]),
            )
        )
    _write_explanation_markdown(explanations, "docs/sample-explanations.md")
    strengths = dc_model.team_strengths()
    strengths.to_csv("results/dixon-coles-team-strengths.csv", index=False, float_format="%.8f")
    Path("docs/dixon-coles-team-strengths.md").write_text(
        "# Dixon–Coles fitted team strengths\n\n"
        "Centered attack/defence log strengths from the model fitted on the training partition. "
        "Positive attack indicates scoring above the training-team mean; positive defence means "
        "higher conceded-goal rate (weaker defence). Home advantage is the shared "
        "log-rate term.\n\n"
        + _to_markdown(strengths.head(12))
        + "\n",
        encoding="utf-8",
    )


def _write_explanation_markdown(explanations: list[dict[str, Any]], path: str | Path) -> None:
    """Render sample predictions and top feature contributions in readable Markdown."""
    lines = [
        "# Example LightGBM explanations",
        "",
        "Generated from the validation season with a model trained on earlier seasons.",
        "SHAP values are contributions to the selected class raw score, not probability points.",
        "",
    ]
    for number, item in enumerate(explanations, start=1):
        probabilities = item["probabilities"]
        lines.extend(
            [
                f"## Example {number}: {item['home_team']} vs {item['away_team']}",
                "",
                f"Most likely outcome: **{item['most_likely_outcome']}**.",
                "",
                "Probabilities: "
                f"home {probabilities['home_win']:.1%}, draw {probabilities['draw']:.1%}, "
                f"away {probabilities['away_win']:.1%}.",
                "",
                "Top contributing features:",
            ]
        )
        for feature in item["top_features"]:
            lines.append(
                f"- `{feature['label']}`: {feature['direction']} selected-class raw score "
                f"({feature['shap_value']:+.4f}); {feature['interpretation']}"
            )
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _bookmaker_binary_probabilities(
    matches: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, str]] | None:
    """Convert available over/under and BTTS odds into proportional no-margin probabilities."""
    output: dict[str, tuple[np.ndarray, str]] = {}
    # In football-data source conventions, >2.5 fields encode total-goal market only,
    # so don't mistake them for BTTS prices. BTTS is emitted only if explicit BTTS columns exist.
    odds_candidates = {
        "over_2_5": (
            ("b365_over_2.5", "b365_under_2.5", "bet365_over_under"),
            ("avg_over_2.5", "avg_under_2.5", "average_over_under"),
        ),
        "btts": (
            ("b365btts_yes", "b365btts_no", "bet365_btts"),
            ("avgbtts_yes", "avgbtts_no", "average_btts"),
        ),
    }
    for market, sources in odds_candidates.items():
        for positive_col, negative_col, label in sources:
            if positive_col not in matches or negative_col not in matches:
                continue
            values = (
                matches[[positive_col, negative_col]]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy()
            )
            valid = np.isfinite(values).all(axis=1) & (values > 1.0).all(axis=1)
            if valid.all():
                implied = 1 / values
                output[market] = (implied[:, 0] / implied.sum(axis=1), label)
                break
    return output or None


def _market_bookmaker_probabilities(matches: pd.DataFrame, market: str) -> np.ndarray:
    """Extract de-vigged market probability for O/U; reject absent market data."""
    column_pairs = {
        "over_2_5": (
            ("b365_over_2.5", "b365_under_2.5"),
            ("avg_over_2.5", "avg_under_2.5"),
        )
    }
    if market not in column_pairs:
        raise ValueError(f"No bookmaker binary market available for {market}")
    for positive, negative in column_pairs[market]:
        if positive not in matches or negative not in matches:
            continue
        values = matches[[positive, negative]].apply(pd.to_numeric, errors="coerce").to_numpy()
        valid = np.isfinite(values).all(axis=1) & (values > 1.0).all(axis=1)
        if valid.all():
            implied = 1 / values
            return implied[:, 0] / implied.sum(axis=1)
    raise ValueError(f"No complete bookmaker odds for {market}")




def _to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact markdown table without an optional tabulate dependency."""
    columns = list(frame.columns)
    rows = [
        [str(value) if not isinstance(value, float) else f"{value:.4f}" for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(column).ljust(widths[index]) for index, column in enumerate(columns)
    ) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])
