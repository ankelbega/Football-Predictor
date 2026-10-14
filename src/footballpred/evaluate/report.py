"""Chronological evaluation with reproducible season-wise market results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from footballpred.models.baselines import (
    BookmakerModel,
    ClassPriorModel,
    bookmaker_probabilities,
    outcome_probabilities,
)
from footballpred.models.dixon_coles import (
    DixonColesConfig,
    DixonColesModel,
)
from footballpred.models.elo import EloConfig, EloModel

LOGGER = logging.getLogger(__name__)


def _metric_row(
    model_name: str,
    partition: str,
    season: str,
    observed: pd.DataFrame,
    predicted: np.ndarray,
    source: str | None = None,
    market: str = "1x2",
) -> dict[str, Any]:
    truth = outcome_probabilities(observed)
    return {
        "model": model_name,
        "partition": partition,
        "season": season,
        "market": "1x2",
        "matches": len(observed),
        "odds_source": source,
        "log_loss": log_loss(observed["result"].to_numpy(), predicted),
        "brier_score": multiclass_brier_score(truth, predicted),
        "rps": ranked_probability_score(truth, predicted),
        "accuracy": accuracy(truth, predicted),
        "ece_top_label": expected_calibration_error(truth, predicted),
    }


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
                _metric_row(model_name, "test", season, test_season_frame, predicted, odds_source)
            )
        elo_model = EloModel(elo_config).fit(earlier)
        elo_probabilities = elo_model.predict_proba(test_season_frame)
        rows.append(_metric_row("elo", "test", season, test_season_frame, elo_probabilities))

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
        "Generated by `footballpred evaluate`. The test rows are season-wise "
        "expanding-window evaluations; scores are descriptive and do not imply "
        "market-beating performance.\n\n"
        + markdown_table
        + "\n\nElo tuning candidates were ranked on the validation season only; see "
        f"`{tuning_path.name}`.\n",
        encoding="utf-8",
    )
    return results


def _binary_metrics(
    model_name: str,
    partition: str,
    season: str,
    truth: np.ndarray,
    probabilities: np.ndarray,
    market: str,
    odds_source: str | None = None,
) -> dict[str, Any]:
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
                    "bookmaker", partition, season, observed, market_probs, source, "1x2"
                )
            )
        for model_name, probabilities in (("class_prior", prior_1x2), ("dixon_coles", dc_1x2)):
            rows.append(
                _metric_row(model_name, partition, season, observed, probabilities, market="1x2")
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
