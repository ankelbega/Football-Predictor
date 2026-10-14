"""Chronological baseline and Elo evaluation with reproducible result tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    outcome_probabilities,
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
) -> dict[str, Any]:
    truth = outcome_probabilities(observed)
    return {
        "model": model_name,
        "partition": partition,
        "season": season,
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
    """Tune Elo on validation; report one-season validation and per-season held-out metrics."""
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
