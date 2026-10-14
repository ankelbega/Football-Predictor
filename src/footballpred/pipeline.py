"""Fit, save and load the model bundle used by CLI and API predictions.

This model-to-API bridge builds causal training features, fits market classifiers and a
Dixon–Coles score model, then serializes them with their team history and cutoff metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from footballpred.features.builder import build_features
from footballpred.models.dixon_coles import DixonColesConfig, DixonColesModel
from footballpred.models.elo import EloConfig
from footballpred.models.lgbm import LightGBM1X2, LightGBMBinary, LightGBMConfig, train_targets


def fit_deployable_models(
    matches: pd.DataFrame,
    settings: dict[str, Any],
    training_cutoff: pd.Timestamp | str | None = None,
) -> dict[str, Any]:
    """Fit serving models from all rows before an optional exclusive cutoff.

    Retrospective validation and model selection remain in ``evaluate``; this function only
    fits the final artifact with the already selected settings.

    Args:
        matches: Processed rows with date, team, score, season and result columns.
        settings: Evaluation YAML section containing LightGBM and Dixon–Coles options.
        training_cutoff: Optional exclusive date; later matches are not used.

    Returns:
        Bundle mapping containing fitted models, causal features, history, teams and cutoff.
    """
    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if training_cutoff is not None:
        cutoff = pd.Timestamp(training_cutoff)
        # SPLIT GUARD: exclusive cutoff keeps matches on/after the requested boundary unseen.
        data = data.loc[data["date"] < cutoff].copy()
    if data.empty:
        raise ValueError("No training matches precede the specified cutoff")
    features = build_features(data)
    model_settings = settings["lightgbm"]
    # ASSUMPTION: reuse the code-generated validation winner when present; otherwise the
    # first configured candidate is a deterministic fallback for a fresh install.
    selected_tuning = pd.read_csv("results/lightgbm-validation-tuning.csv") if Path(
        "results/lightgbm-validation-tuning.csv"
    ).exists() else None
    best_estimators = (
        int(selected_tuning.iloc[0]["n_estimators"])
        if selected_tuning is not None
        else int(model_settings["validation_estimators"][0])
    )
    best_leaves = (
        int(selected_tuning.iloc[0]["num_leaves"])
        if selected_tuning is not None
        else int(model_settings["validation_num_leaves"][0])
    )
    model_config = LightGBMConfig(
        n_estimators=best_estimators,
        learning_rate=float(model_settings["learning_rate"]),
        num_leaves=best_leaves,
        min_child_samples=int(model_settings["min_child_samples"]),
        reg_lambda=float(model_settings["reg_lambda"]),
        random_state=int(model_settings["random_state"]),
    )
    multiclass = LightGBM1X2(model_config).fit(features, data)
    binary_models = {
        market: LightGBMBinary(model_config).fit(features, target)
        for market, target in train_targets(data).items()
    }
    dc_settings = settings["dixon_coles"]
    dixon_coles = DixonColesModel(
        DixonColesConfig(
            xi=float(dc_settings["validation_xi"][1]),
            max_goals=int(dc_settings["max_goals"]),
            maxiter=int(dc_settings["score_optimization_maxiter"]),
        )
    ).fit(data)
    return {
        "models": {"lightgbm": multiclass, **binary_models, "dixon_coles": dixon_coles},
        "features": features,
        "teams": sorted(set(data["home_team"].astype(str)) | set(data["away_team"].astype(str))),
        "history": data,
        "training_cutoff": data["date"].max().isoformat(),
        "config": {
            "lightgbm": asdict(model_config),
            "elo": asdict(EloConfig()),
            "dixon_coles": asdict(dixon_coles.config),
        },
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist a trained local bundle for API startup and CLI prediction.

    Args:
        bundle: Mapping returned by ``fit_deployable_models``.
        path: Destination joblib path, typically gitignored under models/.

    Returns:
        Path to the written artifact.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an existing bundle without fitting any model.

    Args:
        path: Existing joblib artifact path.

    Returns:
        Bundle mapping for prediction and serving.

    Raises:
        FileNotFoundError: If training has not created the configured artifact.
        ValueError: If the serialized object lacks required bundle keys.
    """
    destination = Path(path)
    if not destination.exists():
        raise FileNotFoundError(
            f"Trained model not found at {destination}; run `footballpred train` first"
        )
    bundle = joblib.load(destination)
    if not isinstance(bundle, dict) or not {"models", "teams", "training_cutoff"} <= bundle.keys():
        raise ValueError(f"Invalid model bundle at {destination}")
    return bundle
