"""Train and load a deployable chronological LightGBM forecasting bundle."""

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
    """Fit outcome/market models using all matches available before an optional cutoff.

    Retrospective validation and model selection remain in evaluate. This helper only fits the
    final serving artifacts, using fixed selected configuration settings.
    """
    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if training_cutoff is not None:
        cutoff = pd.Timestamp(training_cutoff)
        data = data.loc[data["date"] < cutoff].copy()
    if data.empty:
        raise ValueError("No training matches precede the specified cutoff")
    features = build_features(data)
    model_settings = settings["lightgbm"]
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
    """Persist a trained bundle with joblib to a local model artifact path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a previously trained model bundle; never retrains on load."""
    destination = Path(path)
    if not destination.exists():
        raise FileNotFoundError(
            f"Trained model not found at {destination}; run `footballpred train` first"
        )
    bundle = joblib.load(destination)
    if not isinstance(bundle, dict) or not {"models", "teams", "training_cutoff"} <= bundle.keys():
        raise ValueError(f"Invalid model bundle at {destination}")
    return bundle
