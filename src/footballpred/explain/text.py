"""Turn a LightGBM forecast and SHAP values into readable match explanations.

This explain-stage module is called by sample-report generation and the prediction API. It
keeps interpretation on raw model-score scale rather than unsupported probability points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footballpred.explain.shap_utils import local_shap_values
from footballpred.models.lgbm import LightGBM1X2

FEATURE_LABELS = {
    "elo_difference": "Elo home advantage over away",
    "elo_home": "home team Elo rating",
    "elo_away": "away team Elo rating",
    "home_overall_last5_points_per_match": "home team's recent points form",
    "away_overall_last5_points_per_match": "away team's recent points form",
    "home_home_last5_points_per_match": "home team's home points form",
    "away_away_last5_points_per_match": "away team's away points form",
    "home_overall_last5_goals_for_per_match": "home team's recent goals scored",
    "away_overall_last5_goals_for_per_match": "away team's recent goals scored",
    "home_overall_last5_goals_against_per_match": "home team's recent goals conceded",
    "away_overall_last5_goals_against_per_match": "away team's recent goals conceded",
    "home_days_rest": "home team's rest",
    "away_days_rest": "away team's rest",
    "season_progress": "season progress",
}


def explain_match(
    model: LightGBM1X2,
    feature_row: pd.DataFrame,
    home_team: str,
    away_team: str,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build a match forecast and rank its largest signed SHAP contributions.

    Args:
        model: Fitted LightGBM 1X2 model.
        feature_row: One causal pre-match feature row.
        home_team: Human-readable home club name.
        away_team: Human-readable away club name.
        top_n: Maximum number of absolute contributions to return.

    Returns:
        JSON-compatible probabilities, predicted class, explanation scale and feature details.

    SHAP contributions explain class raw scores, not normalized probability points; the response
    states this distinction rather than implying direct percentage-point attribution.
    """
    probabilities = model.predict_proba(feature_row)[0]
    class_index = int(np.argmax(probabilities))
    shap_values, base_value = local_shap_values(model, feature_row, class_index)
    ranked = np.argsort(np.abs(shap_values))[::-1][:top_n]
    explanations = []
    for index in ranked:
        feature_name = model.feature_columns_[index]
        value = feature_row.iloc[0][feature_name]
        contribution = float(shap_values[index])
        explanations.append(
            {
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ")),
                "feature_value": None if pd.isna(value) else float(value),
                "shap_value": contribution,
                "direction": "raised" if contribution >= 0 else "lowered",
                "interpretation": _describe_feature(feature_name, value, contribution),
            }
        )
    class_names = ("home win", "draw", "away win")
    return {
        "home_team": home_team,
        "away_team": away_team,
        "probabilities": {
            "home_win": float(probabilities[0]),
            "draw": float(probabilities[1]),
            "away_win": float(probabilities[2]),
        },
        "most_likely_outcome": class_names[class_index],
        "explanation_scale": (
            "SHAP contribution to the selected class raw score; not percentage points"
        ),
        "base_value": base_value,
        "top_features": explanations,
    }


def _describe_feature(feature: str, value: object, shap_value: float) -> str:
    """Format one feature value and its sign without claiming a probability-point change."""
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    if pd.isna(value):
        detail = "is unavailable"
    elif "points_per_match" in feature:
        detail = f"is {float(value):.2f} points per match"
    elif "goals_" in feature:
        detail = f"is {float(value):.2f} per match"
    elif "elo" in feature:
        detail = f"is {float(value):.0f}"
    elif "days_rest" in feature:
        detail = f"is {float(value):.0f} days"
    else:
        detail = f"is {float(value):.3g}"
    verb = "increased" if shap_value >= 0 else "decreased"
    return (
        f"{label} ({detail}) {verb} the model score for the "
        f"{feature.split('_')[0]}-class prediction."
    )
