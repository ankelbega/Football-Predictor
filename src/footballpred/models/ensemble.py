"""Tune and apply a simple Dixon–Coles/LightGBM probability blend.

This model-stage helper is called by evaluation after both forecasts are generated. It searches
weights on validation data only and returns convex combinations that remain valid probabilities.
"""

from __future__ import annotations

import numpy as np

from footballpred.evaluate.metrics import log_loss
from footballpred.models.base import validate_probabilities


def tune_ensemble_weight(
    y_validation: np.ndarray,
    dixon_coles_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, dict[float, float]]:
    """Choose the LightGBM blend weight by multiclass validation log loss.

    Args:
        y_validation: H/D/A labels for validation matches.
        dixon_coles_probabilities: Score-model matrix in H/D/A order.
        lightgbm_probabilities: Classifier matrix in the same order.
        weights: Candidate fraction assigned to LightGBM; zero means Dixon–Coles only.

    Returns:
        Best LightGBM fraction and a mapping of all candidate weights to log loss.
    """
    dc = validate_probabilities(dixon_coles_probabilities)
    lgb = validate_probabilities(lightgbm_probabilities)
    if dc.shape != lgb.shape:
        raise ValueError("Ensemble inputs must have identical dimensions")
    scores: dict[float, float] = {}
    for weight in weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Ensemble weights must lie in [0, 1]")
        combined = (1 - weight) * dc + weight * lgb
        scores[float(weight)] = log_loss(y_validation, combined)
    return min(scores, key=scores.get), scores


def combine_probabilities(
    dixon_coles_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    lightgbm_weight: float,
) -> np.ndarray:
    """Blend two aligned H/D/A matrices while preserving probability validity.

    Args:
        dixon_coles_probabilities: First model's ``(n, 3)`` probability matrix.
        lightgbm_probabilities: Second model's ``(n, 3)`` probability matrix.
        lightgbm_weight: Fraction from zero to one assigned to LightGBM.

    Returns:
        Convex combination with each row summing to one.
    """
    if not 0.0 <= lightgbm_weight <= 1.0:
        raise ValueError("Ensemble weights must lie in [0, 1]")
    dc = validate_probabilities(dixon_coles_probabilities)
    lgb = validate_probabilities(lightgbm_probabilities)
    if dc.shape != lgb.shape:
        raise ValueError("Ensemble inputs must have identical dimensions")
    return validate_probabilities((1 - lightgbm_weight) * dc + lightgbm_weight * lgb)
