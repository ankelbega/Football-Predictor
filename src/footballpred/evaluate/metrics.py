"""Proper scoring rules and probability calibration diagnostics."""

from __future__ import annotations

import numpy as np

from footballpred.models.baselines import OUTCOMES, outcome_probabilities


def log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...] = OUTCOMES,
) -> float:
    """Multiclass log loss with explicit class order."""
    target = _as_one_hot(y_true)
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != target.shape:
        raise ValueError("Target and probability arrays must have matching (n, 3) shapes")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1")
    return float(-np.sum(target * np.log(np.clip(probs, 1e-15, 1.0))) / len(target))


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean multiclass Brier score (sum of squared class errors per fixture)."""
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Three-category RPS: mean sum of squared cumulative-probability errors / 2."""
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    cumulative_target = np.cumsum(target, axis=1)[:, :-1]
    cumulative_probabilities = np.cumsum(probs, axis=1)[:, :-1]
    return float(np.mean(np.sum((cumulative_probabilities - cumulative_target) ** 2, axis=1) / 2.0))


def accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Fraction of correct modal-outcome classifications."""
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    return float(np.mean(np.argmax(probs, axis=1) == np.argmax(target, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Top-label expected calibration error, weighted by bin frequency."""
    if bins < 1:
        raise ValueError("bins must be positive")
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == target.argmax(axis=1)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if in_bin.any():
            score += in_bin.mean() * abs(float(correct[in_bin].mean() - confidence[in_bin].mean()))
    return float(score)


def outcome_rate_vector(result_labels: list[str] | np.ndarray) -> np.ndarray:
    """Convenience wrapper to compute empirical prior probabilities from outcome labels."""
    return outcome_probabilities_from_labels(result_labels).mean(axis=0)


def outcome_probabilities_from_labels(result_labels: list[str] | np.ndarray) -> np.ndarray:
    """Convert H/D/A outcome labels into one-hot rows."""
    import pandas as pd

    return outcome_probabilities(pd.DataFrame({"result": result_labels}))


def _as_one_hot(y_true: np.ndarray) -> np.ndarray:
    values = np.asarray(y_true)
    if values.ndim == 1:
        return outcome_probabilities_from_labels(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("y_true must be H/D/A labels or an (n, 3) one-hot matrix")
    return values.astype(float)


def _validate_shape(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != target.shape:
        raise ValueError("Target and probability arrays must have matching (n, 3) shapes")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1")
    return probs
