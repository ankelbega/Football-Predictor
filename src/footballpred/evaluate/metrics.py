"""Score probability forecasts and measure their reliability.

This evaluate-stage module is used by model selection and result reports. Lower log loss,
Brier score and RPS mean better forecasts; ECE summarizes confidence/frequency mismatch.
"""

from __future__ import annotations

import numpy as np

from footballpred.models.baselines import OUTCOMES, outcome_probabilities


def log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...] = OUTCOMES,
) -> float:
    """Compute mean negative log probability assigned to the observed outcome.

    Args:
        y_true: H/D/A labels or a one-hot ``(n, 3)`` target matrix.
        probabilities: Forecast matrix with columns in the supplied label order.
        labels: Canonical order used when converting string labels.

    Returns:
        Mean ``-log(p_observed)``. Very small probabilities are clipped to avoid log zero.
    """
    # Proper score: punish confident mistakes sharply via negative log assigned to truth.
    # The labels argument documents the expected order; the project uses the shared H/D/A order.
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
    """Compute mean sum of squared errors over the three outcome probabilities.

    Args:
        y_true: H/D/A labels or one-hot target rows.
        probabilities: Matching normalized H/D/A probability rows.

    Returns:
        Mean ``sum((p_k - y_k)^2)`` per match. Zero is perfect; lower is better.
    """
    # Proper score: squared distance between forecast vector and one-hot outcome vector.
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Score cumulative probability errors for ordered home/draw/away categories.

    Args:
        y_true: H/D/A labels or one-hot target rows in that order.
        probabilities: Matching normalized H/D/A forecast rows.

    Returns:
        Mean squared differences between the first two cumulative probabilities, divided by
        two. The ordering makes near misses (for example home vs draw) less costly than away.
    """
    # Ordered score: compare cumulative probabilities at home/draw and home/draw/away cutoffs.
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    cumulative_target = np.cumsum(target, axis=1)[:, :-1]
    cumulative_probabilities = np.cumsum(probs, axis=1)[:, :-1]
    return float(np.mean(np.sum((cumulative_probabilities - cumulative_target) ** 2, axis=1) / 2.0))


def accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Report the fraction of matches whose most probable class was correct.

    Args:
        y_true: H/D/A labels or one-hot target rows.
        probabilities: Matching H/D/A forecast rows.

    Returns:
        Accuracy between zero and one; this is secondary to proper probability scores.
    """
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    return float(np.mean(np.argmax(probs, axis=1) == np.argmax(target, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Estimate confidence/calibration mismatch across fixed top-probability bins.

    Args:
        y_true: H/D/A labels or one-hot target rows.
        probabilities: Matching H/D/A forecast rows.
        bins: Number of equal-width confidence intervals from zero to one.

    Returns:
        Sum over bins of bin frequency times absolute accuracy-minus-confidence gap.
        This top-label ECE depends on bin count and is a diagnostic, not a proper score.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    target = _as_one_hot(y_true)
    probs = _validate_shape(target, probabilities)
    # ECE weights each confidence bin's gap by its share of all matches.
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == target.argmax(axis=1)).astype(float)
    # Equal-width bins are easy to interpret; empty bins contribute zero to weighted ECE.
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if in_bin.any():
            score += in_bin.mean() * abs(float(correct[in_bin].mean() - confidence[in_bin].mean()))
    return float(score)


def binary_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute mean binary negative log likelihood for positive-event forecasts.

    Args:
        y_true: One-dimensional zero/one event labels.
        probabilities: Probability of event one for each row.

    Returns:
        Mean binary cross-entropy; lower is better.
    """
    target = np.asarray(y_true, dtype=float)
    probs = _validate_binary(target, probabilities)
    return float(-np.mean(target * np.log(np.clip(probs, 1e-15, 1.0))
                          + (1 - target) * np.log(np.clip(1 - probs, 1e-15, 1.0))))


def binary_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute the mean squared error of binary event probabilities.

    Args:
        y_true: One-dimensional zero/one event labels.
        probabilities: Probability of event one for each row.

    Returns:
        Mean squared probability error; zero is perfect.
    """
    target = np.asarray(y_true, dtype=float)
    probs = _validate_binary(target, probabilities)
    return float(np.mean((probs - target) ** 2))


def _validate_binary(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Check aligned binary labels and bounded event probabilities.

    Args:
        target: One-dimensional zero/one values.
        probabilities: Same-shaped finite vector in [0, 1].

    Returns:
        Float probability vector, or ValueError for malformed inputs.
    """
    probs = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or probs.shape != target.shape:
        raise ValueError("Binary targets and probabilities must be one-dimensional and aligned")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Binary targets must contain only 0 and 1")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return probs


def outcome_rate_vector(result_labels: list[str] | np.ndarray) -> np.ndarray:
    """Return empirical H/D/A rates from a list of result labels.

    Args:
        result_labels: Iterable of H, D or A result codes.

    Returns:
        Three rates ordered home, draw, away.
    """
    return outcome_probabilities_from_labels(result_labels).mean(axis=0)


def outcome_probabilities_from_labels(result_labels: list[str] | np.ndarray) -> np.ndarray:
    """Convert result codes into one-hot rows in H/D/A order.

    Args:
        result_labels: Iterable containing H, D or A.

    Returns:
        One-hot float matrix with three columns.
    """
    import pandas as pd

    return outcome_probabilities(pd.DataFrame({"result": result_labels}))


def _as_one_hot(y_true: np.ndarray) -> np.ndarray:
    """Accept label vectors or already encoded three-class targets."""
    values = np.asarray(y_true)
    if values.ndim == 1:
        return outcome_probabilities_from_labels(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("y_true must be H/D/A labels or an (n, 3) one-hot matrix")
    return values.astype(float)


def _validate_shape(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Check a multiclass forecast array matches its target and sums to one."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != target.shape:
        raise ValueError("Target and probability arrays must have matching (n, 3) shapes")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1")
    return probs
