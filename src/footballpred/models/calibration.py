"""Fit probability calibration mappings using validation predictions.

This model-stage module is called by evaluation between raw model prediction and scoring.
Calibration can improve reliability, but isotonic's flexibility makes it easy to overfit small
validation samples, so selection rules deliberately keep that risk visible.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from footballpred.evaluate.metrics import binary_log_loss, log_loss
from footballpred.models.baselines import OUTCOMES


class TemperatureScaler:
    """Adjust multiclass confidence with one positive temperature parameter."""

    def __init__(self) -> None:
        """Start with temperature one, which leaves input probabilities unchanged."""
        self.temperature_: float = 1.0

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> TemperatureScaler:
        """Find the temperature that minimizes validation log loss.

        Args:
            probabilities: Validation ``(n, 3)`` raw probabilities in H/D/A order.
            y_true: Validation labels using H, D and A.

        Returns:
            This fitted scaler. The optimizer works on log-temperature so temperature stays
            positive.
        """
        # Temperature divides logits: below one sharpens confidence, above one softens it.
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
        logits = np.log(probs)
        labels = np.asarray(y_true)
        classes = {label: index for index, label in enumerate(OUTCOMES)}
        target = np.array([classes[str(label)] for label in labels], dtype=int)

        def objective(log_temperature: float) -> float:
            scaled = logits / np.exp(log_temperature)
            log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
            return float(-np.mean(log_probs[np.arange(len(target)), target]))

        # Bounds keep extreme temperatures from collapsing probabilities to numerical 0/1.
        result = minimize_scalar(objective, bounds=(-2.5, 2.5), method="bounded")
        self.temperature_ = float(np.exp(result.x))
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature and return normalized class probabilities.

        Args:
            probabilities: Matrix in the same class order as fit.

        Returns:
            Temperature-scaled matrix with rows summing to one.
        """
        logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0))
        scaled = logits / self.temperature_
        scaled -= logsumexp(scaled, axis=1, keepdims=True)
        return np.exp(scaled)


class PlattScaler:
    """Fit one logistic calibration curve per outcome, then renormalize their outputs."""

    def __init__(self) -> None:
        """Create a scaler with no fitted outcome curves."""
        self.models_: list[LogisticRegression] = []

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> PlattScaler:
        """Fit one-vs-rest logistic regressions from validation probabilities.

        Args:
            probabilities: Validation matrix with H/D/A columns.
            y_true: Validation H/D/A labels.

        Returns:
            This fitted scaler.
        """
        # One-vs-rest Platt fits P(class | that class's raw probability); normalize after.
        values = np.asarray(probabilities, dtype=float)
        labels = np.asarray(y_true)
        self.models_ = []
        for index, label in enumerate(OUTCOMES):
            model = LogisticRegression(C=1e6, solver="lbfgs", random_state=2026)
            model.fit(values[:, index].reshape(-1, 1), (labels == label).astype(int))
            self.models_.append(model)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply each calibrated one-vs-rest curve and normalize across classes.

        Args:
            probabilities: Matrix with the same three input probability columns.

        Returns:
            Calibrated matrix in H/D/A order, row-normalized to sum to one.
        """
        values = np.asarray(probabilities, dtype=float)
        calibrated = np.column_stack(
            [model.predict_proba(values[:, index].reshape(-1, 1))[:, 1]
             for index, model in enumerate(self.models_)]
        )
        calibrated = np.clip(calibrated, 1e-12, 1.0)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


class IsotonicBinaryScaler:
    """Binary isotonic calibration, fitted exclusively on validation predictions."""

    def __init__(self) -> None:
        """Create an unfitted binary isotonic mapping."""
        self.model_: IsotonicRegression | None = None
        self.identity_: bool = False

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> IsotonicBinaryScaler:
        """Fit a monotonic map from validation confidence to observed binary frequency.

        Args:
            probabilities: Validation positive-event probabilities.
            y_true: Aligned zero/one event labels.

        Returns:
            This fitted scaler; values outside the training range are clipped to endpoints.
        """
        self.model_ = IsotonicRegression(out_of_bounds="clip").fit(probabilities, y_true)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Map raw binary probabilities to isotonic-calibrated values.

        Args:
            probabilities: Positive-event probability vector.

        Returns:
            Calibrated positive-event probabilities. ``identity_`` is a serving safeguard
            used by the evaluator when a selected map should not be refit on the same rows.
        """
        if self.model_ is None:
            raise RuntimeError("IsotonicBinaryScaler.fit must be called before prediction")
        if self.identity_:
            return np.asarray(probabilities, dtype=float)
        return np.asarray(self.model_.predict(probabilities), dtype=float)


class IsotonicMulticlassScaler:
    """Fit one monotonic calibration curve per class, then normalize outputs."""

    def __init__(self, smoothing: float = 0.01) -> None:
        """Create an unfitted scaler.

        Args:
            smoothing: Small share of observed validation class rates added before normalization.
        """
        self.models_: list[IsotonicRegression] = []
        self.priors_: np.ndarray | None = None
        self.smoothing = smoothing
        self.identity_: bool = False

    def fit(
        self, probabilities: np.ndarray, y_true: np.ndarray
    ) -> IsotonicMulticlassScaler:
        """Fit class curves and class-rate smoothing priors on validation rows.

        Args:
            probabilities: Validation matrix ordered H/D/A.
            y_true: Validation H/D/A labels.

        Returns:
            This fitted scaler.
        """
        raw = np.asarray(probabilities, dtype=float)
        labels = np.asarray(y_true)
        self.priors_ = np.array([(labels == label).mean() for label in OUTCOMES])
        self.identity_ = False
        self.models_ = [
            IsotonicRegression(out_of_bounds="clip").fit(
                raw[:, index], (labels == label).astype(int)
            )
            for index, label in enumerate(OUTCOMES)
        ]
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply one-vs-rest maps and renormalize the class probabilities.

        Args:
            probabilities: Matrix ordered H/D/A.

        Returns:
            Normalized calibrated matrix.
        """
        if self.priors_ is None or not self.models_:
            raise RuntimeError("IsotonicMulticlassScaler.fit must be called before prediction")
        raw = np.asarray(probabilities, dtype=float)
        mapped = np.column_stack(
            [model.predict(raw[:, index]) for index, model in enumerate(self.models_)]
        )
        mapped = mapped + self.smoothing * self.priors_
        mapped = np.clip(mapped, 1e-12, 1.0)
        return mapped / mapped.sum(axis=1, keepdims=True)


def select_multiclass_calibrator(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[str, object, dict[str, float]]:
    """Compare calibration candidates without promoting in-sample isotonic fit scores.

    Args:
        validation_probabilities: Model forecasts from the validation period.
        validation_labels: Actual H/D/A outcomes for those rows.

    Returns:
        Selected name, fitted calibrator or None for raw output, and diagnostic validation
        log losses for raw, temperature, Platt and isotonic candidates.
    """
    raw = np.asarray(validation_probabilities, dtype=float)
    candidates: dict[str, tuple[object, np.ndarray]] = {"raw": (None, raw)}
    temperature = TemperatureScaler().fit(raw, validation_labels)
    candidates["temperature"] = (temperature, temperature.predict_proba(raw))
    platt = PlattScaler().fit(raw, validation_labels)
    candidates["platt"] = (platt, platt.predict_proba(raw))
    isotonic = IsotonicMulticlassScaler().fit(raw, validation_labels)
    candidates["isotonic"] = (isotonic, isotonic.predict_proba(raw))
    scores = {
        name: log_loss(validation_labels, predicted)
        for name, (_, predicted) in candidates.items()
    }
    best_name = min(scores, key=scores.get)
    # Keep calibration only when a fitted calibration strictly beats validation raw score.
    # Isotonic is highly prone to memorizing its calibration sample in this modest dataset.
    # Unless independently selected using a separate temporal calibration split, exclude it
    # from the final selectable set; callers can still inspect its validation-only diagnostic.
    if best_name == "isotonic":
        non_isotonic = {name: score for name, score in scores.items() if name != "isotonic"}
        best_name = min(non_isotonic, key=non_isotonic.get)
    if best_name == "raw" or scores[best_name] >= scores["raw"] - 1e-12:
        return "raw", None, scores
    return best_name, candidates[best_name][0], scores


def apply_multiclass_calibrator(
    name: str,
    calibrator: object,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Apply a selected mapping, or return the raw matrix when selection chose no calibration.

    Args:
        name: Candidate name returned by ``select_multiclass_calibrator``.
        calibrator: Fitted mapping returned by the selector, or None for raw.
        probabilities: Matrix whose columns follow H/D/A order.

    Returns:
        Probability matrix in the same row and column order.
    """
    if name == "raw" or calibrator is None:
        return np.asarray(probabilities, dtype=float)
    if name == "isotonic":
        return calibrator.predict_proba(probabilities)
    return calibrator.predict_proba(probabilities)


def select_binary_isotonic(
    validation_probabilities: np.ndarray,
    validation_target: np.ndarray,
) -> tuple[str, object | None, dict[str, float]]:
    """Report raw and isotonic validation losses while defaulting to the safer raw forecast.

    Args:
        validation_probabilities: Positive-event forecasts from validation matches.
        validation_target: Aligned binary outcomes.

    Returns:
        Raw selection, no fitted serving calibrator, and both validation loss diagnostics.
    """
    raw = np.asarray(validation_probabilities, dtype=float)
    target = np.asarray(validation_target, dtype=int)
    calibrator = IsotonicBinaryScaler().fit(raw, target)
    calibrated = calibrator.predict_proba(raw)
    scores = {
        "raw": binary_log_loss(target, raw),
        "isotonic": binary_log_loss(target, calibrated),
    }
    return "raw", None, scores
