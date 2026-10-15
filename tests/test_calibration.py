"""Validate calibration transform shapes, bounds and selection contracts."""


import numpy as np

from footballpred.models.calibration import (
    TemperatureScaler,
    PlattScaler,
    IsotonicBinaryScaler,
    IsotonicMulticlassScaler,
    select_multiclass_calibrator,
    apply_multiclass_calibrator,
    select_binary_isotonic,
)
from footballpred.models.baselines import OUTCOMES


def multiclass_fixture() -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.60, 0.25],
            [0.20, 0.25, 0.55],
            [0.45, 0.35, 0.20],
            [0.30, 0.30, 0.40],
        ],
        dtype=float,
    )
    labels = np.array(["H", "D", "A", "H", "A"])
    return probabilities, labels


def binary_fixture() -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.array([0.1, 0.3, 0.45, 0.7, 0.85, 0.95], dtype=float)
    targets = np.array([0, 0, 1, 0, 1, 1], dtype=int)
    return probabilities, targets


def test_temperature_scaler_validity() -> None:
    """Risk: temperature scaling must preserve normalization and keep a positive temperature."""
    probabilities, labels = multiclass_fixture()
    scaler = TemperatureScaler().fit(probabilities, labels)
    calibrated = scaler.predict_proba(probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert scaler.temperature_ > 0


def test_platt_scaler_validity() -> None:
    """Risk: one-vs-rest Platt calibration must return normalized three-class vectors."""
    probabilities, labels = multiclass_fixture()
    scaler = PlattScaler().fit(probabilities, labels)
    calibrated = scaler.predict_proba(probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert not np.isnan(calibrated).any()


def test_isotonic_multiclass_scaler_validity() -> None:
    """Risk: isotonic calibration must remain a valid probability mapping rather than NaNs or negatives."""
    probabilities, labels = multiclass_fixture()
    scaler = IsotonicMulticlassScaler().fit(probabilities, labels)
    calibrated = scaler.predict_proba(probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_isotonic_binary_scaler_validity() -> None:
    """Risk: binary isotonic calibration must stay bounded and monotonic-safe on validation output."""
    probabilities, targets = binary_fixture()
    scaler = IsotonicBinaryScaler().fit(probabilities, targets)
    calibrated = scaler.predict_proba(probabilities)
    assert calibrated.shape == probabilities.shape
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert not np.isnan(calibrated).any()


def test_select_multiclass_calibrator_contract() -> None:
    """Risk: calibrator selection must expose every candidate score and avoid invalid chosen labels."""
    probabilities, labels = multiclass_fixture()
    name, calibrator, scores = select_multiclass_calibrator(probabilities, labels)
    assert name in {"raw", "temperature", "platt", "isotonic"} and name != "isotonic"
    assert set(scores) == {"raw", "temperature", "platt", "isotonic"}
    assert all(np.isfinite(value) for value in scores.values())
    if name != "raw":
        assert calibrator is not None
    calibrated = apply_multiclass_calibrator(name, calibrator, probabilities)
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)


def test_select_binary_isotonic_contract() -> None:
    """Risk: binary calibration selection must default to raw serving output while reporting isotonic diagnostics."""
    probabilities, targets = binary_fixture()
    name, calibrator, scores = select_binary_isotonic(probabilities, targets)
    assert name == "raw"
    assert calibrator is None
    assert set(scores) == {"raw", "isotonic"}
    assert scores["raw"] >= 0
    assert scores["isotonic"] >= 0
    raw = np.asarray(probabilities, dtype=float)
    calibrated = calibrator.predict_proba(raw) if calibrator is not None else raw
    assert np.array_equal(calibrated, raw)