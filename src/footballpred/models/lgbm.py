"""Train CPU-friendly LightGBM classifiers for the three supported markets.

This model-stage module receives numeric pre-match features from ``features.builder`` and
targets from processed match rows. Evaluation selects settings; API/pipeline code persists fitted
classifiers and returns probabilities in the shared H/D/A order.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from footballpred.models.base import ProbabilityModel, validate_probabilities
from footballpred.models.baselines import OUTCOMES, outcome_probabilities


@dataclass(frozen=True)
class LightGBMConfig:
    """Reproducible CPU training settings shared by market classifiers.

    Args:
        n_estimators: Boosting rounds; more rounds can fit more detail but may overfit.
        learning_rate: Step size per boosting round; smaller values usually need more rounds.
        num_leaves: Maximum leaves per tree; more leaves allow more complex interactions.
        max_depth: Tree depth cap, or -1 for no explicit depth cap.
        min_child_samples: Minimum training rows per leaf; larger values regularize more.
        reg_lambda: L2 penalty on leaf values.
        random_state: Seed for repeatable CPU model fitting.
    """

    n_estimators: int = 220
    learning_rate: float = 0.035
    num_leaves: int = 15
    max_depth: int = -1
    min_child_samples: int = 25
    reg_lambda: float = 2.0
    random_state: int = 2026


class LightGBM1X2(ProbabilityModel):
    """Multiclass LightGBM that returns home/draw/away probabilities.

    Each tree adds a small correction to class scores; softmax converts the final scores to a
    normalized vector. Median imputation is fitted on training features only.
    """

    name = "lightgbm"
    classes_ = np.asarray(OUTCOMES)

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        """Create an unfitted 1X2 classifier.

        Args:
            config: CPU training settings, or defaults.
        """
        self.config = config or LightGBMConfig()
        self.model_: lgb.LGBMClassifier | None = None
        self.imputer_: SimpleImputer | None = None
        self.feature_columns_: list[str] = []

    def fit(self, features: pd.DataFrame, matches: pd.DataFrame) -> LightGBM1X2:
        """Fit multiclass model and median imputer using aligned training rows.

        Args:
            features: Numeric pre-match feature dataframe; row order matches ``matches``.
            matches: Processed matches with H/D/A values in the ``result`` column.

        Returns:
            This fitted classifier.
        """
        self.feature_columns_ = _feature_columns(features)
        matrix = _impute_fit(features[self.feature_columns_])
        self.imputer_, x_train = matrix
        target = matches["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
        self.model_ = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        self.model_.fit(x_train, target)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict valid three-class probabilities in canonical H/D/A order.

        Args:
            features: Feature dataframe with the fitted model's feature columns.

        Returns:
            Array of shape ``(n_matches, 3)``. Missing training classes retain zero mass
            before a tiny epsilon and final normalization make rows valid.
        """
        if self.model_ is None or self.imputer_ is None:
            raise RuntimeError("LightGBM1X2.fit must be called before predict_proba")
        x = self.imputer_.transform(features[self.feature_columns_])
        # LEAKAGE GUARD: the imputer is fitted on this model's training fold, never on prediction rows.
        raw = self.model_.predict_proba(x)
        # In a degenerate training fold a class can be absent; restore canonical H/D/A columns.
        output = np.zeros((len(features), 3), dtype=float)
        for position, label in enumerate(self.model_.classes_):
            output[:, int(label)] = raw[:, position]
        output += 1e-12
        output /= output.sum(axis=1, keepdims=True)
        return validate_probabilities(output)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return LightGBM split-count importances indexed by feature name."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")
        return pd.Series(self.model_.feature_importances_, index=self.feature_columns_).sort_values(
            ascending=False
        )


class LightGBMBinary:
    """Binary LightGBM model returning probability of the positive market event.

    For Over 2.5, positive means three or more total goals. For BTTS, positive means both teams
    scored. The target meaning is supplied by ``train_targets`` and kept aligned by row order.
    """

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        """Create an unfitted binary classifier.

        Args:
            config: Shared LightGBM fit settings, or defaults.
        """
        self.config = config or LightGBMConfig()
        self.model_: lgb.LGBMClassifier | None = None
        self.imputer_: SimpleImputer | None = None
        self.feature_columns_: list[str] = []
        self.constant_probability_: float | None = None

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> LightGBMBinary:
        """Fit a binary market model; use a constant prior if training has one class.

        Args:
            features: Numeric causal pre-match feature rows.
            target: Aligned binary outcomes (one means the named positive market occurred).

        Returns:
            This fitted model.
        """
        self.feature_columns_ = _feature_columns(features)
        self.imputer_, x_train = _impute_fit(features[self.feature_columns_])
        y = np.asarray(target, dtype=int)
        if len(np.unique(y)) < 2:
            self.constant_probability_ = float(y.mean()) if len(y) else 0.5
            return self
        self.model_ = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        self.model_.fit(x_train, y)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-event probabilities for each feature row.

        Args:
            features: Rows with the same feature columns used in fitting.

        Returns:
            One-dimensional probabilities between zero and one.
        """
        if self.imputer_ is None:
            raise RuntimeError("LightGBMBinary.fit must be called before predict_proba")
        if self.constant_probability_ is not None:
            return np.full(len(features), self.constant_probability_, dtype=float)
        if self.model_ is None:
            raise RuntimeError("Binary model failed to fit")
        return self.model_.predict_proba(self.imputer_.transform(features[self.feature_columns_]))[:, 1]


def train_targets(matches: pd.DataFrame) -> dict[str, np.ndarray]:
    """Derive Over 2.5 and BTTS binary labels from final score columns.

    Args:
        matches: Processed match dataframe with home_goals and away_goals.

    Returns:
        Aligned arrays: ``over_2_5`` is true for at least three goals; ``btts`` is true when
        each team scored at least once.
    """
    return {
        "over_2_5": ((matches["home_goals"] + matches["away_goals"]) >= 3).to_numpy(dtype=int),
        "btts": ((matches["home_goals"] > 0) & (matches["away_goals"] > 0)).to_numpy(dtype=int),
    }


def _feature_columns(features: pd.DataFrame) -> list[str]:
    """Select numeric model columns while excluding the bookkeeping match id."""
    return [
        column
        for column in features.columns
        if column != "match_id" and pd.api.types.is_numeric_dtype(features[column])
    ]


def _impute_fit(frame: pd.DataFrame) -> tuple[SimpleImputer, np.ndarray]:
    """Fit median imputation on training features and return its transformed matrix."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer, imputer.fit_transform(frame)
