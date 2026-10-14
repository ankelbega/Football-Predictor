"""Compute global and local SHAP contributions for fitted LightGBM trees.

This explain-stage module is called by evaluation reports and text explanations. SHAP values
describe model raw-score contributions; they are not direct changes in normalized probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from footballpred.models.lgbm import LightGBM1X2


def compute_global_importance(
    model: LightGBM1X2,
    features: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Compute mean absolute per-class SHAP contribution over a deterministic sample.

    Args:
        model: Fitted LightGBM 1X2 classifier and training-fitted imputer.
        features: Causal feature rows to summarize.
        max_samples: Upper bound on rows explained; evenly spaced row indices keep the summary
            repeatable and reduce CPU/memory cost.

    Returns:
        Feature and mean absolute SHAP score table sorted from largest to smallest.
    """
    if model.model_ is None or model.imputer_ is None:
        raise RuntimeError("Fit LightGBM1X2 before requesting SHAP values")
    x = model.imputer_.transform(features[model.feature_columns_])
    if len(x) > max_samples:
        indices = np.linspace(0, len(x) - 1, max_samples, dtype=int)
        x = x[indices]
    explainer = shap.TreeExplainer(model.model_.booster_)
    # Mean absolute SHAP averages impact size across rows and each of the three class outputs.
    explanation = explainer.shap_values(x)
    if isinstance(explanation, list):
        values = np.stack(explanation, axis=0)
        mean_abs = np.abs(values).mean(axis=(0, 1))
    else:
        array = np.asarray(explanation)
        if array.ndim == 3:
            # LightGBM output shape may be (n, features, classes) or (classes, n, features).
            feature_axis = 1 if array.shape[1] == len(model.feature_columns_) else 2
            if feature_axis == 1:
                mean_abs = np.abs(array).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(array).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(array).mean(axis=0)
    return pd.DataFrame(
        {"feature": model.feature_columns_, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def save_global_importance(
    importance: pd.DataFrame,
    csv_path: str | Path,
    figure_path: str | Path,
    top_n: int = 20,
) -> None:
    """Write the SHAP table and a PNG bar chart for the highest-ranked features.

    Args:
        importance: Feature ranking returned by ``compute_global_importance``.
        csv_path: Destination CSV path.
        figure_path: Destination PNG path.
        top_n: Number of leading features to plot.

    Returns:
        None. Both parent directories are created if needed.
    """
    csv_destination = Path(csv_path)
    image_destination = Path(figure_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(csv_destination, index=False, float_format="%.8f")
    subset = importance.head(top_n).iloc[::-1]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, max(4, 0.28 * len(subset))))
    axis.barh(subset["feature"], subset["mean_abs_shap"], color="#2b6f9c")
    axis.set_xlabel("Mean absolute SHAP value (model output scale)")
    axis.set_title("LightGBM global feature importance")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(image_destination, dpi=160)
    plt.close(figure)


def local_shap_values(
    model: LightGBM1X2,
    one_row: pd.DataFrame,
    class_index: int = 0,
) -> tuple[np.ndarray, float]:
    """Return selected-class contributions and baseline raw score for one match.

    Args:
        model: Fitted LightGBM multiclass model.
        one_row: Single numeric feature row with the model's feature columns.
        class_index: H/D/A column index to explain.

    Returns:
        Feature contribution vector and SHAP expected value, both on model raw-score scale.
    """
    if model.model_ is None or model.imputer_ is None:
        raise RuntimeError("Fit LightGBM1X2 before requesting SHAP values")
    x = model.imputer_.transform(one_row[model.feature_columns_])
    explainer = shap.TreeExplainer(model.model_.booster_)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        vector = np.asarray(values[class_index])[0]
        base = float(np.asarray(explainer.expected_value)[class_index])
    else:
        array = np.asarray(values)
        if array.ndim == 3 and array.shape[1] == len(model.feature_columns_):
            vector = array[0, :, class_index]
        elif array.ndim == 3:
            vector = array[class_index, 0, :]
        else:
            vector = array[0]
        expected = np.asarray(explainer.expected_value)
        base = float(expected[class_index]) if expected.ndim else float(expected)
    return vector, base
