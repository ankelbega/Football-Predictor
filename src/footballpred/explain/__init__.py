"""Explain-stage utilities for global SHAP summaries and fixture explanations.

The explanations are called from evaluation artifact generation and API serving; SHAP values
are reported on raw model-score scale, not as probability points.
"""
